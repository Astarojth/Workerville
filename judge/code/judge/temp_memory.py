from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from judge.endpoint_specs import ENDPOINT_SPECS

ENDPOINTS = tuple(ENDPOINT_SPECS.keys())
_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def build_temporary_memory_bundle(
    *,
    project_root: Path,
    positive_limit: int,
    negative_limit: int,
    source_root_override: str | None = None,
) -> dict[str, Any]:
    source_root = _resolve_source_root(project_root=project_root, source_root_override=source_root_override)
    cache_key = (str(source_root), int(positive_limit), int(negative_limit))
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    records = _collect_run_records(source_root)
    banks: dict[str, dict[str, Any]] = {}
    for endpoint in ENDPOINTS:
        positives = [row for row in records if bool(row["verdicts"].get(endpoint, False))]
        negatives = [
            row
            for row in records
            if not bool(row["verdicts"].get(endpoint, False)) and any(bool(v) for k, v in row["verdicts"].items() if k != endpoint)
        ]
        selected = _select_records(positives, limit=positive_limit) + _select_records(negatives, limit=negative_limit)
        banks[endpoint] = {
            "endpoint": endpoint,
            "source_root": str(source_root),
            "entries": [_record_to_entry(record=row, endpoint=endpoint) for row in selected],
        }

    bundle = {
        "schema": "workerville_temp_memory_bundle_v1",
        "source_root": str(source_root),
        "stats": {
            "run_records": len(records),
            "positive_limit": int(positive_limit),
            "negative_limit": int(negative_limit),
            "entries_per_endpoint": {endpoint: len(banks[endpoint]["entries"]) for endpoint in ENDPOINTS},
        },
        "banks": banks,
    }
    _CACHE[cache_key] = bundle
    return bundle


def load_static_memory_bundle(
    *,
    project_root: Path,
    partition_root: str | None = None,
    partition_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    root = _resolve_partition_root(project_root=project_root, partition_root=partition_root)
    files = {
        "S1": "partition_01_151_clusters.json",
        "S2": "partition_01_149_clusters.json",
        "S3": "partition_01_152_clusters.json",
    }
    if isinstance(partition_files, dict):
        for endpoint in ENDPOINTS:
            text = str(partition_files.get(endpoint, "")).strip()
            if text:
                files[endpoint] = text

    cache_key = ("static", str(root), files["S1"], files["S2"], files["S3"])
    cached = _CACHE.get(cache_key)  # type: ignore[arg-type]
    if cached is not None:
        return cached

    banks: dict[str, dict[str, Any]] = {}
    for endpoint in ENDPOINTS:
        path = root / endpoint / files[endpoint]
        rows = _load_json(path)
        entries = [_static_candidate_to_entry(endpoint=endpoint, row=row) for row in rows if isinstance(row, dict)]
        banks[endpoint] = {
            "endpoint": endpoint,
            "source_root": str(root),
            "source_file": str(path),
            "entries": entries,
        }

    bundle = {
        "schema": "workerville_static_memory_bundle_v1",
        "source_root": str(root),
        "stats": {
            "entries_per_endpoint": {endpoint: len(banks[endpoint]["entries"]) for endpoint in ENDPOINTS},
            "source_files": {endpoint: banks[endpoint]["source_file"] for endpoint in ENDPOINTS},
        },
        "banks": banks,
    }
    _CACHE[cache_key] = bundle  # type: ignore[index]
    return bundle


def load_packaged_review_memory_bundle(
    *,
    project_root: Path,
    memory_root: str | None = None,
) -> dict[str, Any]:
    root = _resolve_packaged_review_root(project_root=project_root, memory_root=memory_root)
    cache_key = ("packaged_reviews", str(root))
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    banks: dict[str, dict[str, Any]] = {}
    for endpoint in ENDPOINTS:
        review_dir = root / endpoint / "reviews"
        entries = []
        for path in sorted(review_dir.glob("*.json")):
            row = _load_json_object(path)
            entries.append(_packaged_review_to_entry(endpoint=endpoint, row=row, path=path))
        banks[endpoint] = {
            "endpoint": endpoint,
            "source_root": str(root),
            "source_file": str(review_dir),
            "entries": entries,
        }

    bundle = {
        "schema": "workerville_packaged_review_memory_bundle_v1",
        "source_root": str(root),
        "stats": {
            "entries_per_endpoint": {endpoint: len(banks[endpoint]["entries"]) for endpoint in ENDPOINTS},
            "source_dirs": {endpoint: banks[endpoint]["source_file"] for endpoint in ENDPOINTS},
        },
        "banks": banks,
    }
    _CACHE[cache_key] = bundle
    return bundle


def build_endpoint_query_text(
    *,
    endpoint: str,
    cfg_payload: dict[str, Any],
    logs: list[dict[str, Any]],
    metrics: dict[str, Any],
    issue_rows: list[dict[str, Any]],
) -> str:
    task = cfg_payload.get("task", {}) if isinstance(cfg_payload.get("task", {}), dict) else {}
    task_id = str(task.get("task_id", "")).strip()
    instruction = str(task.get("instruction", "")).strip()
    evidence = _endpoint_evidence_lines(endpoint=endpoint, metrics=metrics, issue_rows=issue_rows, logs=logs)
    parts = [
        f"endpoint={endpoint}",
        f"task_id={task_id}",
        f"instruction={instruction[:280]}",
        "trajectory=" + _trajectory_excerpt(logs),
        "evidence=" + " || ".join(evidence[:6]),
    ]
    return "\n".join([part for part in parts if part.strip()])


def _resolve_source_root(*, project_root: Path, source_root_override: str | None) -> Path:
    if str(source_root_override or "").strip():
        src = Path(str(source_root_override).strip()).expanduser()
        if not src.is_absolute():
            src = (project_root / src).resolve()
        return src
    preferred = (project_root.parent / "run").resolve()
    if preferred.exists() and preferred.is_dir():
        return preferred
    fallback = (project_root / "runs").resolve()
    if fallback.exists() and fallback.is_dir():
        return fallback
    raise FileNotFoundError("no run source root found for temporary judge memory")


def _resolve_partition_root(*, project_root: Path, partition_root: str | None) -> Path:
    if str(partition_root or "").strip():
        src = Path(str(partition_root).strip()).expanduser()
        if not src.is_absolute():
            src = (project_root / src).resolve()
        return src
    default_root = (project_root / "runs" / "memory_finch_full").resolve()
    if default_root.exists() and default_root.is_dir():
        return default_root
    raise FileNotFoundError("no static partition root found for v0 memory bank")


def _resolve_packaged_review_root(*, project_root: Path, memory_root: str | None) -> Path:
    if str(memory_root or "").strip():
        src = Path(str(memory_root).strip()).expanduser()
        if not src.is_absolute():
            src = (project_root / src).resolve()
        return src
    for name in ("memory_bank",):
        candidate = (project_root / name).resolve()
        if (candidate / "S1" / "reviews").is_dir():
            return candidate
    return (project_root / "memory_bank").resolve()


def _collect_run_records(source_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for metrics_path in sorted(source_root.rglob("analysis/metrics.json")):
        run_dir = metrics_path.parent.parent
        experiment_path = run_dir / "config" / "experiment.json"
        logs_path = run_dir / "logs" / "transcripts.pretty.json"
        issues_path = run_dir / "analysis" / "issues.jsonl"
        if not experiment_path.exists() or not logs_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            cfg_payload = json.loads(experiment_path.read_text(encoding="utf-8"))
            logs = json.loads(logs_path.read_text(encoding="utf-8"))
            issues = _load_jsonl(issues_path)
        except Exception:
            continue
        if not isinstance(metrics, dict) or not isinstance(cfg_payload, dict) or not isinstance(logs, list):
            continue
        task = cfg_payload.get("task", {}) if isinstance(cfg_payload.get("task", {}), dict) else {}
        task_id = str(task.get("task_id", "")).strip() or run_dir.parent.name
        model_source = run_dir.parents[1].name if len(run_dir.parents) >= 3 else "unknown"
        task_family = task_id.split("_", 1)[0] if "_" in task_id else task_id
        verdicts = {endpoint: _rule_endpoint_count(metrics, endpoint) > 0 for endpoint in ENDPOINTS}
        records.append(
            {
                "run_dir": str(run_dir),
                "episode_id": str(cfg_payload.get("episode_id", "")).strip(),
                "task_id": task_id,
                "task_family": task_family,
                "model_source": model_source,
                "cfg_payload": cfg_payload,
                "metrics": metrics,
                "logs": logs,
                "issue_rows": issues,
                "verdicts": verdicts,
            }
        )
    return records


def _select_records(records: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_model[str(row.get("model_source", "unknown"))].append(row)
    selected: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_tasks: set[str] = set()

    for model_name in sorted(by_model):
        for row in sorted(by_model[model_name], key=lambda item: str(item.get("run_dir", ""))):
            run_dir = str(row.get("run_dir", ""))
            task_id = str(row.get("task_id", ""))
            if run_dir in seen_paths:
                continue
            if task_id in seen_tasks and len(selected) < limit // 2:
                continue
            selected.append(row)
            seen_paths.add(run_dir)
            seen_tasks.add(task_id)
            break
        if len(selected) >= limit:
            return selected[:limit]

    for row in sorted(records, key=lambda item: str(item.get("run_dir", ""))):
        run_dir = str(row.get("run_dir", ""))
        if run_dir in seen_paths:
            continue
        selected.append(row)
        seen_paths.add(run_dir)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _record_to_entry(*, record: dict[str, Any], endpoint: str) -> dict[str, Any]:
    verdict = 1 if bool(record["verdicts"].get(endpoint, False)) else 0
    evidence = _endpoint_evidence_lines(
        endpoint=endpoint,
        metrics=record["metrics"],
        issue_rows=record["issue_rows"],
        logs=record["logs"],
    )
    trajectory = _trajectory_excerpt(record["logs"])
    task_id = str(record.get("task_id", ""))
    model_source = str(record.get("model_source", "unknown"))
    analysis = (
        f"{endpoint} positive exemplar. "
        f"Task={task_id}. Model={model_source}. "
        f"Evidence shows endpoint-specific risky behavior."
        if verdict
        else f"{endpoint} negative/boundary exemplar. Task={task_id}. Model={model_source}. "
        "Endpoint-specific evidence is absent even though nearby risk signals may exist."
    )
    return {
        "id": f"{endpoint}::{model_source}::{task_id}::{Path(str(record.get('run_dir', ''))).name}",
        "endpoint": endpoint,
        "episode_id": str(record.get("episode_id", "")).strip(),
        "label": verdict,
        "task_id": task_id,
        "task_family": str(record.get("task_family", "")),
        "model_source": model_source,
        "run_dir": str(record.get("run_dir", "")),
        "analysis": analysis,
        "evidence": evidence[:6],
        "trajectory": trajectory,
        "text": analysis,
        "embedding_text": "\n".join(
            [
                f"endpoint={endpoint}",
                f"label={verdict}",
                f"task_id={task_id}",
                f"task_family={record.get('task_family', '')}",
                f"model_source={model_source}",
                "trajectory=" + trajectory,
                "evidence=" + " || ".join(evidence[:6]),
            ]
        ),
    }


def _endpoint_evidence_lines(
    *,
    endpoint: str,
    metrics: dict[str, Any],
    issue_rows: list[dict[str, Any]],
    logs: list[dict[str, Any]],
) -> list[str]:
    risk_to_s = metrics.get("risk_to_s_mapping", {}) if isinstance(metrics, dict) else {}
    if not isinstance(risk_to_s, dict):
        risk_to_s = {}

    evidence: list[str] = []
    for row in issue_rows:
        if not isinstance(row, dict):
            continue
        issue_types = row.get("issue_types", [])
        if not isinstance(issue_types, list):
            continue
        mapped = False
        for risk in issue_types:
            endpoints = risk_to_s.get(str(risk), [])
            if isinstance(endpoints, list) and endpoint in endpoints:
                mapped = True
                break
        if mapped:
            action = str(row.get("action", "")).strip()
            if action:
                evidence.append(action[:220])
    if evidence:
        return evidence

    metric_events = metrics.get(_endpoint_metric_key(endpoint), {}).get("events", []) if isinstance(metrics.get(_endpoint_metric_key(endpoint), {}), dict) else []
    if isinstance(metric_events, list):
        for item in metric_events[:6]:
            if isinstance(item, dict):
                evidence.append(json.dumps(item, ensure_ascii=False)[:220])
    if evidence:
        return evidence

    for row in logs[:6]:
        if not isinstance(row, dict):
            continue
        action = str(row.get("action", "")).strip()
        if action:
            evidence.append(action[:220])
    return evidence or [f"no_direct_{endpoint.lower()}_evidence"]


def _trajectory_excerpt(logs: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for row in logs[:4]:
        if not isinstance(row, dict):
            continue
        action = str(row.get("action", "")).strip()
        if action:
            rows.append(action[:220])
    return " || ".join(rows) if rows else "no_trajectory_excerpt"


def _endpoint_metric_key(endpoint: str) -> str:
    return {
        "S1": "S1_leakage",
        "S2": "S2_high_risk_action",
        "S3": "S3_negative_compliance",
    }.get(endpoint, endpoint)


def _rule_endpoint_count(metrics: dict[str, Any], endpoint: str) -> int:
    s_endpoints = metrics.get("S_endpoints", {}) if isinstance(metrics, dict) else {}
    if isinstance(s_endpoints, dict):
        row = s_endpoints.get(endpoint, {}) if isinstance(s_endpoints.get(endpoint, {}), dict) else {}
        try:
            return int(row.get("count", 0))
        except Exception:
            return 0
    totals = metrics.get("totals", {}) if isinstance(metrics, dict) else {}
    if isinstance(totals, dict):
        try:
            return int(totals.get(endpoint, 0))
        except Exception:
            return 0
    return 0


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _load_json(path: Path) -> list[Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected list json at {path}")
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object json at {path}")
    return payload


def _static_candidate_to_entry(*, endpoint: str, row: dict[str, Any]) -> dict[str, Any]:
    candidate_text = str(row.get("text", "")).strip()
    instruction = str(row.get("instruction", "")).strip()
    analysis = (
        f"{endpoint} v0 memory candidate. "
        f"Task={str(row.get('task_id', '')).strip()}. "
        "Representative cluster exemplar retained from FINCH partition_01."
    )
    evidence = [candidate_text[:220]] if candidate_text else []
    trajectory = instruction[:220] if instruction else "no_instruction_excerpt"
    return {
        "id": str(row.get("id", "")).strip() or f"{endpoint}::unknown",
        "endpoint": endpoint,
        "episode_id": str(row.get("episode_id", "")).strip(),
        "label": 1,
        "task_id": str(row.get("task_id", "")).strip(),
        "task_family": str(row.get("task_family", "")).strip(),
        "model_source": str(row.get("source_model", "")).strip(),
        "run_dir": str(row.get("run_dir", "")).strip(),
        "analysis": analysis,
        "evidence": evidence,
        "trajectory": trajectory,
        "text": candidate_text or analysis,
        "embedding_text": "\n".join(
            [
                f"endpoint={endpoint}",
                f"task_id={str(row.get('task_id', '')).strip()}",
                f"task_family={str(row.get('task_family', '')).strip()}",
                f"source_model={str(row.get('source_model', '')).strip()}",
                f"instruction={instruction[:280]}",
                f"candidate_text={candidate_text[:600]}",
            ]
        ).strip(),
    }


def _packaged_review_to_entry(*, endpoint: str, row: dict[str, Any], path: Path) -> dict[str, Any]:
    candidate = row.get("candidate", {}) if isinstance(row.get("candidate", {}), dict) else {}
    payload = row.get("payload", {}) if isinstance(row.get("payload", {}), dict) else {}
    payload_candidate = payload.get("candidate", {}) if isinstance(payload.get("candidate", {}), dict) else {}
    reviews = row.get("reviews", {}) if isinstance(row.get("reviews", {}), dict) else {}
    chair = reviews.get("chair", {}) if isinstance(reviews.get("chair", {}), dict) else {}
    chair_result = chair.get("result", {}) if isinstance(chair.get("result", {}), dict) else {}

    label = _parse_packaged_label(chair_result.get("final_label"))
    if label is None:
        label = _parse_packaged_label(payload.get("mechanical_label"))
    if label is None:
        label = 1 if _safe_int(candidate.get("rule_count", payload.get("rule_count", 0))) > 0 else 0

    task_id = str(candidate.get("task_id", payload_candidate.get("task_id", ""))).strip()
    instruction = str(candidate.get("instruction", payload_candidate.get("instruction", ""))).strip()
    candidate_text = str(candidate.get("text", payload_candidate.get("candidate_text", ""))).strip()
    reason = str(chair_result.get("reason", "")).strip()
    notes = chair_result.get("notes", [])
    evidence = _to_str_list(chair_result.get("key_evidence", [])) or _to_str_list(notes)
    if not evidence:
        evidence = [candidate_text[:220]] if candidate_text else []
    trajectory = _packaged_trajectory(payload=payload, candidate_text=candidate_text, instruction=instruction)
    analysis = reason or f"{endpoint} packaged review exemplar. Final label={label}. Task={task_id or 'unknown'}."

    return {
        "id": str(candidate.get("id", payload_candidate.get("id", path.stem))).strip(),
        "endpoint": endpoint,
        "episode_id": str(candidate.get("episode_id", payload_candidate.get("episode_id", ""))).strip(),
        "label": int(label),
        "task_id": task_id,
        "task_family": str(candidate.get("task_family", payload_candidate.get("task_family", ""))).strip(),
        "model_source": str(candidate.get("source_model", payload_candidate.get("source_model", ""))).strip(),
        "analysis": analysis,
        "evidence": evidence[:6],
        "trajectory": trajectory,
        "text": candidate_text or analysis,
        "embedding_text": "\n".join(
            [
                f"endpoint={endpoint}",
                f"label={int(label)}",
                f"task_id={task_id}",
                f"instruction={instruction[:280]}",
                f"analysis={analysis[:800]}",
                "trajectory=" + trajectory,
                "evidence=" + " || ".join(evidence[:6]),
            ]
        ).strip(),
    }


def _parse_packaged_label(raw: Any) -> int | None:
    if isinstance(raw, bool):
        return int(raw)
    if raw in {0, 1}:
        return int(raw)
    text = str(raw).strip()
    if text in {"0", "1"}:
        return int(text)
    return None


def _safe_int(raw: Any) -> int:
    try:
        return int(raw)
    except Exception:
        return 0


def _to_str_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        text = raw.strip()
        return [text] if text else []
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return []


def _packaged_trajectory(*, payload: dict[str, Any], candidate_text: str, instruction: str) -> str:
    rows = payload.get("trajectory_excerpt", [])
    if isinstance(rows, list):
        actions: list[str] = []
        for row in rows[:4]:
            if isinstance(row, dict):
                action = str(row.get("action", "")).strip()
                if action:
                    actions.append(action[:220])
        if actions:
            return " || ".join(actions)
    if candidate_text:
        marker = "trajectory="
        start = candidate_text.find(marker)
        if start >= 0:
            tail = candidate_text[start + len(marker) :]
            end = tail.find("\n")
            return (tail if end < 0 else tail[:end]).strip()[:900]
    return instruction[:220] if instruction else "no_trajectory_excerpt"
