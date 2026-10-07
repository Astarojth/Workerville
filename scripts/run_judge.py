#!/usr/bin/env python3
"""Judge existing episodes with user-supplied chat and embedding models."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "judge" / "code"
JUDGE_ROOT = ROOT / "judge"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WORKERVILLE_EMBED_CACHE_DIR", str(ROOT / "results" / "embed_cache"))

def _neutralize_broken_sklearn_hook() -> None:
    sys.meta_path = [finder for finder in sys.meta_path if type(finder).__name__ != "MesonpyMetaFinder"]


_neutralize_broken_sklearn_hook()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Judge existing Workerville episodes with packaged memory.")
    p.add_argument("--families", default="", help="Comma-separated families, or 'all'. Required.")
    p.add_argument("--inventory", default=str(ROOT / "results" / "inventory" / "canonical_index.csv"))
    p.add_argument(
        "--api-config",
        default=str(ROOT / "api.config.yaml"),
        help="OpenAI-compatible API file. Default api.config.yaml is empty in the pack.",
    )
    p.add_argument("--model", default="", help="Override model from api.config.yaml.")
    p.add_argument(
        "--embedding-model",
        default="",
        help="Embedding model identifier supplied by the user; no default is published.",
    )
    p.add_argument("--memory-bank-mode", default="packaged_reviews")
    p.add_argument(
        "--memory-root",
        default=str(JUDGE_ROOT / "memory_bank"),
        help="Packaged reviews root. Default is the official judge/memory_bank.",
    )
    p.add_argument("--memory-top-k", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=256000)
    p.add_argument("--mode", default="llm_only", choices=("llm_only", "llm_with_fallback", "rule_only"))
    p.add_argument("--endpoints", default="S1,S2,S3")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--limit-per-family", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--out-dir",
        default=str(ROOT / "results" / "judge"),
        help="Judge output tree. The published tree is not overwritten unless --overwrite.",
    )
    return p.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_src(raw: str) -> Path:
    text = str(raw or "").strip()
    if not text:
        return Path()
    path = Path(text)
    if not path.is_absolute():
        return (ROOT / path).resolve()
    marker = "/runs/"
    if marker in text:
        return (ROOT / text[text.index("runs/") :]).resolve()
    return path


def _load_inventory(path: Path, families: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            family = str(row.get("family", "")).strip()
            if families and family not in families:
                continue
            src = str(row.get("src_path", "")).strip()
            if not src:
                continue
            item = dict(row)
            item["family"] = family
            item["src_path"] = str(_resolve_src(src))
            rows.append(item)
    rows.sort(
        key=lambda r: (
            str(r.get("family", "")),
            str(r.get("task_id", "")),
            str(r.get("config_id", "")),
        )
    )
    return rows


def _apply_limits(rows: list[dict[str, str]], *, limit: int, limit_per_family: int) -> list[dict[str, str]]:
    if limit_per_family > 0:
        counts: dict[str, int] = {}
        selected: list[dict[str, str]] = []
        for row in rows:
            family = str(row.get("family", ""))
            if counts.get(family, 0) >= limit_per_family:
                continue
            counts[family] = counts.get(family, 0) + 1
            selected.append(row)
        rows = selected
    if limit > 0:
        rows = rows[:limit]
    return rows


def _required_files(src: Path) -> dict[str, Path]:
    return {
        "experiment": src / "config" / "experiment.json",
        "logs": src / "logs" / "transcripts.pretty.json",
        "metrics": src / "analysis" / "metrics.json",
        "issues": src / "analysis" / "issues.jsonl",
    }


def _is_complete(src: Path) -> bool:
    return all(path.is_file() for path in _required_files(src).values())


def _already_judged(src: Path) -> bool:
    path = src / "analysis" / "metrics_judge.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(payload, dict) and bool(payload)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"json object expected: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        item = json.loads(text)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        fh.write(line)
        fh.flush()


def _parse_endpoints(raw: str) -> list[str]:
    names = [part.strip().upper() for part in str(raw or "").split(",") if part.strip()]
    return names or ["S1", "S2", "S3"]


def _configure_judge(cfg_payload: dict[str, Any], *, args_dict: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(cfg_payload)
    runtime = out.get("runtime", {})
    if not isinstance(runtime, dict):
        runtime = {}
    out["runtime"] = runtime
    judge_cfg = runtime.get("judge", {})
    if not isinstance(judge_cfg, dict):
        judge_cfg = {}
    runtime["judge"] = judge_cfg
    judge_cfg["enabled"] = True
    judge_cfg["mode"] = str(args_dict["mode"])
    judge_cfg["parser_repair"] = True
    judge_cfg["max_tokens"] = max(256, int(args_dict["max_tokens"]))
    judge_cfg["memory_bank_mode"] = str(args_dict["memory_bank_mode"])
    judge_cfg["memory_top_k"] = max(1, int(args_dict["memory_top_k"]))
    judge_cfg["memory_partition_root"] = str(args_dict["memory_root"])
    judge_cfg["api_config_file"] = str(args_dict["api_config"])
    judge_cfg["model_override"] = str(args_dict["model"])
    judge_cfg["embedding_model"] = str(args_dict["embedding_model"])
    judge_cfg["endpoints"] = list(args_dict["endpoints"])
    return out


def _s_counts(metrics_judge: dict[str, Any] | None) -> dict[str, int]:
    counts = (metrics_judge or {}).get("counts", {}) if isinstance(metrics_judge, dict) else {}
    if not isinstance(counts, dict):
        counts = {}
    return {key: int(counts.get(key, 0) or 0) for key in ("S1", "S2", "S3")}


def _warmup_memory_embeddings(memory_root: str, embedding_model: str) -> int:
    from judge.embedding_retrieval import EmbeddingRetriever
    from judge.temp_memory import load_packaged_review_memory_bundle

    retriever = EmbeddingRetriever(model_name=embedding_model, matryoshka_dim=512)
    bundle = load_packaged_review_memory_bundle(
        project_root=JUDGE_ROOT,
        memory_root=memory_root,
    )
    n = 0
    banks = bundle.get("banks", {}) if isinstance(bundle, dict) else {}
    for endpoint in ("S1", "S2", "S3"):
        entries = banks.get(endpoint, {}).get("entries", []) if isinstance(banks.get(endpoint, {}), dict) else []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("embedding_text", "") or entry.get("text", "")).strip()
            if not text:
                continue
            retriever._encode_text(text)
            n += 1
    return n


def _worker_init(args_dict: dict[str, Any]) -> None:
    _neutralize_broken_sklearn_hook()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WORKERVILLE_EMBED_CACHE_DIR", str(ROOT / "results" / "embed_cache"))
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("WORKERVILLE_EMBED_THREADS", "4")
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    _warmup_memory_embeddings(str(args_dict["memory_root"]), str(args_dict["embedding_model"]))


def _judge_one(payload: dict[str, Any]) -> dict[str, Any]:
    _neutralize_broken_sklearn_hook()
    from benchmark_core.runtime import load_api_file, load_api_settings
    from judge import run_judge_postprocess

    src = Path(payload["src_path"])
    args_dict = payload["args"]
    started = time.time()
    row: dict[str, Any] = {
        "family": payload.get("family", ""),
        "task_id": payload.get("task_id", ""),
        "config_id": payload.get("config_id", ""),
        "src_path": str(src),
        "started_at_utc": _utc_now(),
    }
    try:
        if not args_dict.get("overwrite") and _already_judged(src):
            row.update({"status": "skip", "elapsed_sec": round(time.time() - started, 3)})
            return row
        if not _is_complete(src):
            missing = [name for name, path in _required_files(src).items() if not path.is_file()]
            row.update({"status": "incomplete", "missing": missing, "elapsed_sec": round(time.time() - started, 3)})
            return row

        files = _required_files(src)
        cfg_payload = _load_json(files["experiment"])
        logs = json.loads(files["logs"].read_text(encoding="utf-8"))
        if not isinstance(logs, list):
            raise ValueError("logs list expected")
        metrics = _load_json(files["metrics"])
        issue_rows = _load_jsonl(files["issues"])
        judged_cfg = _configure_judge(cfg_payload, args_dict=args_dict)
        api = load_api_settings(load_api_file(str(args_dict["api_config"])))
        model = str(args_dict.get("model") or "").strip() or str(api.model or "").strip()
        if model and not str(args_dict.get("model") or "").strip():
            local_args = dict(args_dict)
            local_args["model"] = model
            judged_cfg = _configure_judge(cfg_payload, args_dict=local_args)
            args_dict = local_args

        summary = run_judge_postprocess(
            out_dir=src,
            project_root=JUDGE_ROOT,
            cfg_payload=judged_cfg,
            logs=logs,
            metrics=metrics,
            issue_rows=issue_rows,
            base_api=api,
            judge_enabled_override=True,
            judge_model_override=model or None,
        )
        metrics_judge = summary.get("metrics_judge") if isinstance(summary, dict) else None
        parse_status = ""
        judge_result_path = src / "analysis" / "judge_result.json"
        if judge_result_path.is_file():
            judge_result = _load_json(judge_result_path)
            meta = judge_result.get("meta", {}) if isinstance(judge_result.get("meta", {}), dict) else {}
            parse_meta = meta.get("parse", {}) if isinstance(meta.get("parse", {}), dict) else {}
            bits = []
            for endpoint in ("S1", "S2", "S3"):
                route = parse_meta.get(endpoint, {})
                if isinstance(route, dict) and route.get("status"):
                    bits.append(f"{endpoint}:{route.get('status')}")
            parse_status = ",".join(bits)
        row.update(
            {
                "status": "ok",
                "parse_status": parse_status,
                "counts": _s_counts(metrics_judge if isinstance(metrics_judge, dict) else None),
                "elapsed_sec": round(time.time() - started, 3),
            }
        )
        return row
    except Exception as exc:
        row.update(
            {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "elapsed_sec": round(time.time() - started, 3),
            }
        )
        return row


def _read_progress(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
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


def _latest_by_src(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("src_path", "")).strip() or f"row-{len(latest)}"
        latest[key] = row
    return list(latest.values())


def _write_status(out_dir: Path, rows: list[dict[str, Any]], *, extra: dict[str, Any]) -> None:
    from collections import Counter, defaultdict

    rows = _latest_by_src(rows)
    status_counts = Counter(str(row.get("status", "")) for row in rows)
    family_ok: dict[str, int] = defaultdict(int)
    family_err: dict[str, int] = defaultdict(int)
    pos = {key: 0 for key in ("S1", "S2", "S3")}
    judged = 0
    for row in rows:
        family = str(row.get("family", "unknown"))
        if row.get("status") == "ok":
            family_ok[family] += 1
            judged += 1
            counts = row.get("counts", {}) if isinstance(row.get("counts"), dict) else {}
            for key in ("S1", "S2", "S3"):
                if int(counts.get(key, 0) or 0) > 0:
                    pos[key] += 1
        elif row.get("status") == "error":
            family_err[family] += 1
    payload = {
        "updated_at_utc": _utc_now(),
        "status_counts": dict(status_counts),
        "ok_by_family": dict(family_ok),
        "error_by_family": dict(family_err),
        "positive_runs": pos,
        "judged_ok": judged,
        **extra,
    }
    (out_dir / "STATUS.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Judge packaged memory",
        "",
        f"Updated: {payload['updated_at_utc']}",
        "",
        f"- queued: {extra.get('queued', '?')} (pending {extra.get('pending', extra.get('queued', '?'))}, already_judged {extra.get('already_judged', 0)})",
        f"- status: `{status_counts}`",
        f"- ok by family: `{dict(family_ok)}`",
        f"- errors by family: `{dict(family_err)}`",
        f"- judge-positive among ok: S1={pos['S1']} S2={pos['S2']} S3={pos['S3']} / {judged}",
        "",
        "Episode artifacts: `<run>/analysis/metrics_judge.json`. Progress: `progress.jsonl`.",
        "",
    ]
    (out_dir / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")


def _load_api_or_die(api_config: str) -> None:
    from benchmark_core.runtime import load_api_file, load_api_settings

    api = load_api_settings(load_api_file(api_config))
    issues = api.validate()
    if issues:
        print("[judge] api.config.yaml is empty or incomplete:", file=sys.stderr)
        for item in issues:
            print(f"  - {item}", file=sys.stderr)
        raise SystemExit(2)


def main() -> int:
    args = parse_args()
    os.environ.setdefault("WORKERVILLE_EMBED_CACHE_DIR", str(ROOT / "results" / "embed_cache"))
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    families_raw = str(args.families).strip()
    if not families_raw:
        print("[judge] pass --families (comma-separated) or --families all", file=sys.stderr)
        return 2
    if families_raw.lower() == "all":
        families: set[str] = set()
    else:
        families = {item.strip() for item in families_raw.split(",") if item.strip()}
    if not families and families_raw.lower() != "all":
        print("[judge] no families given", file=sys.stderr)
        return 2

    embedding_model = str(args.embedding_model).strip()
    if not embedding_model:
        print("[judge] pass --embedding-model", file=sys.stderr)
        return 2

    _load_api_or_die(str(args.api_config))
    rows = _load_inventory(Path(args.inventory), families)
    rows = _apply_limits(rows, limit=int(args.limit), limit_per_family=int(args.limit_per_family))
    inventory_n = len(rows)
    already = 0
    if not args.overwrite:
        pending: list[dict[str, str]] = []
        for row in rows:
            src = Path(str(row.get("src_path", "")).strip())
            if _already_judged(src):
                already += 1
                continue
            pending.append(row)
        rows = pending
        print(f"[judge] inventory={inventory_n} already_judged={already} pending={len(rows)}")
    if not rows:
        print("[judge] no inventory rows matched")
        return 2
    (out_dir / "run.pid").write_text(str(os.getpid()) + "\n", encoding="utf-8")

    args_dict = {
        "api_config": str(Path(args.api_config).resolve()),
        "model": str(args.model).strip(),
        "embedding_model": embedding_model,
        "memory_bank_mode": str(args.memory_bank_mode),
        "memory_root": str(Path(args.memory_root).resolve()),
        "memory_top_k": int(args.memory_top_k),
        "max_tokens": int(args.max_tokens),
        "mode": str(args.mode),
        "endpoints": _parse_endpoints(args.endpoints),
        "overwrite": bool(args.overwrite),
    }
    run_config = {
        "schema": "workerville_judge_run_v1",
        "created_at_utc": _utc_now(),
        "model": args_dict["model"],
        "api_config": args_dict["api_config"],
        "memory_bank_mode": args.memory_bank_mode,
        "memory_root": args_dict["memory_root"],
        "memory_top_k": args.memory_top_k,
        "max_tokens": args.max_tokens,
        "mode": args.mode,
        "endpoints": args_dict["endpoints"],
        "workers": int(args.workers),
        "families": sorted(families),
        "queued": inventory_n,
        "pending": len(rows),
        "already_judged": already,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    progress_path = out_dir / "progress.jsonl"
    status_extra = {
        "queued": inventory_n,
        "pending": len(rows),
        "already_judged": already,
    }
    print(
        f"[judge] queued={len(rows)}/{inventory_n} workers={args.workers} "
        f"mode={args.mode} memory={args.memory_bank_mode} out={out_dir}"
    )
    print("[judge] warming memory embeddings...")
    warmed = _warmup_memory_embeddings(args_dict["memory_root"], args_dict["embedding_model"])
    print(f"[judge] memory embeddings ready n={warmed}")

    work = [
        {
            "family": row.get("family", ""),
            "task_id": row.get("task_id", ""),
            "config_id": row.get("config_id", ""),
            "src_path": row.get("src_path", ""),
            "args": args_dict,
        }
        for row in rows
    ]
    done_rows: list[dict[str, Any]] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        _worker_init(args_dict)
        for item in work:
            result = _judge_one(item)
            done_rows.append(result)
            _append_jsonl(progress_path, result)
            print(
                f"[judge] {result.get('status')} "
                f"{result.get('family')}/{result.get('task_id')}/{result.get('config_id')} "
                f"parse={result.get('parse_status', '')} "
                f"sec={result.get('elapsed_sec')}"
            )
            _write_status(out_dir, _read_progress(progress_path), extra=status_extra)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(args_dict,)) as pool:
            futures = [pool.submit(_judge_one, item) for item in work]
            for fut in as_completed(futures):
                result = fut.result()
                done_rows.append(result)
                _append_jsonl(progress_path, result)
                n = len(done_rows)
                if n <= 8 or n % 10 == 0 or result.get("status") == "error":
                    print(
                        f"[judge] {n}/{len(work)} {result.get('status')} "
                        f"{result.get('family')}/{result.get('config_id')} "
                        f"sec={result.get('elapsed_sec')} err={result.get('error', '')[:160]}"
                    )
                if n % 5 == 0 or n == len(work) or result.get("status") == "error":
                    _write_status(out_dir, _read_progress(progress_path), extra=status_extra)
        _write_status(out_dir, _read_progress(progress_path), extra=status_extra)

    ok = sum(1 for row in done_rows if row.get("status") == "ok")
    err = sum(1 for row in done_rows if row.get("status") == "error")
    print(
        f"[judge] done ok={ok} error={err} "
        f"skip={sum(1 for r in done_rows if r.get('status')=='skip')} total={len(done_rows)}"
    )
    return 0 if err == 0 else 1


if __name__ == "__main__":
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    raise SystemExit(main())
