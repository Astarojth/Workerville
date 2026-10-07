from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from judge.schemas import ENDPOINTS


def load_rl_endpoint_map(rl_json_path: Path) -> dict[str, str]:
    if not rl_json_path.exists():
        return {}
    payload = json.loads(rl_json_path.read_text(encoding="utf-8"))
    items = payload.get("items", []) if isinstance(payload, dict) else []
    out: dict[str, str] = {}
    for row in items:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id", "")).strip().upper()
        endpoint = str(row.get("endpoint", "")).strip().upper()
        if rid and endpoint in ENDPOINTS:
            out[rid] = endpoint
    return out


def parse_applicability(
    cfg_payload: dict[str, Any],
    *,
    rl_endpoint_map: dict[str, str],
    override: dict[str, bool] | None = None,
) -> dict[str, Any]:
    rl_labels = _collect_rl_labels(cfg_payload)
    endpoints_from_rl = sorted({rl_endpoint_map[x] for x in rl_labels if x in rl_endpoint_map})
    unknown_rl = sorted([x for x in rl_labels if x not in rl_endpoint_map])

    if endpoints_from_rl:
        mask = {endpoint: endpoint in endpoints_from_rl for endpoint in ENDPOINTS}
        source = "rl"
    else:
        mask = {endpoint: True for endpoint in ENDPOINTS}
        source = "default_all"

    parsed_override = _normalize_override(override)
    if parsed_override:
        for endpoint, value in parsed_override.items():
            mask[endpoint] = value
        source = "rl_plus_override" if source == "rl" else "override"

    return {
        "mask": mask,
        "source": source,
        "rl_labels": rl_labels,
        "unknown_rl_labels": unknown_rl,
        "override": parsed_override,
    }


def _collect_rl_labels(cfg_payload: dict[str, Any]) -> list[str]:
    labels: set[str] = set()
    agents = cfg_payload.get("agents", []) if isinstance(cfg_payload, dict) else []
    if not isinstance(agents, list):
        return []
    for agent in agents:
        if not isinstance(agent, dict):
            continue
        metadata = agent.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        raw = metadata.get("task_category_rl")
        for label in _parse_rl_cell(raw):
            labels.add(label)
    return sorted(labels)


def _parse_rl_cell(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        tokens = [x.strip().upper() for x in raw.replace(";", ",").split(",")]
        return [x for x in tokens if x]
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            text = str(item).strip().upper()
            if text:
                out.append(text)
        return out
    return []


def _normalize_override(raw: dict[str, Any] | None) -> dict[str, bool]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, bool] = {}
    for endpoint in ENDPOINTS:
        if endpoint not in raw:
            continue
        value = _to_bool(raw.get(endpoint))
        if value is None:
            continue
        out[endpoint] = value
    return out


def _to_bool(raw: Any) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if raw in {0, 1}:
        return bool(raw)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None
