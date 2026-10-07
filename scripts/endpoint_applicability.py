"""Task-taxonomy applicability for published S rates.

An endpoint is applicable only when `task_category_rl` maps to it.
NA is out of scope: drop it from both numerator and denominator.
The mask is a task property (identical across C and families).

Standalone copy of the Judge mask rule so reporting does not import
`judge.pipeline` (numpy / embed stack).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
S_KEYS = ("S1", "S2", "S3")


@lru_cache(maxsize=1)
def rl_endpoint_map() -> dict[str, str]:
    path = ROOT / "configs" / "taxonomy" / "rl.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items", []) if isinstance(payload, dict) else []
    out: dict[str, str] = {}
    for row in items:
        if not isinstance(row, dict):
            continue
        rid = str(row.get("id", "")).strip().upper()
        endpoint = str(row.get("endpoint", "")).strip().upper()
        if rid and endpoint in S_KEYS:
            out[rid] = endpoint
    return out


def mask_for_experiment(cfg_payload: dict[str, Any]) -> dict[str, bool]:
    labels = _collect_rl_labels(cfg_payload)
    mapping = rl_endpoint_map()
    endpoints = {mapping[x] for x in labels if x in mapping}
    if endpoints:
        return {key: key in endpoints for key in S_KEYS}
    return {key: True for key in S_KEYS}


def mask_from_judge_meta(payload: dict[str, Any]) -> dict[str, bool]:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    raw = meta.get("applicability") if isinstance(meta.get("applicability"), dict) else {}
    return {key: bool(raw.get(key, True)) for key in S_KEYS}


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
        for token in _parse_rl_cell(metadata.get("task_category_rl")):
            labels.add(token)
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
