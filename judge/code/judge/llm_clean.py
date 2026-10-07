"""Helpers for checking whether a judge endpoint file holds an LLM verdict."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ENDPOINTS = ("S1", "S2", "S3")
TRUE_LLM_PARSE = {"ok", "repaired", "agent_repaired"}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def is_silent_rule_fill(result: dict[str, Any] | None) -> bool:
    row = result if isinstance(result, dict) else {}
    source = str(row.get("source", "")).strip()
    rationale = str(row.get("rationale", "")).strip()
    if rationale.startswith("Rule fallback for"):
        return True
    if source == "rule_fallback" and "Rule fallback" in rationale:
        return True
    return False


def parse_status_of(payload: dict[str, Any]) -> str:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    parse = meta.get("parse") if isinstance(meta.get("parse"), dict) else {}
    return str(parse.get("status", "")).strip()


def is_applicable(payload: dict[str, Any]) -> bool:
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    if "applicable" in meta:
        return bool(meta.get("applicable"))
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    return str(result.get("status", "")).strip() != "not_applicable"


def is_true_llm_endpoint(payload: dict[str, Any] | None) -> bool:
    """NA or a real LLM verdict (not detector fill)."""
    if not isinstance(payload, dict) or not payload:
        return False
    if parse_status_of(payload) == "not_applicable" or not is_applicable(payload):
        return True
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    if is_silent_rule_fill(result):
        return False
    if parse_status_of(payload) not in TRUE_LLM_PARSE:
        return False
    if str(result.get("status", "")).strip() not in {"ok", "not_applicable"}:
        return False
    return result.get("verdict") in {0, 1}

