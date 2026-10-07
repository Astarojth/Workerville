#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
URL_RE = re.compile(r"\b(?:https?|wss?)://\S+", re.IGNORECASE)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SECRET_RE = re.compile(r"\b(?:sk|priv|token|key|pwd)[-_][A-Za-z0-9_-]{6,}\b", re.IGNORECASE)
LONG_ID_RE = re.compile(r"\b(?=[A-Z0-9-]{12,}\b)(?=[A-Z0-9-]*\d)[A-Z0-9-]+\b", re.IGNORECASE)
LONG_NUMBER_RE = re.compile(r"\b\d{5,}\b")
ABS_PATH_RE = re.compile(r"(?<!\w)/(?:Users|home|workspace)/[^\s,'\"]+")
ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+(?:[A-Z][A-Za-z0-9.-]*\s+){0,4}"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b",
    re.IGNORECASE,
)
TAGGED_ID_RE = re.compile(r"\b[A-Z]{2,8}-\d{3,}\b")
SOURCE_CASE_RE = re.compile(r"\b(?:user_task|injection_task|abstract)[_-]?\d+\b", re.IGNORECASE)
CONFIG_RE = re.compile(r"\bConfig=C\d+\b|\bLevel=L\d+\b", re.IGNORECASE)
ROLE_REPLACEMENTS = {
    "worker_main": "[WORKER]",
    "owner_1": "[OWNER]",
    "peer_neutral": "[PEER]",
    "partner_ops": "[PARTNER]",
    "external_proxy_counsel": "[EXTERNAL_REVIEWER]",
    "spoofed_owner_thread": "[UNVERIFIED_OWNER]",
    "forum_signal_bot": "[FORUM]",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a de-identified Judge memory bank.")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    return parser.parse_args()


def redact_text(value: Any, replacements: dict[str, str]) -> str:
    text = str(value or "")
    for old in sorted((key for key in replacements if key), key=len, reverse=True):
        text = text.replace(old, replacements[old])
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = URL_RE.sub("[URL]", text)
    text = IP_RE.sub("[IP]", text)
    text = SECRET_RE.sub("[SECRET]", text)
    text = ABS_PATH_RE.sub("[PATH]", text)
    text = ADDRESS_RE.sub("[ADDRESS]", text)
    text = TAGGED_ID_RE.sub("[IDENTIFIER]", text)
    text = SOURCE_CASE_RE.sub("[SOURCE_CASE]", text)
    text = CONFIG_RE.sub("[CONFIG]", text)
    text = LONG_ID_RE.sub("[IDENTIFIER]", text)
    text = LONG_NUMBER_RE.sub("[NUMBER]", text)
    for old, new in ROLE_REPLACEMENTS.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text


def clean_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        return redact_text(value, replacements)
    if isinstance(value, list):
        return [clean_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: clean_strings(item, replacements) for key, item in value.items()}
    return value


def sanitize_trajectory(rows: Any, replacements: dict[str, str]) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return clean
    for row in rows:
        if not isinstance(row, dict):
            continue
        clean.append(
            {
                "step_id": row.get("step_id"),
                "agent_id": "agent",
                "role": redact_text(row.get("role", ""), replacements),
                "action": redact_text(row.get("action", ""), replacements),
                "tool_calls": clean_strings(row.get("tool_calls", []), replacements),
                "messages": clean_strings(row.get("messages", []), replacements),
            }
        )
    return clean


def sanitize_issues(rows: Any, replacements: dict[str, str]) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return clean
    for row in rows:
        if not isinstance(row, dict):
            continue
        clean.append(
            {
                "step_id": row.get("step_id"),
                "action": redact_text(row.get("action", ""), replacements),
                "issue_types": clean_strings(row.get("issue_types", []), replacements),
            }
        )
    return clean


def sanitize_row(row: dict[str, Any], *, endpoint: str, index: int) -> dict[str, Any]:
    candidate = row.get("candidate", {}) if isinstance(row.get("candidate"), dict) else {}
    payload = row.get("payload", {}) if isinstance(row.get("payload"), dict) else {}
    payload_candidate = payload.get("candidate", {}) if isinstance(payload.get("candidate"), dict) else {}
    reviews = row.get("reviews", {}) if isinstance(row.get("reviews"), dict) else {}
    chair = reviews.get("chair", {}) if isinstance(reviews.get("chair"), dict) else {}
    chair_result = chair.get("result", {}) if isinstance(chair.get("result"), dict) else {}

    example_id = f"{endpoint.lower()}_example_{index:04d}"
    task_id = f"anonymous_task_{index:04d}"
    replacements: dict[str, str] = {}
    for source in (candidate, payload_candidate):
        for key in ("id", "task_id", "episode_id", "experiment_name"):
            raw = str(source.get(key, "")).strip()
            if raw:
                replacements[raw] = example_id if key in {"id", "episode_id"} else task_id
        model = str(source.get("source_model", "")).strip()
        if model:
            replacements[model] = "[MODEL]"
    for review in reviews.values():
        if isinstance(review, dict):
            model = str(review.get("model", "")).strip()
            if model:
                replacements[model] = "[MODEL]"
            relabel = review.get("relabel", {})
            if isinstance(relabel, dict):
                model = str(relabel.get("reviewer_model", "")).strip()
                if model:
                    replacements[model] = "[MODEL]"

    clean_candidate = {
        "id": example_id,
        "endpoint": endpoint,
        "task_id": task_id,
        "task_family": "anonymous",
        "instruction": redact_text(
            candidate.get("instruction", payload_candidate.get("instruction", "")), replacements
        ),
        "rule_count": int(candidate.get("rule_count", payload.get("rule_count", 0)) or 0),
        "text": "",
    }
    minimal_result = {
        key: clean_strings(chair_result[key], replacements)
        for key in (
            "final_label",
            "rule_label",
            "rule_is_reliable",
            "agreement_with_rule",
            "confidence",
            "reason",
            "key_evidence",
            "notes",
        )
        if key in chair_result
    }
    return {
        "candidate": clean_candidate,
        "payload": {
            "candidate": dict(clean_candidate),
            "mechanical_label": payload.get("mechanical_label"),
            "rule_count": int(payload.get("rule_count", clean_candidate["rule_count"]) or 0),
            "metric_summary": {
                "endpoint_count": (payload.get("metric_summary", {}) or {}).get("endpoint_count", 0)
                if isinstance(payload.get("metric_summary", {}), dict)
                else 0
            },
            "trajectory_excerpt": sanitize_trajectory(payload.get("trajectory_excerpt", []), replacements),
            "issue_excerpt": sanitize_issues(payload.get("issue_excerpt", []), replacements),
        },
        "reviews": {"chair": {"result": minimal_result}},
    }


def main() -> None:
    args = parse_args()
    for endpoint_dir in sorted(path for path in args.source.iterdir() if path.is_dir()):
        review_dir = endpoint_dir / "reviews"
        if not review_dir.is_dir():
            continue
        output_dir = args.destination / endpoint_dir.name / "reviews"
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, path in enumerate(sorted(review_dir.glob("*.json")), start=1):
            row = json.loads(path.read_text(encoding="utf-8"))
            clean = sanitize_row(row, endpoint=endpoint_dir.name, index=index)
            target = output_dir / f"{index:04d}.json"
            target.write_text(json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
