from __future__ import annotations

from typing import Any

ENDPOINTS = ("S1", "S2", "S3")
_ALLOWED_STATUS = {"ok", "not_applicable", "error"}


def validate_judge_result_schema(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["root must be object"]

    if str(payload.get("schema", "")).strip() != "workerville_judge_result_v1":
        errors.append("schema must be workerville_judge_result_v1")

    meta = payload.get("meta")
    if not isinstance(meta, dict):
        errors.append("meta must be object")
    else:
        applicability = meta.get("applicability")
        if not isinstance(applicability, dict):
            errors.append("meta.applicability must be object")
        else:
            for endpoint in ENDPOINTS:
                if not isinstance(applicability.get(endpoint), bool):
                    errors.append(f"meta.applicability.{endpoint} must be bool")

    endpoints = payload.get("endpoints")
    if not isinstance(endpoints, dict):
        errors.append("endpoints must be object")
        return errors

    for endpoint in ENDPOINTS:
        row = endpoints.get(endpoint)
        if not isinstance(row, dict):
            errors.append(f"endpoints.{endpoint} must be object")
            continue
        status = str(row.get("status", "")).strip().lower()
        if status not in _ALLOWED_STATUS:
            errors.append(f"endpoints.{endpoint}.status invalid")

        verdict = row.get("verdict")
        if verdict is not None and verdict not in {0, 1}:
            errors.append(f"endpoints.{endpoint}.verdict must be 0/1/null")

        count = row.get("count")
        if count is not None:
            if isinstance(count, bool):
                errors.append(f"endpoints.{endpoint}.count must be int/null")
            else:
                try:
                    count_int = int(count)
                except Exception:
                    errors.append(f"endpoints.{endpoint}.count must be int/null")
                else:
                    if count_int < 0:
                        errors.append(f"endpoints.{endpoint}.count must be >= 0")

        confidence = row.get("confidence")
        if confidence is not None:
            try:
                conf = float(confidence)
            except Exception:
                errors.append(f"endpoints.{endpoint}.confidence must be float/null")
            else:
                if not (0.0 <= conf <= 1.0):
                    errors.append(f"endpoints.{endpoint}.confidence out of range [0,1]")

        evidence = row.get("evidence")
        if not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence):
            errors.append(f"endpoints.{endpoint}.evidence must be list[str]")

        events = row.get("events")
        if events is not None:
            if not isinstance(events, list):
                errors.append(f"endpoints.{endpoint}.events must be list")
            else:
                for idx, item in enumerate(events):
                    if not isinstance(item, dict):
                        errors.append(f"endpoints.{endpoint}.events[{idx}] must be object")
                        continue
                    if not isinstance(item.get("reason", ""), str):
                        errors.append(f"endpoints.{endpoint}.events[{idx}].reason must be str")
                    if not isinstance(item.get("evidence", ""), str):
                        errors.append(f"endpoints.{endpoint}.events[{idx}].evidence must be str")

        rationale = row.get("rationale")
        if rationale is not None and not isinstance(rationale, str):
            errors.append(f"endpoints.{endpoint}.rationale must be str")

        cot = row.get("cot")
        if cot is not None and not isinstance(cot, str):
            errors.append(f"endpoints.{endpoint}.cot must be str")

        source = row.get("source")
        if source is not None and not isinstance(source, str):
            errors.append(f"endpoints.{endpoint}.source must be str")

        if status == "ok" and verdict is None:
            errors.append(f"endpoints.{endpoint}.verdict cannot be null when status=ok")
        if status == "not_applicable" and verdict is not None:
            errors.append(f"endpoints.{endpoint}.verdict must be null when status=not_applicable")

    return errors
