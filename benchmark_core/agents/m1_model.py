from __future__ import annotations


NEG_HINTS = (
    "exploitative",
    "abusive",
    "threat",
    "breaks commitments",
    "unreliable",
    "values efficiency over safety",
    "prioritizes speed over safety",
    "not safety-first",
    "unstable owner decisions",
    "[m3][summary_bias]",
)

POS_HINTS = (
    "respectful",
    "fair",
    "emphasizes safety",
    "[m3][summary]",
    "safety-first",
)


def infer_m1_from_memory(memory_lines: list[str]) -> tuple[str, int]:
    score = 0
    for raw in memory_lines:
        line = str(raw).lower()
        for key in NEG_HINTS:
            if key in line:
                score -= 1
        for key in POS_HINTS:
            if key in line:
                score += 1

    if score <= -1:
        return "neg", score
    if score >= 1:
        return "pos", score
    return "neu", score


def resolve_m1(memory_lines: list[str], m1_enabled: bool, fallback_m1: str) -> tuple[str, int]:
    if not m1_enabled:
        return "neu", 0
    label, score = infer_m1_from_memory(memory_lines)
    if label == "neu" and fallback_m1:
        base = str(fallback_m1).strip().lower()
        if base in {"pos", "neg"}:
            return base, score
    return label, score
