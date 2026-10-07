from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class APISettings:
    provider: str = "openai"
    model: str = ""
    api_base: str = ""
    api_endpoint: str = "/chat/completions"
    api_key: str = ""
    api_keys: tuple[str, ...] = field(default_factory=tuple)
    timeout_sec: int = 60
    temperature: float | None = None
    max_tokens: int | None = None

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.provider not in {"openai", "openrouter", "anthropic"}:
            issues.append("unsupported provider; use openai/openrouter/anthropic")
        if self.provider in {"openai", "anthropic", "openrouter"} and not (self.api_key or self.api_keys):
            issues.append(f"missing API key for provider={self.provider}")
        if not self.model:
            issues.append("missing model")
        if not self.api_base:
            issues.append("missing api_base")
        if not self.api_endpoint.startswith("/"):
            issues.append("api_endpoint must start with '/'")
        if self.timeout_sec <= 0:
            issues.append("timeout_sec must be > 0")
        if self.temperature is not None and not (0.0 <= self.temperature <= 2.0):
            issues.append("temperature should be in [0,2] when provided")
        if self.max_tokens is not None and self.max_tokens <= 0:
            issues.append("max_tokens must be > 0 when provided")
        return issues

    def public_dict(self) -> dict:
        payload = asdict(self)
        payload["api_key"] = "***" if self.api_key else ""
        payload["api_keys"] = ["***" for _ in self.api_keys]
        return payload


def load_api_settings(config_llm: dict | None = None) -> APISettings:
    config_llm = config_llm or {}

    provider = str(config_llm.get("provider") or os.getenv("WORKERVILLE_API_PROVIDER", "openai")).lower()
    model = str(config_llm.get("model") or os.getenv("WORKERVILLE_MODEL", ""))
    api_base = str(config_llm.get("api_base") or os.getenv("WORKERVILLE_API_BASE", ""))
    api_endpoint = str(config_llm.get("api_endpoint") or os.getenv("WORKERVILLE_API_ENDPOINT", "/chat/completions"))

    key_from_config = str(config_llm.get("api_key", ""))
    key_from_env = (
        os.getenv("WORKERVILLE_API_KEY", "")
        or os.getenv(_provider_key_env_name(provider), "")
    )
    parsed_api_keys = _parse_api_keys(config_llm.get("api_keys"))
    api_key = key_from_config or key_from_env or (parsed_api_keys[0] if parsed_api_keys else "")
    api_keys = _merge_api_keys(api_key, parsed_api_keys)

    timeout_sec = int(_pick(config_llm, "timeout_sec", os.getenv("WORKERVILLE_TIMEOUT_SEC"), "60"))
    # Force deterministic sampling policy for all runs.
    # User-level config/env temperature is intentionally ignored.
    temperature = 0.0
    max_tokens = _parse_optional_int(_pick(config_llm, "max_tokens", os.getenv("WORKERVILLE_MAX_TOKENS"), None))

    return APISettings(
        provider=provider,
        model=model,
        api_base=api_base,
        api_endpoint=api_endpoint,
        api_key=api_key,
        api_keys=api_keys,
        timeout_sec=timeout_sec,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _provider_key_env_name(provider: str) -> str:
    if provider == "anthropic":
        return "ANTHROPIC_API_KEY"
    if provider == "openrouter":
        return "OPENROUTER_API_KEY"
    return "OPENAI_API_KEY"


def _pick(config_llm: dict, key: str, env_value: str | None, default: str | None) -> str | None:
    if key in config_llm:
        value = config_llm[key]
        if value is None:
            return None
        return str(value)
    if env_value is not None:
        return env_value
    return default


def _parse_optional_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return float(raw)


def _parse_optional_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return int(raw)


def _parse_api_keys(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        return (text,) if text else ()
    if not isinstance(raw, (list, tuple)):
        return ()
    out: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _merge_api_keys(primary: str, others: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    text = str(primary).strip()
    if text:
        out.append(text)
    for item in others:
        candidate = str(item).strip()
        if candidate and candidate not in out:
            out.append(candidate)
    return tuple(out)
