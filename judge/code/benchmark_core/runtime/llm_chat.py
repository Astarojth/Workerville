from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from benchmark_core.runtime.api_config import APISettings

_RETRYABLE_MARKERS = (
    "http_error 408",
    "http_error 409",
    "http_error 429",
    "http_error 500",
    "http_error 502",
    "http_error 503",
    "http_error 504",
    "url_error",
    "timed out",
    "timeout",
    "temporarily",
    "connection reset",
    "connection refused",
    "invalid_json_response",
)


def chat_completion(
    api: APISettings,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int | None = None,
) -> str:
    provider = str(api.provider).strip().lower()
    if provider in {"openai", "openrouter"}:
        return _chat_openai_compatible(
            api=api,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )
    if provider == "anthropic":
        return _chat_anthropic(
            api=api,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )
    raise RuntimeError(f"unsupported provider for chat_completion: {provider}")


def _chat_openai_compatible(
    api: APISettings,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int | None,
) -> str:
    endpoint = _join_url(api.api_base, api.api_endpoint)
    TOKEN_CAP = 256000
    token_cap = min(TOKEN_CAP, int(max_tokens or api.max_tokens or 384))
    attempts = [max(256, token_cap)]
    if token_cap < TOKEN_CAP:
        doubled = min(TOKEN_CAP, max(512, token_cap * 2))
        if doubled not in attempts:
            attempts.append(doubled)
    api_keys = _ordered_api_keys(api)
    if not api_keys:
        raise RuntimeError("missing_api_keys")
    for cap in attempts:
        last_error = ""
        for api_key in api_keys:
            payload: dict[str, Any] = {
                "model": api.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": float(api.temperature if api.temperature is not None else 0.0),
                "max_tokens": int(cap),
            }
            req = Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            try:
                body = _http_json_retry(req, timeout_sec=int(api.timeout_sec))
            except Exception as e:  # pragma: no cover
                last_error = str(e)
                continue
            choices = body.get("choices", []) if isinstance(body, dict) else []
            if not isinstance(choices, list) or not choices:
                last_error = "missing_choices"
                continue
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            content = message.get("content", "")
            text = _coerce_content_text(content)
            if not str(text).strip():
                text = _coerce_content_text(message.get("reasoning_content", ""))
            if text.strip():
                return text
            finish_reason = str(choices[0].get("finish_reason", "")).strip().lower() if isinstance(choices[0], dict) else ""
            if finish_reason != "length":
                last_error = f"empty_content finish_reason={finish_reason or 'unknown'}"
                break
        if last_error and cap == attempts[-1]:
            raise RuntimeError(last_error)
    raise RuntimeError(f"empty_chat_content_after_retry attempts={len(attempts)}")


def _chat_anthropic(
    api: APISettings,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int | None,
) -> str:
    endpoint = _join_url(api.api_base, api.api_endpoint or "/v1/messages")
    payload: dict[str, Any] = {
        "model": api.model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": float(api.temperature if api.temperature is not None else 0.0),
        "max_tokens": int(max_tokens or api.max_tokens or 256),
    }
    req = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api.api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    body = _http_json(req, timeout_sec=int(api.timeout_sec))
    content = body.get("content", []) if isinstance(body, dict) else []
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    return str(content)


def _is_retryable_chat_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def _http_json_retry(req: Request, timeout_sec: int, *, attempts: int = 4) -> dict[str, Any]:
    last: Exception | None = None
    total = max(1, int(attempts))
    for idx in range(total):
        try:
            return _http_json(req, timeout_sec=timeout_sec)
        except Exception as exc:  # pragma: no cover
            last = exc
            if idx + 1 >= total or not _is_retryable_chat_error(exc):
                raise
            time.sleep(min(8.0, 1.0 * (2**idx)))
    raise RuntimeError(str(last) if last else "chat_retry_failed")


def _http_json(req: Request, timeout_sec: int) -> dict[str, Any]:
    try:
        with urlopen(req, timeout=max(5, int(timeout_sec))) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as e:  # pragma: no cover
        detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
        raise RuntimeError(f"http_error {e.code}: {detail[:300]}")
    except TimeoutError as e:  # pragma: no cover
        raise RuntimeError(f"url_error: timed out: {e}")
    except URLError as e:  # pragma: no cover
        raise RuntimeError(f"url_error: {str(e)}")
    try:
        body = json.loads(raw)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"invalid_json_response: {type(e).__name__}")
    if not isinstance(body, dict):
        raise RuntimeError("response is not a json object")
    return body


def _coerce_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    return str(content)


def _join_url(base: str, endpoint: str) -> str:
    b = str(base).rstrip("/")
    e = str(endpoint).strip()
    if not e.startswith("/"):
        e = "/" + e
    return b + e


def _ordered_api_keys(api: APISettings) -> list[str]:
    out: list[str] = []
    primary = str(api.api_key).strip()
    if primary:
        out.append(primary)
    for item in api.api_keys:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out
