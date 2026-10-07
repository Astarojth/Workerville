from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from benchmark_core.runtime.api_config import APISettings


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
    token_cap = int(max_tokens or api.max_tokens or 384)
    attempts = [max(256, token_cap), max(512, token_cap * 2)]
    seen_payloads: list[dict[str, Any]] = []
    for cap in attempts:
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
                "Authorization": f"Bearer {api.api_key}",
            },
            method="POST",
        )
        body = _http_json(req, timeout_sec=int(api.timeout_sec))
        seen_payloads.append(body)
        choices = body.get("choices", []) if isinstance(body, dict) else []
        if not isinstance(choices, list) or not choices:
            continue
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "")
        text = _coerce_content_text(content)
        if text.strip():
            return text
        finish_reason = str(choices[0].get("finish_reason", "")).strip().lower() if isinstance(choices[0], dict) else ""
        if finish_reason != "length":
            break
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


def _http_json(req: Request, timeout_sec: int) -> dict[str, Any]:
    try:
        with urlopen(req, timeout=max(5, int(timeout_sec))) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as e:  # pragma: no cover
        detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
        raise RuntimeError(f"http_error {e.code}: {detail[:300]}")
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
