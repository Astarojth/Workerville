from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_api_file(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    src = Path(path)
    if not src.exists():
        return {}

    if src.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML API config files")
        data = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    elif src.suffix == ".txt":
        data = _load_api_modellist_txt(src)
    else:
        data = json.loads(src.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("API config file must be an object/dict")
    return {str(k): v for k, v in data.items()}


def _load_api_modellist_txt(src: Path) -> dict[str, Any]:
    raw_lines = src.read_text(encoding="utf-8").splitlines()

    provider = "openai"
    api_base = ""
    api_endpoint = "/v1/chat/completions"
    api_keys: list[str] = []
    models: list[str] = []

    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        lowered = line.lower()
        if lowered.startswith("provider:"):
            provider = line.split(":", 1)[1].strip() or provider
            continue
        if lowered.startswith("api_base:"):
            api_base = line.split(":", 1)[1].strip()
            continue
        if lowered.startswith("api_endpoint:"):
            api_endpoint = line.split(":", 1)[1].strip() or api_endpoint
            continue
        if lowered in {"api_key:", "api keys:", "model list:", "models:"}:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("sk-"):
            if line not in api_keys:
                api_keys.append(line)
            continue
        if line not in models:
            models.append(line)

    default_model = models[0] if models else ""

    payload: dict[str, Any] = {
        "provider": provider,
        "api_base": api_base,
        "api_endpoint": api_endpoint,
        "api_keys": api_keys,
        "models": models,
    }
    if default_model:
        payload["model"] = default_model
    if api_keys:
        payload["api_key"] = api_keys[0]
    return payload
