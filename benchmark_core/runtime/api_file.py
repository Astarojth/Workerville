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
    else:
        data = json.loads(src.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("API config file must be an object/dict")
    return {str(k): v for k, v in data.items()}
