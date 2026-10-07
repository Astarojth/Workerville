from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | None, override: bool = False) -> bool:
    if not path:
        return False
    src = Path(path)
    if not src.exists():
        return False

    for raw in src.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = _clean_value(val.strip())
        if not key:
            continue
        if key in os.environ and not override:
            continue
        os.environ[key] = val
    return True


def _clean_value(v: str) -> str:
    if len(v) >= 2 and ((v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'"))):
        return v[1:-1]
    return v
