from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .types import AgentSpec, EpisodeConfig, TaskSpec


try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_config(path: str, seed_override: int | None = None) -> EpisodeConfig:
    raw = _read_struct(path)
    task = TaskSpec(**raw["task"])
    agents = [AgentSpec(**item) for item in raw["agents"]]
    seed = seed_override if seed_override is not None else int(raw.get("seed", 42))
    return EpisodeConfig(
        experiment_name=raw["experiment_name"],
        episode_id=raw["episode_id"],
        steps=int(raw["steps"]),
        seed=seed,
        task=task,
        agents=agents,
        mechanisms=raw.get("mechanisms", {}),
        resources=raw.get("resources", {}),
        runtime=raw.get("runtime", {}),
        llm=raw.get("llm", {}),
        initial_messages=raw.get("initial_messages", []),
        timeline_events=raw.get("timeline_events", []),
    )


def dump_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def dump_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def hash_payload(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def hash_memory(lines: list[str]) -> str:
    blob = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _read_struct(path: str) -> dict[str, Any]:
    src = Path(path)
    if src.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML config files")
        with open(src, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(src, "r", encoding="utf-8") as f:
        return json.load(f)
