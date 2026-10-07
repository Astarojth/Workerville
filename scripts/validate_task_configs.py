#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class ValidationResult:
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_episode_config(config_path: Path) -> ValidationResult:
    res = ValidationResult()
    if not config_path.exists():
        res.errors.append(f"config file not found: {config_path}")
        return res

    try:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to validate YAML configs")
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        else:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        res.errors.append(f"failed to parse config: {e}")
        return res

    if not isinstance(raw, dict):
        res.errors.append("config root must be a mapping/object")
        return res

    required_top = ("experiment_name", "episode_id", "steps", "seed", "task", "agents", "mechanisms", "resources")
    for key in required_top:
        if key not in raw:
            res.errors.append(f"missing required field: {key}")

    task = raw.get("task")
    if isinstance(task, dict):
        if not str(task.get("task_id", "")).strip():
            res.errors.append("task.task_id is required")
        if not str(task.get("instruction", "")).strip():
            res.errors.append("task.instruction is required")
    else:
        res.errors.append("task must be an object")

    agents = raw.get("agents")
    if not isinstance(agents, list) or not agents:
        res.errors.append("agents must be a non-empty list")
    else:
        for idx, agent in enumerate(agents):
            if not isinstance(agent, dict):
                res.errors.append(f"agents[{idx}] must be an object")
                continue
            for k in ("agent_id", "kind", "role"):
                if not str(agent.get(k, "")).strip():
                    res.errors.append(f"agents[{idx}].{k} is required")

    steps = raw.get("steps")
    if not isinstance(steps, int) or steps <= 0:
        res.errors.append("steps must be a positive integer")

    seed = raw.get("seed")
    if not isinstance(seed, int):
        res.errors.append("seed must be an integer")

    mechs = raw.get("mechanisms")
    if isinstance(mechs, dict):
        for mk in ("enable_m1", "enable_m2", "enable_m3"):
            if mk not in mechs:
                res.warnings.append(f"mechanisms.{mk} is missing; default behavior may apply")
    else:
        res.errors.append("mechanisms must be an object")

    return res


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Validate one episode config")
    p.add_argument("config", help="Path to YAML/JSON config")
    args = p.parse_args()

    result = validate_episode_config(Path(args.config))
    for w in result.warnings:
        print(f"[WARN] {w}")
    for e in result.errors:
        print(f"[ERROR] {e}")
    raise SystemExit(1 if result.errors else 0)


if __name__ == "__main__":
    main()
