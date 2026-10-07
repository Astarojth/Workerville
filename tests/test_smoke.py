from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_os import OpenClawAdapter, OpenClawNativeAgent, OpenClawSummarizerAgent
from benchmark_core.orchestrator import EpisodeRunner
from benchmark_core.shared.io import load_config
from scripts.run_experiment import _normalize_openclaw_runtime
from scripts.validate_task_configs import validate_episode_config


class SmokeTests(unittest.TestCase):
    @staticmethod
    def _example_config() -> Path:
        return ROOT / "configs" / "tasks" / "agdojo2_banking_user_task_15_25_v1" / "C0__L3.yaml"

    def test_template_config_validation_and_load(self):
        cfg_path = self._example_config()
        res = validate_episode_config(cfg_path)
        self.assertEqual(res.errors, [])
        cfg = load_config(str(cfg_path))
        self.assertEqual(cfg.task.task_id, "agdojo2_banking_user_task_15_25_v1")
        self.assertEqual(len(cfg.agents), 1)
        self.assertGreater(cfg.steps, 0)

    def test_core_imports(self):
        self.assertTrue(callable(EpisodeRunner))
        self.assertTrue(callable(OpenClawAdapter))
        self.assertTrue(callable(OpenClawNativeAgent))
        self.assertTrue(callable(OpenClawSummarizerAgent))

    def test_openclaw_identity_path_prefers_api_config(self):
        cfg = load_config(str(self._example_config()), seed_override=42)
        api_cfg = {
            "openclaw_gateway_base_url": "ws://127.0.0.1:28789",
            "openclaw_gateway_api_key": "dummy",
            "openclaw_gateway_protocol": "ws",
            "openclaw_gateway_workspace_root": "/tmp/openclaw/ws",
            "openclaw_gateway_device_identity_path": "/tmp/openclaw/device_identity/paired.json",
        }
        _normalize_openclaw_runtime(
            cfg,
            ROOT / "runs" / "template_episode" / "test_scope",
            api_file_cfg=api_cfg,
            run_scope="template_episode|seed=42|run=test_scope",
        )
        openclaw_cfg = cfg.runtime.get("openclaw", {})
        self.assertEqual(
            str(openclaw_cfg.get("gateway_device_identity_path", "")),
            "/tmp/openclaw/device_identity/paired.json",
        )


if __name__ == "__main__":
    unittest.main()
