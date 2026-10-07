from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_minimal_closure import _apply_shard, _retry_wait_seconds
from agent_os.openclaw_adapter import OpenClawAdapter


class ShardTests(unittest.TestCase):
    def test_round_robin_split_has_no_overlap(self):
        items = [Path(f"C{i}__L3.yaml") for i in range(16)]
        shards = [_apply_shard(items, i, 8) for i in range(8)]
        flat = [p for shard in shards for p in shard]
        self.assertEqual(len(flat), 16)
        self.assertEqual(len(set(flat)), 16)
        self.assertEqual(shards[0], [items[0], items[8]])
        self.assertEqual(shards[7], [items[7], items[15]])

    def test_single_shard_keeps_order(self):
        items = [Path("a"), Path("b")]
        self.assertEqual(_apply_shard(items, 0, 1), items)

    def test_retry_wait_honors_retry_after(self):
        self.assertEqual(_retry_wait_seconds("rate limit exceeded; retry after 35s"), 37.0)
        self.assertEqual(_retry_wait_seconds("rate limit exceeded for config.patch"), 40.0)
        self.assertEqual(_retry_wait_seconds("unrelated boom"), 2.0)

    def test_adapter_retry_wait_parses_gateway_message(self):
        self.assertEqual(
            OpenClawAdapter._retry_wait_seconds(
                "openclaw gateway method=config.patch failed: rate limit exceeded for config.patch; retry after 35s"
            ),
            36.0,
        )


if __name__ == "__main__":
    unittest.main()
