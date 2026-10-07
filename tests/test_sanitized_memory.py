from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JUDGE_CODE = ROOT / "judge" / "code"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(JUDGE_CODE) not in sys.path:
    sys.path.append(str(JUDGE_CODE))

from judge.temp_memory import load_packaged_review_memory_bundle


class SanitizedMemoryTests(unittest.TestCase):
    def test_memory_loads_with_expected_counts(self):
        bundle = load_packaged_review_memory_bundle(
            project_root=ROOT / "judge",
            memory_root=str(ROOT / "judge" / "memory_bank"),
        )
        counts = {
            endpoint: len(bundle["banks"][endpoint]["entries"])
            for endpoint in ("S1", "S2", "S3")
        }
        self.assertEqual(counts, {"S1": 150, "S2": 148, "S3": 451})

    def test_loaded_entries_do_not_publish_source_models_or_paths(self):
        bundle = load_packaged_review_memory_bundle(
            project_root=ROOT / "judge",
            memory_root=str(ROOT / "judge" / "memory_bank"),
        )
        for endpoint in ("S1", "S2", "S3"):
            for entry in bundle["banks"][endpoint]["entries"]:
                self.assertEqual(entry["model_source"], "")
                self.assertTrue(entry["task_id"].startswith("anonymous_task_"))
                text = " ".join(
                    str(entry.get(key, ""))
                    for key in ("analysis", "trajectory", "embedding_text")
                )
                self.assertNotIn("/" + "Users/", text)
                self.assertNotIn("/" + "home/", text)
                self.assertNotIn("http://", text)
                self.assertNotIn("https://", text)
                self.assertNotIn("Source seed:", text)


if __name__ == "__main__":
    unittest.main()
