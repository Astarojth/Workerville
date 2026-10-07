from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
JUDGE_CODE = ROOT / "judge" / "code"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(JUDGE_CODE) not in sys.path:
    sys.path.append(str(JUDGE_CODE))

from judge.embedding_retrieval import EmbeddingRetriever


class JudgeRetrievalTests(unittest.TestCase):
    def test_current_episode_is_excluded_before_top_k_selection(self):
        retriever = object.__new__(EmbeddingRetriever)
        vectors = {
            "query": np.array([1.0, 0.0], dtype=np.float32),
            "self": np.array([1.0, 0.0], dtype=np.float32),
            "other-a": np.array([0.9, 0.1], dtype=np.float32),
            "other-b": np.array([0.8, 0.2], dtype=np.float32),
        }
        retriever._encode_text = vectors.__getitem__

        hits = retriever.retrieve(
            query_text="query",
            entries=[
                {"id": "self", "episode_id": "episode-current", "embedding_text": "self"},
                {"id": "a", "episode_id": "episode-a", "embedding_text": "other-a"},
                {"id": "b", "episode_id": "episode-b", "embedding_text": "other-b"},
            ],
            top_k=2,
            exclude_episode_id="episode-current",
        )

        self.assertEqual([entry["id"] for entry in hits], ["a", "b"])

    def test_empty_episode_id_does_not_exclude_entries(self):
        retriever = object.__new__(EmbeddingRetriever)
        retriever._encode_text = lambda _: np.array([1.0, 0.0], dtype=np.float32)

        hits = retriever.retrieve(
            query_text="query",
            entries=[{"id": "kept", "episode_id": "episode-a", "embedding_text": "entry"}],
            top_k=1,
        )

        self.assertEqual([entry["id"] for entry in hits], ["kept"])


if __name__ == "__main__":
    unittest.main()
