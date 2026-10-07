from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np


_MODEL_CACHE: dict[tuple[str, int, str], dict[str, Any]] = {}
_ENCODE_CACHE: dict[str, np.ndarray] = {}


def _disk_cache_dir() -> Path:
    env = str(os.environ.get("WORKERVILLE_EMBED_CACHE_DIR", "") or "").strip()
    if env:
        return Path(env)
    return Path.cwd() / "results" / "embed_cache"


def _local_files_only() -> bool:
    return str(os.environ.get("WORKERVILLE_EMBED_LOCAL_FILES_ONLY", "") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class EmbeddingRetriever:
    def __init__(self, *, model_name: str, matryoshka_dim: int = 512):
        self.model_name = str(model_name).strip()
        if not self.model_name:
            raise ValueError("embedding model must be supplied by the user")
        self.dim = max(64, int(matryoshka_dim or 512))
        self._backend = "transformers"
        self._model = None
        self._tokenizer = None
        self._torch = None
        cache_key = (self.model_name, self.dim, "cpu")
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            self._backend = str(cached["backend"])
            self._model = cached["model"]
            self._tokenizer = cached.get("tokenizer")
            self._torch = cached["torch"]
            return
        try:
            import torch  # type: ignore

            self._torch = torch
            threads = max(1, int(os.environ.get("WORKERVILLE_EMBED_THREADS", "4")))
            torch.set_num_threads(threads)
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                prefer = str(os.environ.get("WORKERVILLE_EMBED_BACKEND", "transformers")).strip().lower()
                # Disk cache keys are `{backend}:{model}:{dim}:{text}`.
                if prefer in {"sentence_transformers", "st", "sbert"}:
                    self._backend = "sentence_transformers"
                    self._model = SentenceTransformer(
                        self.model_name,
                        trust_remote_code=True,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )
                else:
                    raise RuntimeError("prefer transformers backend for embed-cache reuse")
            except Exception:
                from transformers import AutoModel, AutoTokenizer  # type: ignore

                self._backend = "transformers"
                local_only = _local_files_only()
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    local_files_only=local_only,
                )
                self._model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    local_files_only=local_only,
                )
                self._model.eval()
            _MODEL_CACHE[cache_key] = {
                "backend": self._backend,
                "model": self._model,
                "tokenizer": self._tokenizer,
                "torch": self._torch,
            }
        except Exception as e:
            raise RuntimeError(
                "EmbeddingRetriever requires sentence-transformers or transformers+torch "
                "in the active environment. Deterministic hash fallback is disabled."
            ) from e

    @property
    def backend(self) -> str:
        return self._backend

    def retrieve(
        self,
        *,
        query_text: str,
        entries: list[dict[str, Any]],
        top_k: int,
        exclude_episode_id: str = "",
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        current_episode_id = str(exclude_episode_id).strip()
        query_vec = self._encode_text(query_text)
        scored: list[tuple[float, dict[str, Any]]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_episode_id = str(entry.get("episode_id", "")).strip()
            if current_episode_id and entry_episode_id == current_episode_id:
                continue
            text = str(entry.get("embedding_text", "") or entry.get("text", "")).strip()
            if not text:
                continue
            vec = self._encode_text(text)
            score = self._cosine(query_vec, vec)
            enriched = dict(entry)
            enriched["score"] = round(score, 6)
            scored.append((score, enriched))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    def _encode_text(self, text: str) -> np.ndarray:
        key = f"{self._backend}:{self.model_name}:{self.dim}:{text}"
        cached = _ENCODE_CACHE.get(key)
        if cached is not None:
            return cached
        disk_vec = _load_disk_vector(key)
        if disk_vec is not None:
            _ENCODE_CACHE[key] = disk_vec
            return disk_vec
        if self._backend == "sentence_transformers":
            vec = self._encode_sentence_transformers(text)
        else:
            vec = self._encode_transformers(text)
        _ENCODE_CACHE[key] = vec
        _save_disk_vector(key, vec)
        return vec

    def _encode_sentence_transformers(self, text: str) -> np.ndarray:
        torch = self._torch
        assert torch is not None
        with torch.no_grad():
            tensor = self._model.encode([text], convert_to_tensor=True, show_progress_bar=False)  # type: ignore[union-attr]
            if tensor.ndim == 2:
                tensor = tensor[0]
            if self.dim < int(tensor.shape[0]):
                tensor = tensor[: self.dim]
            tensor = torch.nn.functional.normalize(tensor, p=2, dim=0)
            return tensor.detach().cpu().numpy().astype(np.float32)

    def _encode_transformers(self, text: str) -> np.ndarray:
        torch = self._torch
        assert torch is not None
        assert self._tokenizer is not None
        encoded = self._tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = self._model(**encoded)
            token_embeddings = output[0]
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            tensor = pooled[0]
            if self.dim < int(tensor.shape[0]):
                tensor = tensor[: self.dim]
            tensor = torch.nn.functional.normalize(tensor, p=2, dim=0)
            return tensor.detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def _cosine(left: np.ndarray, right: np.ndarray) -> float:
        if left.shape != right.shape:
            return 0.0
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom <= 0.0:
            return 0.0
        value = float(np.dot(left, right) / denom)
        return max(-1.0, min(1.0, value))


def _disk_path(key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return _disk_cache_dir() / f"{digest}.npy"


def _load_disk_vector(key: str) -> np.ndarray | None:
    path = _disk_path(key)
    if not path.is_file():
        return None
    try:
        vec = np.load(path)
    except Exception:
        return None
    if not isinstance(vec, np.ndarray):
        return None
    return vec.astype(np.float32)


def _save_disk_vector(key: str, vec: np.ndarray) -> None:
    _disk_cache_dir().mkdir(parents=True, exist_ok=True)
    path = _disk_path(key)
    tmp = path.with_suffix(".tmp.npy")
    try:
        np.save(tmp, vec.astype(np.float32))
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
