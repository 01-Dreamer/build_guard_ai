from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm <= 0:
        raise ValueError("face embedding has zero norm")
    return vector / norm


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as tmp:
        json.dump(payload, tmp, ensure_ascii=False)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


@dataclass(frozen=True)
class FaceRecord:
    person_id: str
    name: str
    embedding: np.ndarray
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class FaceMatch:
    person_id: str
    name: str
    score: float


class FaissFaceStore:
    """Persistent one-vector-per-person FAISS store."""

    def __init__(self, data_dir: Path, dimension: int | None = None) -> None:
        self.data_dir = data_dir
        self.index_path = data_dir / "faces.faiss"
        self.meta_path = data_dir / "faces_meta.json"
        self.dimension = dimension
        self.records: dict[str, FaceRecord] = {}
        self.ids: list[str] = []
        self.index: faiss.IndexFlatIP | None = None
        self._lock = threading.RLock()
        self.load()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.records)

    def load(self) -> None:
        with self._lock:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.records = {}
            self.ids = []

            if self.meta_path.exists():
                with self.meta_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                self.dimension = payload.get("dimension") or self.dimension
                self.ids = list(payload.get("ids", []))
                for person_id, item in payload.get("records", {}).items():
                    self.records[person_id] = FaceRecord(
                        person_id=person_id,
                        name=item.get("name") or person_id,
                        embedding=_normalize(np.asarray(item["embedding"], dtype=np.float32)),
                        created_at=item.get("created_at") or _utc_now(),
                        updated_at=item.get("updated_at") or _utc_now(),
                    )

            if self.dimension is not None and self.index_path.exists():
                index = faiss.read_index(str(self.index_path))
                if index.d == self.dimension and index.ntotal == len(self.ids):
                    self.index = index
                    return

            self._rebuild_index()

    def upsert(self, person_id: str, name: str | None, embedding: np.ndarray) -> bool:
        person_id = person_id.strip()
        if not person_id:
            raise ValueError("id cannot be empty")

        vector = _normalize(embedding)
        with self._lock:
            if self.dimension is None:
                self.dimension = int(vector.shape[0])
            if vector.shape[0] != self.dimension:
                raise ValueError(
                    f"embedding dimension mismatch: expected {self.dimension}, got {vector.shape[0]}"
                )

            now = _utc_now()
            replaced = person_id in self.records
            created_at = self.records[person_id].created_at if replaced else now
            self.records[person_id] = FaceRecord(
                person_id=person_id,
                name=(name or person_id).strip() or person_id,
                embedding=vector,
                created_at=created_at,
                updated_at=now,
            )
            self._rebuild_index()
            self.save()
            return replaced

    def delete(self, person_id: str) -> bool:
        person_id = person_id.strip()
        with self._lock:
            if person_id not in self.records:
                return False
            del self.records[person_id]
            self._rebuild_index()
            self.save()
            return True

    def search(self, embedding: np.ndarray, threshold: float, k: int = 1) -> FaceMatch | None:
        vector = _normalize(embedding).reshape(1, -1)
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return None
            if vector.shape[1] != self.index.d:
                raise ValueError(
                    f"embedding dimension mismatch: expected {self.index.d}, got {vector.shape[1]}"
                )

            scores, indices = self.index.search(vector.astype(np.float32), k)
            best_index = int(indices[0][0])
            best_score = float(scores[0][0])
            if best_index < 0 or best_score < threshold:
                return None

            person_id = self.ids[best_index]
            record = self.records[person_id]
            return FaceMatch(person_id=record.person_id, name=record.name, score=best_score)

    def save(self) -> None:
        with self._lock:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "dimension": self.dimension,
                "ids": self.ids,
                "records": {
                    person_id: {
                        "name": record.name,
                        "embedding": record.embedding.astype(float).tolist(),
                        "created_at": record.created_at,
                        "updated_at": record.updated_at,
                    }
                    for person_id, record in self.records.items()
                },
            }

            if self.index is not None:
                tmp_index = self.index_path.with_suffix(".faiss.tmp")
                faiss.write_index(self.index, str(tmp_index))
                os.replace(tmp_index, self.index_path)
            _atomic_write_json(self.meta_path, payload)

    def _rebuild_index(self) -> None:
        if self.dimension is None:
            self.index = None
            self.ids = []
            return

        self.ids = sorted(self.records)
        index = faiss.IndexFlatIP(int(self.dimension))
        if self.ids:
            vectors = np.vstack([self.records[person_id].embedding for person_id in self.ids])
            index.add(vectors.astype(np.float32))
        self.index = index
