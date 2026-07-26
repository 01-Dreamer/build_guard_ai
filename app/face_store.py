"""FAISS 人脸特征向量存储。

本模块基于 FAISS 实现了"一人一向量"的持久化人脸特征存储，
支持注册（upsert）、删除（delete）、搜索（search）和重建索引。

存储结构:
    - faces.faiss     : FAISS 内积索引文件（IndexFlatIP）
    - faces_meta.json : 人员元数据（ID、姓名、特征向量、时间戳）

线程安全：所有读写操作均使用可重入锁保护。
"""

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
    """获取当前 UTC 时间 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _normalize(vector: np.ndarray) -> np.ndarray:
    """将向量归一化为单位向量（L2 范数=1），用于内积搜索。"""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vector)
    if norm <= 0:
        raise ValueError("人脸特征向量为零向量，无法归一化")
    return vector / norm


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """原子写入 JSON 文件：先写临时文件，再 rename 覆盖，避免写入过程中文件损坏。"""
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
    """注册人员的人脸记录（不可变数据类）。

    属性:
        person_id:  人员唯一ID
        name:       人员姓名
        embedding:  归一化后的 512 维特征向量
        created_at: 首次注册时间
        updated_at: 最近更新时间
    """
    person_id: str
    name: str
    embedding: np.ndarray
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class FaceMatch:
    """人脸搜索结果（不可变数据类）。

    属性:
        person_id: 匹配到的人员ID
        name:      人员姓名
        score:     余弦相似度分数
    """
    person_id: str
    name: str
    score: float


class FaissFaceStore:
    """基于 FAISS IndexFlatIP 的持久化人脸特征存储。

    每个人员只保留一个特征向量（最近一次注册的人脸）。
    使用内积索引（IndexFlatIP），归一化向量下的内积等价于余弦相似度。

    用法:
        store = FaissFaceStore(Path("data/faces"))
        store.upsert("1", "张三", embedding)  # 注册或更新
        match = store.search(query_embedding, threshold=0.55)  # 搜索
    """

    def __init__(self, data_dir: Path, dimension: int | None = None) -> None:
        """初始化人脸存储。

        参数:
            data_dir:  数据持久化目录
            dimension: 特征向量维度，首次使用时从数据中推断
        """
        self.data_dir = data_dir
        self.index_path = data_dir / "faces.faiss"     # FAISS 索引文件
        self.meta_path = data_dir / "faces_meta.json"  # 元数据 JSON
        self.dimension = dimension
        self.records: dict[str, FaceRecord] = {}        # person_id → FaceRecord
        self.ids: list[str] = []                        # 有序的人员 ID 列表（与 FAISS 索引行对应）
        self.index: faiss.IndexFlatIP | None = None     # FAISS 内积索引
        self._lock = threading.RLock()                  # 可重入锁，保证线程安全
        self.load()

    @property
    def count(self) -> int:
        """已注册人数。"""
        with self._lock:
            return len(self.records)

    def load(self) -> None:
        """从磁盘加载 FAISS 索引和元数据，如果数据损坏则重建索引。"""
        with self._lock:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.records = {}
            self.ids = []

            # 加载元数据 JSON
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

            # 加载 FAISS 索引（维度匹配且条目数一致才认为有效）
            if self.dimension is not None and self.index_path.exists():
                index = faiss.read_index(str(self.index_path))
                if index.d == self.dimension and index.ntotal == len(self.ids):
                    self.index = index
                    return

            # 数据不完整，重建索引
            self._rebuild_index()

    def upsert(self, person_id: str, name: str | None, embedding: np.ndarray) -> bool:
        """注册或更新人脸特征。

        参数:
            person_id: 人员唯一ID
            name:      人员姓名，为 None 时使用 person_id
            embedding: 原始特征向量（会自动归一化）

        返回:
            True 表示更新已有记录，False 表示新增
        """
        person_id = person_id.strip()
        if not person_id:
            raise ValueError("人员 ID 不能为空")

        vector = _normalize(embedding)
        with self._lock:
            # 首次注册时推断向量维度
            if self.dimension is None:
                self.dimension = int(vector.shape[0])
            if vector.shape[0] != self.dimension:
                raise ValueError(
                    f"特征向量维度不匹配: 期望 {self.dimension}，实际 {vector.shape[0]}"
                )

            now = _utc_now()
            replaced = person_id in self.records
            # 更新时保留原始创建时间
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
        """删除指定人员的人脸记录。

        返回:
            True 表示删除成功，False 表示记录不存在
        """
        person_id = person_id.strip()
        with self._lock:
            if person_id not in self.records:
                return False
            del self.records[person_id]
            self._rebuild_index()
            self.save()
            return True

    def search(self, embedding: np.ndarray, threshold: float, k: int = 1) -> FaceMatch | None:
        """搜索最匹配的人脸。

        参数:
            embedding: 查询人脸的原始特征向量
            threshold: 余弦相似度阈值，低于此值视为未匹配
            k:         返回前 k 个结果（当前固定返回最优的一个）

        返回:
            匹配成功返回 FaceMatch，否则返回 None
        """
        vector = _normalize(embedding).reshape(1, -1)
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return None
            if vector.shape[1] != self.index.d:
                raise ValueError(
                    f"特征向量维度不匹配: 期望 {self.index.d}，实际 {vector.shape[1]}"
                )

            # FAISS 内积搜索
            scores, indices = self.index.search(vector.astype(np.float32), k)
            best_index = int(indices[0][0])
            best_score = float(scores[0][0])
            if best_index < 0 or best_score < threshold:
                return None

            person_id = self.ids[best_index]
            record = self.records[person_id]
            return FaceMatch(person_id=record.person_id, name=record.name, score=best_score)

    def save(self) -> None:
        """持久化 FAISS 索引和元数据到磁盘（原子写入）。"""
        with self._lock:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            # 构建元数据
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

            # 原子写入 FAISS 索引文件
            if self.index is not None:
                tmp_index = self.index_path.with_suffix(".faiss.tmp")
                faiss.write_index(self.index, str(tmp_index))
                os.replace(tmp_index, self.index_path)
            # 原子写入元数据 JSON
            _atomic_write_json(self.meta_path, payload)

    def _rebuild_index(self) -> None:
        """从 records 重新构建 FAISS 内积索引。

        IndexFlatIP：暴力搜索，内积值在归一化向量上等价于余弦相似度。
        """
        if self.dimension is None:
            self.index = None
            self.ids = []
            return

        self.ids = sorted(self.records)
        index = faiss.IndexFlatIP(int(self.dimension))
        if self.ids:
            # 按 person_id 排序后的顺序构建向量矩阵
            vectors = np.vstack([self.records[person_id].embedding for person_id in self.ids])
            index.add(vectors.astype(np.float32))
        self.index = index
