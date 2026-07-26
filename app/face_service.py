"""InsightFace 人脸识别服务。

基于 InsightFace（buffalo_l 模型）实现人脸检测、特征提取和身份比对。
支持的功能：
- 人脸注册：检测单张人脸 → 提取嵌入向量 → 存入 FAISS 索引
- 人脸识别：检测所有人脸 → 提取嵌入向量 → FAISS 余弦相似度搜索
- 最佳匹配识别：只返回图中最大人脸的最优匹配结果

技术栈:
    - InsightFace (ArcFace): 人脸检测 + 特征提取
    - FAISS IndexFlatIP: 特征向量内积搜索（归一化后等价于余弦相似度）
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from app.config import (
    FACE_MATCH_THRESHOLD,
    INSIGHTFACE_CTX_ID,
    INSIGHTFACE_DET_SIZE,
    INSIGHTFACE_MODEL,
    INSIGHTFACE_PROVIDERS,
    INSIGHTFACE_ROOT,
    INSIGHTFACE_SMALL_DET_SIZE,
)
from app.face_store import FaissFaceStore


class FaceServiceError(Exception):
    """人脸服务异常。"""
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class DetectedFace:
    """检测到的人脸信息（不可变数据类）。

    属性:
        bbox:            人脸边界框 [x1, y1, x2, y2]
        detection_score: 人脸检测置信度
        embedding:       512 维特征向量
    """
    bbox: list[float]
    detection_score: float
    embedding: np.ndarray


class InsightFaceEngine:
    """InsightFace 引擎封装：负责模型的加载、初始化和人脸检测。

    特点:
        - 懒加载：首次使用时才加载模型
        - 自适应检测尺寸：小图使用较小检测尺寸以加速
        - 线程安全：模型加载和推理使用锁保护
    """

    def __init__(self) -> None:
        self._app: FaceAnalysis | None = None
        self._prepared_det_size: tuple[int, int] | None = None
        self._lock = threading.RLock()

    def get_app(self) -> FaceAnalysis:
        """获取或创建 FaceAnalysis 实例（懒加载）。"""
        if self._app is None:
            with self._lock:
                if self._app is None:
                    app = FaceAnalysis(
                        name=INSIGHTFACE_MODEL,
                        root=INSIGHTFACE_ROOT,
                        providers=INSIGHTFACE_PROVIDERS,
                    )
                    self._app = app
        return self._app

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """检测图片中的人脸并提取特征向量。

        自动根据图片尺寸选择检测分辨率（大图用 640，小图用 320）。
        返回的人脸按面积从大到小排序。
        """
        det_size = self._det_size_for_image(image)
        with self._lock:
            app = self.get_app()
            # 只在检测尺寸变化时才重新 prepare
            if self._prepared_det_size != det_size:
                app.prepare(ctx_id=INSIGHTFACE_CTX_ID, det_size=det_size)
                self._prepared_det_size = det_size
            faces = app.get(image)

        detected: list[DetectedFace] = []
        for face in faces:
            # 优先使用归一化嵌入向量
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = getattr(face, "embedding", None)
            if embedding is None:
                continue
            bbox = np.asarray(face.bbox, dtype=float).round(2).tolist()
            score = float(getattr(face, "det_score", 0.0))
            detected.append(
                DetectedFace(
                    bbox=bbox,
                    detection_score=score,
                    embedding=np.asarray(embedding, dtype=np.float32),
                )
            )
        # 按人脸面积降序排列（通常最大的人脸是目标人物）
        detected.sort(key=lambda face: _bbox_area(face.bbox), reverse=True)
        return detected

    def _det_size_for_image(self, image: np.ndarray) -> tuple[int, int]:
        """根据图片尺寸选择检测分辨率。

        小图（max ≤ 320px）使用 320 分辨率以加速，
        大图使用 640 分辨率以获得更准确的检测。
        """
        height, width = image.shape[:2]
        if max(height, width) <= 320:
            return INSIGHTFACE_SMALL_DET_SIZE
        return INSIGHTFACE_DET_SIZE


class FaceRecognitionService:
    """人脸识别业务服务：组合 InsightFace 引擎和 FAISS 存储。

    用法:
        store = FaissFaceStore(data_dir)
        service = FaceRecognitionService(store)
        service.register("1", "张三", image)    # 注册人脸
        result = service.identify(image)         # 识别所有人脸
        best = service.identify_best(image)      # 识别最佳人脸
    """

    def __init__(
        self,
        store: FaissFaceStore,
        engine: InsightFaceEngine | None = None,
        threshold: float = FACE_MATCH_THRESHOLD,
    ) -> None:
        """初始化人脸识别服务。

        参数:
            store:     FAISS 人脸特征存储
            engine:    InsightFace 引擎，为 None 时自动创建
            threshold: 人脸匹配相似度阈值
        """
        self.store = store
        self.engine = engine or InsightFaceEngine()
        self.threshold = threshold

    def register(self, person_id: str, name: str | None, image: np.ndarray) -> dict[str, Any]:
        """注册人员人脸。

        - 要求图片中恰好只有一张人脸
        - 多人脸或无脸会抛出异常
        - 同一 person_id 重复注册会覆盖旧特征

        参数:
            person_id: 人员唯一ID
            name:      人员姓名，可选
            image:     注册图片（OpenCV BGR 格式）

        返回:
            {"id": person_id, "name": ..., "replaced": bool, "faces_total": int}
        """
        faces = self.engine.detect(image)
        if not faces:
            raise FaceServiceError("未检测到人脸，请使用清晰的正面照片", status_code=422)
        if len(faces) > 1:
            raise FaceServiceError(
                "检测到多张人脸，注册照片中只能有一人",
                status_code=422,
            )

        replaced = self.store.upsert(person_id=person_id, name=name, embedding=faces[0].embedding)
        return {
            "id": person_id,
            "name": (name or person_id).strip() or person_id,
            "replaced": replaced,
            "faces_total": self.store.count,
        }

    def identify(self, image: np.ndarray) -> dict[str, Any]:
        """识别图片中所有人脸。

        返回:
            {
                "count":     检测到的人脸数量,
                "threshold": 匹配阈值,
                "faces":     [{人脸信息 + 匹配身份}, ...],
            }
        """
        faces = self.engine.detect(image)
        results = []
        for face in faces:
            results.append(self._face_identity_payload(face))

        return {
            "count": len(results),
            "threshold": self.threshold,
            "faces": results,
        }

    def identify_best(self, image: np.ndarray) -> dict[str, Any]:
        """识别图片中最大的人脸（通常为主角），返回最佳匹配结果。

        这是 PPE 违规检测中关联人员身份的主要入口。

        返回:
            {
                "recognized":          是否识别成功,
                "id":                  人员ID（识别成功时）,
                "name":                姓名,
                "match_score":         匹配置信度,
                "face_bbox":           人脸框,
                "face_detection_score":人脸检测置信度,
                "reason":              未识别原因,
            }
        """
        faces = self.engine.detect(image)
        if not faces:
            return {
                "recognized": False,
                "id": None,
                "name": None,
                "match_score": None,
                "face_bbox": None,
                "face_detection_score": None,
                "reason": "未检测到人脸",
            }

        identity = self._face_identity_payload(faces[0])
        return {
            "recognized": identity["recognized"],
            "id": identity["id"],
            "name": identity["name"],
            "match_score": identity["match_score"],
            "face_bbox": identity["bbox"],
            "face_detection_score": identity["detection_score"],
            "reason": None if identity["recognized"] else "未匹配到注册人员",
        }

    def delete(self, person_id: str) -> dict[str, Any]:
        """删除人员的人脸记录。"""
        deleted = self.store.delete(person_id)
        return {
            "id": person_id,
            "deleted": deleted,
            "faces_total": self.store.count,
        }

    def _face_identity_payload(self, face: DetectedFace) -> dict[str, Any]:
        """将检测到的人脸与 FAISS 索引比对，返回完整身份信息。"""
        match = self.store.search(face.embedding, threshold=self.threshold)
        item: dict[str, Any] = {
            "bbox": face.bbox,
            "detection_score": face.detection_score,
            "recognized": match is not None,
            "id": None,
            "name": None,
            "match_score": None,
        }
        if match is not None:
            item.update(
                {
                    "id": match.person_id,
                    "name": match.name,
                    "match_score": match.score,
                }
            )
        return item


def decode_image(content: bytes) -> np.ndarray:
    """将图片字节内容解码为 OpenCV BGR 格式。"""
    if not content:
        raise FaceServiceError("上传的图片为空", status_code=400)
    array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise FaceServiceError("上传的文件不是有效的图片格式", status_code=400)
    return image


def _bbox_area(bbox: list[float]) -> float:
    """计算边界框的面积。"""
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
