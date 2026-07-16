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
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class DetectedFace:
    bbox: list[float]
    detection_score: float
    embedding: np.ndarray


class InsightFaceEngine:
    def __init__(self) -> None:
        self._app: FaceAnalysis | None = None
        self._prepared_det_size: tuple[int, int] | None = None
        self._lock = threading.RLock()

    def get_app(self) -> FaceAnalysis:
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
        det_size = self._det_size_for_image(image)
        with self._lock:
            app = self.get_app()
            if self._prepared_det_size != det_size:
                app.prepare(ctx_id=INSIGHTFACE_CTX_ID, det_size=det_size)
                self._prepared_det_size = det_size
            faces = app.get(image)
        detected: list[DetectedFace] = []
        for face in faces:
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
        detected.sort(key=lambda face: _bbox_area(face.bbox), reverse=True)
        return detected

    def _det_size_for_image(self, image: np.ndarray) -> tuple[int, int]:
        height, width = image.shape[:2]
        if max(height, width) <= 320:
            return INSIGHTFACE_SMALL_DET_SIZE
        return INSIGHTFACE_DET_SIZE


class FaceRecognitionService:
    def __init__(
        self,
        store: FaissFaceStore,
        engine: InsightFaceEngine | None = None,
        threshold: float = FACE_MATCH_THRESHOLD,
    ) -> None:
        self.store = store
        self.engine = engine or InsightFaceEngine()
        self.threshold = threshold

    def register(self, person_id: str, name: str | None, image: np.ndarray) -> dict[str, Any]:
        faces = self.engine.detect(image)
        if not faces:
            raise FaceServiceError("no face detected in image", status_code=422)
        if len(faces) > 1:
            raise FaceServiceError(
                "multiple faces detected in registration image; use a single-person image",
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
        faces = self.engine.detect(image)
        if not faces:
            return {
                "recognized": False,
                "id": None,
                "name": None,
                "match_score": None,
                "face_bbox": None,
                "face_detection_score": None,
                "reason": "no_face_detected",
            }

        identity = self._face_identity_payload(faces[0])
        return {
            "recognized": identity["recognized"],
            "id": identity["id"],
            "name": identity["name"],
            "match_score": identity["match_score"],
            "face_bbox": identity["bbox"],
            "face_detection_score": identity["detection_score"],
            "reason": None if identity["recognized"] else "unknown_face",
        }

    def delete(self, person_id: str) -> dict[str, Any]:
        deleted = self.store.delete(person_id)
        return {
            "id": person_id,
            "deleted": deleted,
            "faces_total": self.store.count,
        }

    def _face_identity_payload(self, face: DetectedFace) -> dict[str, Any]:
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
    if not content:
        raise FaceServiceError("uploaded image is empty", status_code=400)
    array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise FaceServiceError("uploaded file is not a valid image", status_code=400)
    return image


def _bbox_area(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
