from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from app.config import (
    HELMET_CONF_THRESHOLD,
    HELMET_IMAGE_SIZE,
    HELMET_NO_HELMET_CLASSES,
    HELMET_NO_VEST_CLASSES,
    HELMET_PERSON_CLASSES,
    HELMET_PPE_MODEL_PATH,
    HELMET_WITH_HELMET_CLASSES,
    HELMET_WITH_VEST_CLASSES,
)


class HelmetServiceError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class Detection:
    bbox: list[float]
    confidence: float
    class_id: int
    class_name: str


def _model_path_is_local(path: str) -> bool:
    return not (
        path.startswith("hf://")
        or path.startswith("http://")
        or path.startswith("https://")
        or path.endswith(".pt")
        and "/" not in path
    )


class HelmetDetectionService:
    """Assign combined PPE detections to person boxes.

    Expected classes for the default model include:
    Person, Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest.
    """

    def __init__(
        self,
        ppe_model_path: str = HELMET_PPE_MODEL_PATH,
        ppe_conf: float = HELMET_CONF_THRESHOLD,
        image_size: int = HELMET_IMAGE_SIZE,
    ) -> None:
        self.ppe_model_path = ppe_model_path
        self.ppe_conf = ppe_conf
        self.image_size = image_size
        self._model: YOLO | None = None
        self._lock = threading.RLock()

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        detections = self._predict(image)
        person_detections = _suppress_overlapping_persons(
            [
                item
                for item in detections
                if item.class_name.strip().lower() in HELMET_PERSON_CLASSES
            ]
        )
        ppe_detections = [
            item for item in detections if item.class_name.strip().lower() not in HELMET_PERSON_CLASSES
        ]

        persons = []
        violations = []
        for person in person_detections:
            assigned = _detections_inside(person.bbox, ppe_detections)
            helmet_items = [
                item
                for item in assigned
                if item.class_name.strip().lower() in HELMET_WITH_HELMET_CLASSES
            ]
            no_helmet_items = [
                item for item in assigned if item.class_name.strip().lower() in HELMET_NO_HELMET_CLASSES
            ]
            vest_items = [
                item
                for item in assigned
                if item.class_name.strip().lower() in HELMET_WITH_VEST_CLASSES
            ]
            no_vest_items = [
                item for item in assigned if item.class_name.strip().lower() in HELMET_NO_VEST_CLASSES
            ]

            helmet_status = _status(
                ok_items=helmet_items,
                missing_items=no_helmet_items,
                ok_label="with_helmet",
                missing_label="no_helmet",
                unknown_label="unknown",
            )
            vest_status = _status(
                ok_items=vest_items,
                missing_items=no_vest_items,
                ok_label="with_vest",
                missing_label="no_vest",
                unknown_label="unknown",
            )

            person_payload = {
                "bbox": person.bbox,
                "confidence": person.confidence,
                "helmet_status": helmet_status,
                "vest_status": vest_status,
                "helmet_detections": [_detection_payload(item) for item in helmet_items],
                "no_helmet_detections": [_detection_payload(item) for item in no_helmet_items],
                "vest_detections": [_detection_payload(item) for item in vest_items],
                "no_vest_detections": [_detection_payload(item) for item in no_vest_items],
                "ppe_detections": [_detection_payload(item) for item in assigned],
            }
            persons.append(person_payload)

            missing = []
            if helmet_status == "no_helmet":
                missing.append("helmet")
            if vest_status == "no_vest":
                missing.append("vest")
            if missing:
                violations.append(
                    {
                        "bbox": person.bbox,
                        "person_bbox": person.bbox,
                        "confidence": person.confidence,
                        "person_confidence": person.confidence,
                        "helmet_status": helmet_status,
                        "vest_status": vest_status,
                        "missing": missing,
                        "matched_person": True,
                    }
                )

        return {
            "count": len(violations),
            "violations": violations,
            "persons": persons,
            "detections": [_detection_payload(item) for item in detections],
            "model": {
                "ppe": self.ppe_model_path,
                "ppe_classes": self._names(),
                "ppe_conf": self.ppe_conf,
            },
        }

    def _predict(self, image: np.ndarray) -> list[Detection]:
        model = self._get_model()
        with self._lock:
            results = model.predict(
                source=image,
                conf=self.ppe_conf,
                imgsz=self.image_size,
                verbose=False,
            )

        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names

        detections = []
        for bbox, score, class_id in zip(xyxy, confs, class_ids):
            detections.append(
                Detection(
                    bbox=[round(float(value), 2) for value in bbox.tolist()],
                    confidence=float(score),
                    class_id=int(class_id),
                    class_name=str(names.get(int(class_id), int(class_id))),
                )
            )
        return detections

    def _get_model(self) -> YOLO:
        with self._lock:
            if self._model is None:
                self._ensure_model_exists(self.ppe_model_path)
                self._model = YOLO(self.ppe_model_path)
            return self._model

    def _ensure_model_exists(self, model_path: str) -> None:
        if _model_path_is_local(model_path) and not Path(model_path).exists():
            raise HelmetServiceError(
                f"PPE model not found: {model_path}. Download Vinayakmane47/PPE_detection_YOLO "
                "ppe.pt there or set HELMET_PPE_MODEL_PATH.",
                status_code=503,
            )

    def _names(self) -> dict[int, str]:
        model = self._get_model()
        return {int(key): str(value) for key, value in model.names.items()}


def _status(
    ok_items: list[Detection],
    missing_items: list[Detection],
    ok_label: str,
    missing_label: str,
    unknown_label: str,
) -> str:
    candidates = [(item.confidence, ok_label) for item in ok_items]
    candidates.extend((item.confidence, missing_label) for item in missing_items)
    if not candidates:
        return unknown_label
    return max(candidates, key=lambda item: item[0])[1]


def _detection_payload(detection: Detection) -> dict[str, Any]:
    return {
        "bbox": detection.bbox,
        "confidence": detection.confidence,
        "class_id": detection.class_id,
        "class_name": detection.class_name,
    }


def _detections_inside(person_bbox: list[float], detections: list[Detection]) -> list[Detection]:
    return [item for item in detections if _point_inside_bbox(_bbox_center(item.bbox), person_bbox)]


def _suppress_overlapping_persons(persons: list[Detection], iou_threshold: float = 0.55) -> list[Detection]:
    kept: list[Detection] = []
    for person in sorted(persons, key=lambda item: item.confidence, reverse=True):
        if all(_iou(person.bbox, kept_person.bbox) < iou_threshold for kept_person in kept):
            kept.append(person)
    return kept


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _point_inside_bbox(point: tuple[float, float], bbox: list[float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union
