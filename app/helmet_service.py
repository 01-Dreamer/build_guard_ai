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
    HELMET_PERSON_CONF_THRESHOLD,
    HELMET_PERSON_MODEL_PATH,
    HELMET_PPE_MODEL_PATH,
    HELMET_WITH_HELMET_CLASSES,
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
    """Detect people without helmets by assigning NO-Hardhat heads to person boxes."""

    def __init__(
        self,
        ppe_model_path: str = HELMET_PPE_MODEL_PATH,
        person_model_path: str = HELMET_PERSON_MODEL_PATH,
        ppe_conf: float = HELMET_CONF_THRESHOLD,
        person_conf: float = HELMET_PERSON_CONF_THRESHOLD,
        image_size: int = HELMET_IMAGE_SIZE,
    ) -> None:
        self.ppe_model_path = ppe_model_path
        self.person_model_path = person_model_path
        self.ppe_conf = ppe_conf
        self.person_conf = person_conf
        self.image_size = image_size
        self._ppe_model: YOLO | None = None
        self._person_model: YOLO | None = None
        self._lock = threading.RLock()

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        ppe_detections = self._predict_ppe(image)
        person_detections = self._predict_persons(image)

        no_helmet_heads = [
            item for item in ppe_detections if item.class_name.strip().lower() in HELMET_NO_HELMET_CLASSES
        ]
        helmet_heads = [
            item for item in ppe_detections if item.class_name.strip().lower() in HELMET_WITH_HELMET_CLASSES
        ]

        persons = []
        for person in person_detections:
            no_helmet_matches = _detections_inside(person.bbox, no_helmet_heads)
            helmet_matches = _detections_inside(person.bbox, helmet_heads)
            status = "unknown"
            if no_helmet_matches:
                status = "no_helmet"
            elif helmet_matches:
                status = "with_helmet"
            persons.append(
                {
                    "bbox": person.bbox,
                    "confidence": person.confidence,
                    "helmet_status": status,
                    "no_helmet_heads": [_detection_payload(item) for item in no_helmet_matches],
                    "helmet_heads": [_detection_payload(item) for item in helmet_matches],
                }
            )

        violations = []
        used_person_indexes: set[int] = set()
        for head in no_helmet_heads:
            person_index = _find_person_for_head(head, person_detections)
            if person_index is not None:
                if person_index in used_person_indexes:
                    continue
                used_person_indexes.add(person_index)
                person = person_detections[person_index]
                violations.append(
                    {
                        "bbox": person.bbox,
                        "person_bbox": person.bbox,
                        "head_bbox": head.bbox,
                        "confidence": min(person.confidence, head.confidence),
                        "person_confidence": person.confidence,
                        "head_confidence": head.confidence,
                        "head_class": head.class_name,
                        "matched_person": True,
                    }
                )
            else:
                violations.append(
                    {
                        "bbox": head.bbox,
                        "person_bbox": None,
                        "head_bbox": head.bbox,
                        "confidence": head.confidence,
                        "person_confidence": None,
                        "head_confidence": head.confidence,
                        "head_class": head.class_name,
                        "matched_person": False,
                    }
                )

        return {
            "count": len(violations),
            "violations": violations,
            "persons": persons,
            "detections": [_detection_payload(item) for item in ppe_detections],
            "model": {
                "ppe": self.ppe_model_path,
                "person": self.person_model_path,
                "ppe_classes": self._ppe_names(),
                "person_classes": self._person_names(),
                "ppe_conf": self.ppe_conf,
                "person_conf": self.person_conf,
            },
        }

    def _predict_ppe(self, image: np.ndarray) -> list[Detection]:
        model = self._get_ppe_model()
        return self._predict(model, image, conf=self.ppe_conf)

    def _predict_persons(self, image: np.ndarray) -> list[Detection]:
        model = self._get_person_model()
        detections = self._predict(model, image, conf=self.person_conf, classes=[0])
        return [item for item in detections if item.class_name.lower() == "person" or item.class_id == 0]

    def _predict(
        self,
        model: YOLO,
        image: np.ndarray,
        conf: float,
        classes: list[int] | None = None,
    ) -> list[Detection]:
        with self._lock:
            results = model.predict(
                source=image,
                conf=conf,
                imgsz=self.image_size,
                classes=classes,
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

    def _get_ppe_model(self) -> YOLO:
        with self._lock:
            if self._ppe_model is None:
                self._ensure_model_exists(self.ppe_model_path, "PPE")
                self._ppe_model = YOLO(self.ppe_model_path)
            return self._ppe_model

    def _get_person_model(self) -> YOLO:
        with self._lock:
            if self._person_model is None:
                self._person_model = YOLO(self.person_model_path)
            return self._person_model

    def _ensure_model_exists(self, model_path: str, label: str) -> None:
        if _model_path_is_local(model_path) and not Path(model_path).exists():
            raise HelmetServiceError(
                f"{label} model not found: {model_path}. Set HELMET_PPE_MODEL_PATH to a trained helmet model.",
                status_code=503,
            )

    def _ppe_names(self) -> dict[int, str]:
        model = self._get_ppe_model()
        return {int(key): str(value) for key, value in model.names.items()}

    def _person_names(self) -> dict[int, str]:
        model = self._get_person_model()
        return {int(key): str(value) for key, value in model.names.items()}


def _detection_payload(detection: Detection) -> dict[str, Any]:
    return {
        "bbox": detection.bbox,
        "confidence": detection.confidence,
        "class_id": detection.class_id,
        "class_name": detection.class_name,
    }


def _detections_inside(person_bbox: list[float], detections: list[Detection]) -> list[Detection]:
    return [item for item in detections if _point_inside_bbox(_bbox_center(item.bbox), person_bbox)]


def _find_person_for_head(head: Detection, persons: list[Detection]) -> int | None:
    center = _bbox_center(head.bbox)
    containing = [
        (index, _bbox_area(person.bbox))
        for index, person in enumerate(persons)
        if _point_inside_bbox(center, person.bbox)
    ]
    if containing:
        return min(containing, key=lambda item: item[1])[0]

    overlaps = [
        (index, _intersection_ratio(head.bbox, person.bbox))
        for index, person in enumerate(persons)
    ]
    overlaps = [item for item in overlaps if item[1] > 0]
    if not overlaps:
        return None
    return max(overlaps, key=lambda item: item[1])[0]


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _point_inside_bbox(point: tuple[float, float], bbox: list[float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _bbox_area(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection_ratio(inner_bbox: list[float], outer_bbox: list[float]) -> float:
    ax1, ay1, ax2, ay2 = inner_bbox
    bx1, by1, bx2, by2 = outer_bbox
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = _bbox_area([ix1, iy1, ix2, iy2])
    area = _bbox_area(inner_bbox)
    if area <= 0:
        return 0.0
    return intersection / area
