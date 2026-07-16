from __future__ import annotations

from typing import Any

import numpy as np

from app.face_service import FaceRecognitionService
from app.helmet_service import HelmetDetectionService


class NoHelmetSafetyService:
    def __init__(
        self,
        helmet_service: HelmetDetectionService,
        face_service: FaceRecognitionService,
    ) -> None:
        self.helmet_service = helmet_service
        self.face_service = face_service

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        helmet_result = self.helmet_service.detect(image)
        persons = []
        for person in helmet_result["persons"]:
            item = dict(person)
            item["identity"] = self._identify_person(image, item)
            persons.append(item)

        violations = []
        for violation in helmet_result["violations"]:
            item = dict(violation)
            item["identity"] = self._identity_for_violation(persons, item)
            violations.append(item)

        return {
            "count": len(violations),
            "violations": violations,
            "persons": persons,
            "ppe_detections": helmet_result["detections"],
            "model": helmet_result["model"],
        }

    def _identify_person(self, image: np.ndarray, person: dict[str, Any]) -> dict[str, Any]:
        bbox = person["bbox"]
        crop, offset = _crop_bbox(image, bbox)
        if crop.size == 0:
            return {
                "recognized": False,
                "id": None,
                "name": None,
                "match_score": None,
                "face_bbox": None,
                "face_detection_score": None,
                "reason": "empty_person_crop",
            }

        identity = self.face_service.identify_best(crop)
        if identity["face_bbox"] is not None:
            identity["face_bbox"] = _offset_bbox(identity["face_bbox"], offset)
        return identity

    def _identity_for_violation(
        self,
        persons: list[dict[str, Any]],
        violation: dict[str, Any],
    ) -> dict[str, Any]:
        target_bbox = violation.get("person_bbox") or violation["bbox"]
        for person in persons:
            if person["bbox"] == target_bbox:
                return person["identity"]
        return {
            "recognized": False,
            "id": None,
            "name": None,
            "match_score": None,
            "face_bbox": None,
            "face_detection_score": None,
            "reason": "person_not_found",
        }


def _crop_bbox(image: np.ndarray, bbox: list[float]) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return image[0:0, 0:0], (x1, y1)
    return image[y1:y2, x1:x2], (x1, y1)


def _offset_bbox(bbox: list[float], offset: tuple[int, int]) -> list[float]:
    ox, oy = offset
    x1, y1, x2, y2 = bbox
    return [
        round(float(x1) + ox, 2),
        round(float(y1) + oy, 2),
        round(float(x2) + ox, 2),
        round(float(y2) + oy, 2),
    ]
