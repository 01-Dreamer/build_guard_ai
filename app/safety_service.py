"""综合安全分析服务。

将 PPE 检测和人员身份识别组合在一起：
1. YOLO 检测安全帽/反光衣违规
2. 对每个违规人员裁剪人体区域
3. 在人脸区域上进行人脸识别
4. 将人脸身份信息附加到违规记录上

这样摄像头画面中"谁没戴安全帽"就能直接关联到具体人员。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.face_service import FaceRecognitionService
from app.ppe_service import PpeDetectionService


class PpeSafetyService:
    """PPE 安全综合服务：检测 + 识别一体化。

    用法:
        service = PpeSafetyService(ppe_service, face_service)
        result = service.detect(image)
        # 每个违规记录中自动包含人员身份 identity 信息
    """

    def __init__(
        self,
        ppe_service: PpeDetectionService,
        face_service: FaceRecognitionService,
    ) -> None:
        self.ppe_service = ppe_service
        self.face_service = face_service

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        """执行检测 + 识别，返回带人员身份的违规列表。

        流程:
            1. YOLO PPE 检测 → 人员列表 + 违规列表
            2. 对每个人员裁剪人体区域 → 人脸识别 → 附加身份信息
            3. 将身份信息同步到对应的违规记录上
        """
        ppe_result = self.ppe_service.detect(image)

        # 为每个检测到的人员进行人脸识别
        persons = []
        for person in ppe_result["persons"]:
            item = dict(person)
            item["identity"] = self._identify_person(image, item)
            persons.append(item)

        # 将身份信息附加到违规记录上
        violations = []
        for violation in ppe_result["violations"]:
            item = dict(violation)
            item["identity"] = self._identity_for_violation(persons, item)
            violations.append(item)

        return {
            "count": len(violations),
            "violations": violations,
            "persons": persons,
            "ppe_detections": ppe_result["detections"],
            "model": ppe_result["model"],
        }

    def _identify_person(self, image: np.ndarray, person: dict[str, Any]) -> dict[str, Any]:
        """在人员的人体框内裁剪区域进行人脸识别。

        如果裁剪区域与全图一致（异常情况），返回未识别。
        人脸坐标会偏移回原图坐标系。
        """
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
                "reason": "人体区域裁剪为空",
            }

        identity = self.face_service.identify_best(crop)
        # 将裁剪区域的人脸坐标偏移回原图坐标系
        if identity["face_bbox"] is not None:
            identity["face_bbox"] = _offset_bbox(identity["face_bbox"], offset)
        return identity

    def _identity_for_violation(
        self,
        persons: list[dict[str, Any]],
        violation: dict[str, Any],
    ) -> dict[str, Any]:
        """通过人体框匹配，将人员身份关联到对应的违规记录上。"""
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
            "reason": "未找到对应的人员记录",
        }


def _crop_bbox(image: np.ndarray, bbox: list[float]) -> tuple[np.ndarray, tuple[int, int]]:
    """从图片中裁剪出指定边界框区域。

    返回:
        (裁剪后的图像, 裁剪偏移量 (x_offset, y_offset))
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    # 边界裁剪，确保坐标在图片范围内
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return image[0:0, 0:0], (x1, y1)
    return image[y1:y2, x1:x2], (x1, y1)


def _offset_bbox(bbox: list[float], offset: tuple[int, int]) -> list[float]:
    """将裁剪区域中的坐标偏移回原图坐标系。"""
    ox, oy = offset
    x1, y1, x2, y2 = bbox
    return [
        round(float(x1) + ox, 2),
        round(float(y1) + oy, 2),
        round(float(x2) + ox, 2),
        round(float(y2) + oy, 2),
    ]
