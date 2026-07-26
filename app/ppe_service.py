"""YOLO PPE（个人防护装备）检测服务。

使用 ultralytics YOLO 模型对工地摄像头画面进行安全帽和反光衣检测。
检测逻辑：
1. YOLO 推理 → 人员框 + PPE 检测框
2. NMS 去重（重叠的人员框合并）
3. 将 PPE 检测框通过 IoU 匹配到对应的人员框
4. 判断每个人员是否佩戴安全帽/穿着反光衣
5. 输出违规列表（未戴安全帽、未穿反光衣）

模型来源: Vinayakmane47/PPE_detection_YOLO
模型类别: Person, Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest 等
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from app.config import (
    PPE_CONF_THRESHOLD,
    PPE_IMAGE_SIZE,
    PPE_MODEL_PATH,
    PPE_NO_HELMET_CLASSES,
    PPE_NO_VEST_CLASSES,
    PPE_PERSON_CLASSES,
    PPE_WITH_HELMET_CLASSES,
    PPE_WITH_VEST_CLASSES,
)


class PpeServiceError(Exception):
    """PPE 服务异常，携带 HTTP 状态码用于错误响应。"""
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class Detection:
    """YOLO 单次检测结果（不可变）。"""
    bbox: list[float]       # 边界框 [x1, y1, x2, y2]
    confidence: float        # 置信度
    class_id: int            # 类别 ID
    class_name: str          # 类别名称


def _model_path_is_local(path: str) -> bool:
    """判断模型路径是否为本地文件路径（非 HuggingFace/HTTP）。"""
    return not (
        path.startswith("hf://")
        or path.startswith("http://")
        or path.startswith("https://")
        or path.endswith(".pt")
        and "/" not in path
    )


class PpeDetectionService:
    """PPE 检测服务：将 YOLO 检测结果与人员框关联，判定每个人员的 PPE 佩戴状态。

    用法:
        service = PpeDetectionService()
        result = service.detect(image)  # 返回违规列表和人员列表
    """

    def __init__(
        self,
        ppe_model_path: str = PPE_MODEL_PATH,
        ppe_conf: float = PPE_CONF_THRESHOLD,
        image_size: int = PPE_IMAGE_SIZE,
    ) -> None:
        """初始化 PPE 检测服务。

        参数:
            ppe_model_path: YOLO 模型权重文件路径
            ppe_conf:       置信度阈值
            image_size:     推理时的输入图像尺寸
        """
        self.ppe_model_path = ppe_model_path
        self.ppe_conf = ppe_conf
        self.image_size = image_size
        self._model: YOLO | None = None
        self._lock = threading.RLock()  # 模型加载和推理需要线程安全

    def detect(self, image: np.ndarray) -> dict[str, Any]:
        """对输入图片进行 PPE 检测。

        参数:
            image: OpenCV 格式的 BGR 图片 (numpy array)

        返回:
            {
                "count":      违规人数,
                "violations": [违规详情列表],
                "persons":    [每个人员的 PPE 状态],
                "detections": [所有 YOLO 原始检测结果],
                "model":      {模型信息},
            }
        """
        # YOLO 推理
        detections = self._predict(image)

        # 从检测结果中分离人员框和 PPE 检测框
        person_detections = _suppress_overlapping_persons(
            [
                item
                for item in detections
                if item.class_name.strip().lower() in PPE_PERSON_CLASSES
            ]
        )
        ppe_detections = [
            item for item in detections if item.class_name.strip().lower() not in PPE_PERSON_CLASSES
        ]

        persons = []
        violations = []
        for person in person_detections:
            # 找出落在当前人员框内的所有 PPE 检测结果
            assigned = _detections_inside(person.bbox, ppe_detections)

            # 分别筛选安全帽和反光衣的检测结果
            helmet_items = [
                item
                for item in assigned
                if item.class_name.strip().lower() in PPE_WITH_HELMET_CLASSES
            ]
            no_helmet_items = [
                item for item in assigned if item.class_name.strip().lower() in PPE_NO_HELMET_CLASSES
            ]
            vest_items = [
                item
                for item in assigned
                if item.class_name.strip().lower() in PPE_WITH_VEST_CLASSES
            ]
            no_vest_items = [
                item for item in assigned if item.class_name.strip().lower() in PPE_NO_VEST_CLASSES
            ]

            # 以置信度最高的检测结果决定穿戴状态
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

            # 判断是否有 PPE 缺失（违规）
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
        """使用 YOLO 模型进行单次推理。"""
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

        # 将 YOLO 结果转为 Detection 对象列表
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
        """获取 YOLO 模型实例（懒加载 + 线程安全）。"""
        with self._lock:
            if self._model is None:
                self._ensure_model_exists(self.ppe_model_path)
                self._model = YOLO(self.ppe_model_path)
            return self._model

    def _ensure_model_exists(self, model_path: str) -> None:
        """检查模型文件是否存在，不存在时抛出明确错误。"""
        if _model_path_is_local(model_path) and not Path(model_path).exists():
            raise PpeServiceError(
                f"PPE 模型文件不存在: {model_path}。请下载 Vinayakmane47/PPE_detection_YOLO "
                "ppe.pt 或设置 PPE_MODEL_PATH 环境变量。",
                status_code=503,
            )

    def _names(self) -> dict[int, str]:
        """获取 YOLO 模型的类别 ID → 类别名映射。"""
        model = self._get_model()
        return {int(key): str(value) for key, value in model.names.items()}


# ---- 辅助函数 ----


def _status(
    ok_items: list[Detection],
    missing_items: list[Detection],
    ok_label: str,
    missing_label: str,
    unknown_label: str,
) -> str:
    """根据检测结果的最高置信度判定穿戴状态。

    将合格和违规的检测结果按置信度比较，取置信度最高的一方作为判定结果。
    """
    candidates = [(item.confidence, ok_label) for item in ok_items]
    candidates.extend((item.confidence, missing_label) for item in missing_items)
    if not candidates:
        return unknown_label
    return max(candidates, key=lambda item: item[0])[1]


def _detection_payload(detection: Detection) -> dict[str, Any]:
    """将 Detection 对象序列化为字典。"""
    return {
        "bbox": detection.bbox,
        "confidence": detection.confidence,
        "class_id": detection.class_id,
        "class_name": detection.class_name,
    }


def _detections_inside(person_bbox: list[float], detections: list[Detection]) -> list[Detection]:
    """筛选出中心点落在人员框内的 PPE 检测结果。"""
    return [item for item in detections if _point_inside_bbox(_bbox_center(item.bbox), person_bbox)]


def _suppress_overlapping_persons(persons: list[Detection], iou_threshold: float = 0.55) -> list[Detection]:
    """NMS（非极大值抑制）：去除重叠度过高的人员框，保留置信度高的。

    这解决了 YOLO 可能对同一人检测出多个重叠框的问题。
    """
    kept: list[Detection] = []
    for person in sorted(persons, key=lambda item: item.confidence, reverse=True):
        # 与所有已保留的人员框比较 IoU
        if all(_iou(person.bbox, kept_person.bbox) < iou_threshold for kept_person in kept):
            kept.append(person)
    return kept


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    """计算边界框的中心点 (x_center, y_center)。"""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _point_inside_bbox(point: tuple[float, float], bbox: list[float]) -> bool:
    """判断点是否在边界框内。"""
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _iou(a: list[float], b: list[float]) -> float:
    """计算两个边界框的 IoU（交并比）。

    IoU = 交集面积 / 并集面积，值域 [0, 1]，越大表示重叠越多。
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    # 交集矩形的坐标
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
