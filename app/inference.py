"""BuildGuard AI 推理任务路由与核心逻辑。

本模块是 AI worker 的推理核心，负责：
1. 根据 taskType 将任务分发到对应的处理函数
2. YOLO PPE 检测（安全帽/反光衣）
3. 时序预测（塔吊/升降机/高支模/深基坑/环境传感器）
4. 人脸注册与识别
5. 检测结果标注图绘制

数据流：RabbitMQ 消息 → process_ai_task() → 具体处理函数 → 统一返回结构
"""

from __future__ import annotations

import base64
import hashlib
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


# 摄像头 YOLO 检测类任务的 taskType 集合
CAMERA_YOLO_TASK_TYPES = {"camera_yolo", "yolo_detection", "ppe_detection"}

# Worker 版本标识，随 AI 结果一起返回，便于排查问题
MODEL_VERSION = "buildguard-ai-worker-1"


def process_ai_task(message: dict[str, Any]) -> dict[str, Any]:
    """AI 任务主入口，根据 taskType 路由到对应的处理器。

    参数:
        message: 来自 RabbitMQ 的 AI 请求消息

    返回:
        统一格式的 AI 结果字典
    """
    task_type = str(message.get("taskType") or "").strip()
    response = _base_response(message)

    if task_type in CAMERA_YOLO_TASK_TYPES:
        # 摄像头图片 PPE 检测（安全帽/反光衣）
        response.update(_process_camera_yolo(message))
    elif task_type.endswith("_prediction"):
        # 设备时序预测（tower_crane_prediction / elevator_prediction 等）
        response.update(_process_sequence_prediction(message))
    elif task_type == "face_register":
        # 人脸注册
        response.update(_process_face_register(message))
    elif task_type == "face_recognition":
        # 人脸识别
        response.update(_process_face_recognition(message))
    else:
        # 不支持的任务类型
        response.update(_failed("unsupported", f"不支持 taskType: {task_type or '<空>'}", "unsupported_task"))

    response["finishedAt"] = _utc_now()
    return response


def _base_response(message: dict[str, Any]) -> dict[str, Any]:
    """构建 AI 结果的基础字段，包含任务标识和关联信息。"""
    return {
        "messageId": str(uuid.uuid4()),
        "requestMessageId": message.get("messageId"),
        "eventType": "ai.result",
        "taskId": message.get("taskId"),
        "taskType": message.get("taskType"),
        "deviceCode": message.get("deviceCode"),
        "deviceType": message.get("deviceType"),
        "occurredAt": message.get("occurredAt"),
        "sentAt": _utc_now(),
    }


# =============================================================================
# 摄像头 YOLO PPE 检测
# =============================================================================


def _process_camera_yolo(message: dict[str, Any]) -> dict[str, Any]:
    """处理摄像头图片的 PPE 检测任务。

    优先使用真实 YOLO 模型推理，模型不可用时降级为 mock 数据。
    检测到违规时自动关联人脸识别结果。
    """
    source_file = message.get("sourceFile") or _payload(message).get("sourceFile")
    image = _load_image(source_file)
    if image is not None:
        # 尝试使用真实模型推理
        real_result = _try_camera_yolo_model(image)
        if real_result is not None:
            return real_result

    # 模型不可用，使用 mock 降级策略
    fallback = _camera_fallback_result(message, image)
    fallback["rawResult"] = {"mode": "mock_fallback", "sourceFileAvailable": image is not None}
    return fallback


def _try_camera_yolo_model(image: Any) -> dict[str, Any] | None:
    """使用真实 YOLO 模型对图片进行 PPE 检测，并关联人脸身份。

    返回:
        成功时返回完整检测结果字典，模型不可用或推理失败时返回 None
    """
    try:
        from app.ppe_service import PpeDetectionService
    except Exception:
        return None

    try:
        result = PpeDetectionService().detect(image)
    except Exception:
        return None

    # 在违规人员的人体框中识别人脸
    identities = _identify_faces_for_violations(image)
    detections = []
    for item in result.get("violations", []):
        missing = item.get("missing") or []
        # 根据缺失的 PPE 类型确定违规类别
        detect_type = "ppe_violation"
        if "helmet" in missing:
            detect_type = "no_helmet"
        elif "vest" in missing:
            detect_type = "no_vest"
        # 尝试将人脸身份匹配到此违规人员上
        identity = _identity_for_bbox(item.get("person_bbox") or item.get("bbox"), identities)
        personnel_id = _personnel_id(identity.get("id")) if identity else None
        detections.append(
            {
                "detectType": detect_type,
                "confidence": _round_float(item.get("confidence"), 0.0),
                "bbox": item.get("bbox"),
                "alarmLevel": "warn",
                "personnelId": personnel_id,
                "personnelName": identity.get("name") if identity else None,
                "attributes": {
                    "helmetStatus": item.get("helmet_status"),
                    "vestStatus": item.get("vest_status"),
                    "missing": missing,
                    "source": "yolo",
                    "identity": identity,
                },
            }
        )

    # 绘制标注图
    annotated_image = _annotated_image_payload(image, detections)
    response = {
        "resultStatus": "success",
        "detections": detections,
        "prediction": None,
        "model": _model_payload("camera_yolo", "yolo", "local_model"),
        "errorMessage": None,
        "rawResult": {
            "persons": result.get("persons", []),
            "ppeDetections": result.get("detections", []),
            "faceIdentities": identities,
            "model": result.get("model", {}),
        },
    }
    if annotated_image is not None:
        response["annotatedImage"] = annotated_image
    return response


def _camera_fallback_result(message: dict[str, Any], image: Any | None = None) -> dict[str, Any]:
    """YOLO 模型不可用时生成 mock 检测结果，保证后端流程可测试。"""
    payload = _payload(message)
    configured = payload.get("mockDetections") or payload.get("detections")
    if isinstance(configured, list):
        detections = [_normalize_detection(item) for item in configured if isinstance(item, dict)]
    elif payload.get("mockViolation") or payload.get("violationType"):
        # 使用 payload 中配置的 mock 违规类型
        detect_type = str(payload.get("violationType") or "no_helmet")
        detections = [
            {
                "detectType": detect_type,
                "confidence": 0.82,
                "bbox": [96.0, 72.0, 252.0, 338.0],
                "alarmLevel": "warn",
                "personnelId": _personnel_id(payload.get("personnelId")),
                "personnelName": None,
                "attributes": {"source": "mock_fallback", "reason": "configured_mock_violation"},
            }
        ]
    else:
        detections = []

    annotated_image = _annotated_image_payload(image, detections)
    response = {
        "resultStatus": "success",
        "detections": detections,
        "prediction": None,
        "model": _model_payload("camera_yolo", "mock-v1", "mock_fallback"),
        "errorMessage": None,
    }
    if annotated_image is not None:
        response["annotatedImage"] = annotated_image
    return response


# =============================================================================
# 设备时序安全预测
# =============================================================================


def _process_sequence_prediction(message: dict[str, Any]) -> dict[str, Any]:
    """处理设备时序预测任务。

    从 payload 中提取历史遥测序列，使用加权平均算法预测未来各指标值，
    并与阈值比较判定风险等级。支持塔吊、升降机、高支模、深基坑、环境传感器。
    """
    payload = _payload(message)
    device_type = str(message.get("deviceType") or payload.get("deviceType") or "").strip()
    samples = _sequence_samples(payload)
    prediction = _weighted_average_prediction(samples, payload, device_type)
    return {
        "resultStatus": "success",
        "detections": [],
        "prediction": prediction,
        "model": _model_payload(str(message.get("taskType") or "sequence_prediction"), "weighted-average-v1", "fallback"),
        "errorMessage": None,
        "rawResult": {"sampleCount": len(samples), "latest": samples[-1] if samples else None},
    }


# =============================================================================
# 人脸注册与识别
# =============================================================================


def _process_face_register(message: dict[str, Any]) -> dict[str, Any]:
    """处理人脸注册任务。

    接收人员图片，提取人脸特征向量，存入 FAISS 索引。
    要求图片中只有一张人脸，多人脸会注册失败。
    """
    payload = _payload(message)
    personnel_id = payload.get("personnelId")
    if personnel_id is None:
        return _failed("face_register", "缺少 personnelId 参数")

    image = _load_image(message.get("sourceFile") or payload.get("sourceFile"))
    if image is None:
        return _failed("face_register", "无法加载注册图片")

    try:
        from app.config import FACE_DATA_DIR
        from app.face_service import FaceRecognitionService
        from app.face_store import FaissFaceStore

        result = FaceRecognitionService(FaissFaceStore(FACE_DATA_DIR)).register(str(personnel_id), None, image)
    except Exception as exc:
        return _failed("face_register", str(exc))

    return {
        "resultStatus": "success",
        "detections": [],
        "prediction": None,
        "model": _model_payload("face_register", "insightface-faiss-v1", "local_model"),
        "errorMessage": None,
        "rawResult": result,
    }


def _process_face_recognition(message: dict[str, Any]) -> dict[str, Any]:
    """处理人脸识别任务。

    检测图片中所有人脸，与 FAISS 索引中的注册人脸比对，
    返回每张人脸的识别结果和匹配身份。
    图片不可用时降级为 mock 透传模式。
    """
    payload = _payload(message)
    image = _load_image(message.get("sourceFile") or payload.get("sourceFile"))
    if image is not None:
        try:
            from app.config import FACE_DATA_DIR
            from app.face_service import FaceRecognitionService
            from app.face_store import FaissFaceStore

            result = FaceRecognitionService(FaissFaceStore(FACE_DATA_DIR)).identify(image)
        except Exception as exc:
            return _failed("face_recognition", str(exc))

        detections = []
        for face in result.get("faces", []):
            detections.append(
                {
                    "detectType": "face_recognition",
                    "confidence": _round_float(face.get("match_score"), 0.0),
                    "bbox": face.get("bbox"),
                    "alarmLevel": "warn",
                    "personnelId": _personnel_id(face.get("id")),
                    "personnelName": None,
                    "attributes": {
                        "recognized": face.get("recognized"),
                        "faceDetectionScore": face.get("detection_score"),
                        "source": "insightface",
                    },
                }
            )
        return {
            "resultStatus": "success",
            "detections": detections,
            "prediction": None,
            "model": _model_payload("face_recognition", "insightface-faiss-v1", "local_model"),
            "errorMessage": None,
            "rawResult": result,
        }

    # 图片不可用时使用 mock 数据
    matches = payload.get("matches") or payload.get("mockMatches") or []
    if not isinstance(matches, list):
        matches = []
    return {
        "resultStatus": "success",
        "detections": [_normalize_detection(item) for item in matches if isinstance(item, dict)],
        "prediction": None,
        "model": _model_payload("face_recognition", "mock-v1", "pass_through"),
        "errorMessage": None,
        "rawResult": {"matches": [item for item in matches if isinstance(item, dict)], "sourceFileAvailable": False},
    }


# =============================================================================
# 人脸身份关联（YOLO 检测结果中嵌入人脸识别）
# =============================================================================


def _identify_faces_for_violations(image: Any) -> list[dict[str, Any]]:
    """对 PPE 违规图片进行人脸识别，返回所有人脸身份列表。

    使用更低的匹配阈值（FACE_VIOLATION_MATCH_THRESHOLD），
    因为摄像头截图通常角度不佳、分辨率有限。
    """
    try:
        from app.config import FACE_DATA_DIR, FACE_VIOLATION_MATCH_THRESHOLD
        from app.face_service import FaceRecognitionService
        from app.face_store import FaissFaceStore
    except Exception:
        return []

    try:
        result = FaceRecognitionService(
            FaissFaceStore(FACE_DATA_DIR),
            threshold=FACE_VIOLATION_MATCH_THRESHOLD,
        ).identify(image)
    except Exception:
        return []

    identities = []
    for face in result.get("faces", []):
        if not isinstance(face, dict):
            continue
        bbox = face.get("bbox")
        identities.append(
            {
                "recognized": bool(face.get("recognized")),
                "id": face.get("id"),
                "name": face.get("name"),
                "matchScore": _round_float(face.get("match_score"), 0.0) if face.get("match_score") is not None else None,
                "detectionScore": _round_float(face.get("detection_score"), 0.0),
                "bbox": bbox,
                "center": _bbox_center(bbox),
                "threshold": FACE_VIOLATION_MATCH_THRESHOLD,
            }
        )
    # 按匹配置信度从高到低排序
    identities.sort(key=lambda item: item.get("matchScore") or 0.0, reverse=True)
    return identities


def _identity_for_bbox(bbox: Any, identities: list[dict[str, Any]]) -> dict[str, Any] | None:
    """将人脸身份匹配到 PPE 违规人员的人体框。

    通过判断人脸中心点是否落在人体检测框内进行匹配。
    如果无法空间匹配，则返回唯一识别到的那张人脸（单人次场景）。
    """
    if not identities:
        return None
    target = _bbox_values(bbox)
    if target is None:
        return identities[0] if identities[0].get("recognized") else None

    # 优先：通过空间位置匹配（人脸中心在人体框内）
    matched = []
    for identity in identities:
        if not identity.get("recognized"):
            continue
        center = identity.get("center")
        if not isinstance(center, tuple):
            continue
        if _point_in_bbox(center, target):
            matched.append(identity)
    if matched:
        return max(matched, key=lambda item: item.get("matchScore") or 0.0)

    # 退而求其次：如果只有一个人被识别到，直接返回
    recognized = [item for item in identities if item.get("recognized")]
    if len(recognized) == 1:
        return recognized[0]
    return None


# =============================================================================
# 时序预测算法
# =============================================================================


def _weighted_average_prediction(samples: list[dict[str, Any]], payload: dict[str, Any], device_type: str) -> dict[str, Any]:
    """加权平均时序预测。

    使用线性递增权重（越新的数据权重越大），预测每个数值指标的下一时刻值，
    然后与对应设备类型的阈值比较，生成风险评分和等级。

    算法公式:
        预测值 = Σ(value_i × weight_i) / Σ(weight_i)
        其中 weight_i = i（即 [1, 2, 3, ..., n]）

    返回:
        riskScore:    风险评分，所有指标 ratio 的最大值，上限 1.5
        riskLevel:    风险等级 normal / warn / alarm
        safe:         是否安全
        metrics:      各指标的预测值
        ratios:       各指标的预测值/阈值比率
        factors:      触发风险的指标名列表
    """
    thresholds = _thresholds(device_type, payload)
    # 对所有数值型指标做加权平均预测
    predicted: dict[str, float] = {}
    for metric in _numeric_metric_keys(samples):
        values = [_number(sample.get(metric), None) for sample in samples]
        numeric_values = [value for value in values if value is not None]
        if numeric_values:
            predicted[metric] = round(_weighted_average(numeric_values), 4)

    # 将预测值与阈值比较，判定风险
    factors = []
    ratios = {}
    for metric, value in predicted.items():
        warn = thresholds.get(metric, {}).get("warn")
        alarm = thresholds.get(metric, {}).get("alarm")
        limit = alarm or warn
        if limit:
            ratios[metric] = round(_ratio(abs(value), float(limit)), 4)
        # 超 alarm 阈值直接标记为 alarm 级别
        if alarm is not None and abs(value) >= alarm:
            factors.append(metric)
        elif warn is not None and abs(value) >= warn:
            factors.append(metric)

    risk_score = max(ratios.values(), default=0.0)
    risk_level = "alarm" if risk_score >= 1.0 else "warn" if risk_score >= 0.8 or factors else "normal"
    return {
        "riskScore": round(min(risk_score, 1.5), 4),
        "riskLevel": risk_level,
        "safe": risk_level == "normal",
        "confidence": 0.72 if len(samples) >= 3 else 0.52,  # 样本越多置信度越高
        "horizonSeconds": int(_number(payload.get("horizonSeconds") or payload.get("horizon_seconds"), 300) or 300),
        "factors": factors,
        "metrics": predicted,
        "ratios": ratios,
        "reason": "所有指标均在安全范围内" if risk_level == "normal" else "预测风险指标: " + ", ".join(factors),
    }


def _sequence_samples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """从 payload 中提取历史遥测序列。

    按优先级查找: telemetry → samples → records → window。
    如果 payload 本身就是单条数据（包含数值字段），直接返回。
    """
    for key in ("telemetry", "samples", "records", "window"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    # payload 本身可能是一个扁平的遥测数据点
    if any(isinstance(value, (int, float)) and not isinstance(value, bool) for value in payload.values()):
        return [payload]
    latest = payload.get("latest")
    if isinstance(latest, dict):
        return [latest]
    return []


def _numeric_metric_keys(samples: list[dict[str, Any]]) -> list[str]:
    """从样本中提取所有数值型指标字段名（去重排序）。"""
    keys: set[str] = set()
    for sample in samples:
        for key, value in sample.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                keys.add(str(key))
    return sorted(keys)


def _weighted_average(values: list[float]) -> float:
    """计算加权平均值，权重线性递增：新数据权重更大。

    公式: Σ(v_i × i) / Σ(i)，其中 i = 1, 2, ..., n
    """
    weights = list(range(1, len(values) + 1))
    return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)


def _thresholds(device_type: str, payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    """获取指定设备类型的预警/报警阈值。

    内置五种设备类型的默认阈值，同时支持通过 payload 传入自定义阈值覆盖。
    阈值参考来源：设备厂家铭牌参数、监管要求及行业保守参考值。
    """
    defaults: dict[str, dict[str, dict[str, float]]] = {
        "tower_crane": {
            "weight": {"warn": 8.0, "alarm": 10.0},       # 吊重 (吨)
            "moment": {"warn": 680.0, "alarm": 760.0},     # 力矩 (kN·m)
            "windSpeed": {"warn": 10.8, "alarm": 13.8},    # 风速 (m/s)，warn≈6级风
            "obliquity": {"warn": 4.0, "alarm": 6.0},       # 倾角 (度)
        },
        "environment_sensor": {
            "pm25": {"warn": 60.0, "alarm": 75.0},          # PM2.5 (μg/m³)
            "pm10": {"warn": 120.0, "alarm": 150.0},        # PM10 (μg/m³)
            "tsp": {"warn": 240.0, "alarm": 300.0},         # TSP (μg/m³)
            "noise": {"warn": 68.0, "alarm": 75.0},          # 噪声 (dB)
        },
        "elevator": {
            "loadWeight": {"warn": 800.0, "alarm": 950.0},  # 载重 (kg)
            "peopleCount": {"warn": 12.0, "alarm": 15.0},    # 人数
            "tilt": {"warn": 2.0, "alarm": 3.0},              # 倾斜角 (度)
        },
        "formwork": {
            "settlement": {"warn": 5.0, "alarm": 8.0},       # 沉降 (mm)
            "pressure": {"warn": 20.0, "alarm": 25.0},        # 立杆压力 (kPa)
            "xAngle": {"warn": 2.0, "alarm": 3.0},           # X轴倾角 (度)
            "yAngle": {"warn": 2.0, "alarm": 3.0},           # Y轴倾角 (度)
        },
        "deep_pit": {
            "waterLevel": {"warn": 2.8, "alarm": 3.2},       # 地下水位 (m)
            "settlement": {"warn": 6.0, "alarm": 8.0},        # 沉降 (mm)
            "strain": {"warn": 90.0, "alarm": 110.0},         # 应变 (με)
            "soilPressure": {"warn": 55.0, "alarm": 65.0},    # 土压力 (kPa)
        },
    }
    thresholds = {key: dict(value) for key, value in defaults.get(device_type, {}).items()}
    # payload 中可以传入自定义阈值覆盖默认值
    payload_thresholds = payload.get("thresholds")
    if isinstance(payload_thresholds, dict):
        for metric, config in payload_thresholds.items():
            if isinstance(config, dict):
                thresholds[str(metric)] = {
                    "warn": _number(config.get("warn"), thresholds.get(str(metric), {}).get("warn")),
                    "alarm": _number(config.get("alarm"), thresholds.get(str(metric), {}).get("alarm")),
                }
    return thresholds


# =============================================================================
# 检测结果标注图绘制
# =============================================================================


def _annotated_image_payload(image: Any | None, detections: list[dict[str, Any]]) -> dict[str, Any] | None:
    """在原始图片上绘制检测框和标签，生成标注图。

    标注内容：检测框 + 违规类型标签 + 置信度百分比。
    颜色方案：未戴安全帽=红色，未穿反光衣=蓝色，其他=橙色。
    返回 base64 编码的 JPEG 图片数据。
    """
    if image is None or not detections:
        return None
    try:
        import cv2
    except Exception:
        return None

    try:
        canvas = image.copy()
        height, width = canvas.shape[:2]
        # 根据图片尺寸自适应线条粗细和字体大小
        thickness = max(2, round(min(width, height) / 220))
        font_scale = max(0.55, min(width, height) / 1050)
        for detection in detections:
            box = _bbox_xyxy(detection.get("bbox"), width, height)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            color = _annotation_color(str(detection.get("detectType") or detection.get("type") or "unknown"))
            label = _annotation_label(detection)
            # 绘制检测框
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

            # 绘制标签背景
            (label_width, label_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness,
            )
            pad_x = 8
            pad_y = 6
            label_x = max(0, min(x1, width - label_width - pad_x * 2))
            label_y = y1 - label_height - baseline - pad_y * 2
            if label_y < 0:
                label_y = min(height - label_height - baseline - pad_y * 2, y1 + 2)
            rect_top_left = (label_x, max(0, label_y))
            rect_bottom_right = (
                min(width - 1, label_x + label_width + pad_x * 2),
                min(height - 1, label_y + label_height + baseline + pad_y * 2),
            )
            # 半透明标签背景
            overlay = canvas.copy()
            cv2.rectangle(overlay, rect_top_left, rect_bottom_right, (16, 24, 40), -1)
            cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
            cv2.rectangle(canvas, rect_top_left, rect_bottom_right, color, max(1, thickness - 1))
            # 绘制标签文字
            text_origin = (label_x + pad_x, label_y + pad_y + label_height)
            cv2.putText(
                canvas,
                label,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        ok, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            return None
        return {
            "contentType": "image/jpeg",
            "fileName": "ai-yolo-annotated.jpg",
            "base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }
    except Exception:
        return None


def _bbox_xyxy(value: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    """将检测框坐标转换为图片上的 (x1, y1, x2, y2) 整数像素坐标。

    自动识别归一化坐标（0~1）并转换到实际像素。
    """
    if not isinstance(value, list | tuple) or len(value) < 4:
        return None
    x1 = _number(value[0], None)
    y1 = _number(value[1], None)
    x2 = _number(value[2], None)
    y2 = _number(value[3], None)
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None
    # 归一化坐标转换为像素坐标
    if x2 <= 1.0 and y2 <= 1.0:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height
    left = round(max(0, min(width - 1, min(x1, x2))))
    top = round(max(0, min(height - 1, min(y1, y2))))
    right = round(max(left + 1, min(width, max(x1, x2))))
    bottom = round(max(top + 1, min(height, max(y1, y2))))
    return int(left), int(top), int(right), int(bottom)


def _annotation_label(detection: dict[str, Any]) -> str:
    """生成检测框标签文字，格式为 "违规类型 置信度%"。"""
    detect_type = str(detection.get("detectType") or detection.get("type") or detection.get("className") or "unknown")
    # 违规类型中文映射
    name = {
        "no_helmet": "未戴安全帽",
        "helmet_missing": "未戴安全帽",
        "no_vest": "未穿反光衣",
        "vest_missing": "未穿反光衣",
        "smoke": "吸烟",
        "smoking": "吸烟",
        "fire": "明火",
        "open_fire": "明火",
    }.get(detect_type, detect_type.replace("_", " "))
    confidence = _number(detection.get("confidence"), None)
    if confidence is None:
        return name
    return f"{name} {confidence * 100:.0f}%"


def _annotation_color(detect_type: str) -> tuple[int, int, int]:
    """根据违规类型返回对应的 BGR 颜色值。

    未戴安全帽 → 红色(239, 68, 68)
    未穿反光衣 → 蓝色(245, 158, 11)
    其他       → 橙色(59, 130, 246)
    """
    normalized = detect_type.strip().lower()
    if normalized in {"no_helmet", "helmet_missing"}:
        return 68, 68, 239    # BGR 红色
    if normalized in {"no_vest", "vest_missing"}:
        return 11, 158, 245   # BGR 蓝色
    return 246, 130, 59       # BGR 橙色


def _normalize_detection(item: dict[str, Any]) -> dict[str, Any]:
    """将检测结果标准化为统一格式。"""
    detect_type = str(item.get("detectType") or item.get("type") or item.get("className") or "unknown")
    return {
        "detectType": detect_type,
        "confidence": _round_float(item.get("confidence"), 0.8),
        "bbox": item.get("bbox") or item.get("box"),
        "alarmLevel": item.get("alarmLevel") or "warn",
        "personnelId": _personnel_id(item.get("personnelId")),
        "personnelName": item.get("personnelName"),
        "attributes": item.get("attributes") if isinstance(item.get("attributes"), dict) else {},
    }


# =============================================================================
# 图片加载
# =============================================================================


def _load_image(source_file: Any) -> Any | None:
    """加载图片，支持本地路径、HTTP/HTTPS URL。

    返回 OpenCV 格式的 numpy 数组，加载失败返回 None。
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return None

    value: Any = source_file
    if isinstance(source_file, dict):
        value = source_file.get("localPath") or source_file.get("path") or source_file.get("url")
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    # HTTP/HTTPS 远程图片
    if value.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(value, timeout=10) as response:
                data = response.read()
            array = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(array, cv2.IMREAD_COLOR)
        except Exception:
            return None
    # 本地图片
    path = _local_image_path(value)
    if path is None or not path.exists():
        return None
    return cv2.imread(str(path))


def _local_image_path(source_file: Any) -> Path | None:
    """从 sourceFile 中提取本地文件路径。"""
    value: Any = source_file
    if isinstance(source_file, dict):
        value = source_file.get("localPath") or source_file.get("path") or source_file.get("url")
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    # 跳过远程地址
    if value.startswith(("http://", "https://", "oss://")):
        return None
    if value.startswith("file://"):
        parsed = urlparse(value)
        return Path(unquote(parsed.path)).expanduser()
    return Path(value).expanduser()


# =============================================================================
# 辅助工具函数
# =============================================================================


def _payload(message: dict[str, Any]) -> dict[str, Any]:
    """从消息中提取 payload，确保返回字典。"""
    payload = message.get("payload")
    return payload if isinstance(payload, dict) else {}


def _model_payload(code: str, version: str, mode: str) -> dict[str, str]:
    """构建模型信息字段。"""
    return {"code": code, "version": version, "mode": mode, "workerVersion": MODEL_VERSION}


def _number(value: Any, default: float | None) -> float | None:
    """安全地将值转换为 float，失败返回默认值。"""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _ratio(value: float, limit: float) -> float:
    """计算指标值与阈值的比率，用于衡量超标程度。"""
    if limit <= 0:
        return 0.0
    return max(0.0, value / limit)


def _round_float(value: Any, default: float) -> float:
    """安全转换为 float 并四舍五入到 4 位小数。"""
    parsed = _number(value, default)
    return round(float(parsed if parsed is not None else default), 4)


def _personnel_id(value: Any) -> int | None:
    """安全地将值转换为 personnel_id (int)。"""
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _failed(model_code: str, message: str, mode: str = "failed") -> dict[str, Any]:
    """构建失败结果。"""
    return {
        "resultStatus": "failed",
        "detections": [],
        "prediction": None,
        "model": _model_payload(model_code, "none", mode),
        "errorMessage": message,
    }


def _utc_now() -> str:
    """获取当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat()


def _bbox_values(value: Any) -> tuple[float, float, float, float] | None:
    """标准化边界框为 (x1, y1, x2, y2) 格式。"""
    if not isinstance(value, list | tuple) or len(value) < 4:
        return None
    x1 = _number(value[0], None)
    y1 = _number(value[1], None)
    x2 = _number(value[2], None)
    y2 = _number(value[3], None)
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _bbox_center(value: Any) -> tuple[float, float] | None:
    """计算边界框的中心点坐标。"""
    bbox = _bbox_values(value)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _point_in_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    """判断点是否在边界框内。"""
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def stable_task_hash(message: dict[str, Any]) -> str:
    """生成 AI 任务的稳定哈希值，用于幂等判断。"""
    text = "|".join(
        str(message.get(key) or "")
        for key in ("messageId", "taskId", "taskType", "deviceCode", "occurredAt")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
