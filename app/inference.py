from __future__ import annotations

import hashlib
import base64
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


CAMERA_YOLO_TASK_TYPES = {"camera_yolo", "yolo_detection", "ppe_detection"}
MODEL_VERSION = "buildguard-ai-worker-1"


def process_ai_task(message: dict[str, Any]) -> dict[str, Any]:
    task_type = str(message.get("taskType") or "").strip()
    response = _base_response(message)

    if task_type in CAMERA_YOLO_TASK_TYPES:
        response.update(_process_camera_yolo(message))
    elif task_type.endswith("_prediction"):
        response.update(_process_sequence_prediction(message))
    elif task_type == "face_register":
        response.update(_process_face_register(message))
    elif task_type == "face_recognition":
        response.update(_process_face_recognition(message))
    else:
        response.update(_failed("unsupported", f"unsupported taskType: {task_type or '<empty>'}", "unsupported_task"))

    response["finishedAt"] = _utc_now()
    return response


def _base_response(message: dict[str, Any]) -> dict[str, Any]:
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


def _process_camera_yolo(message: dict[str, Any]) -> dict[str, Any]:
    source_file = message.get("sourceFile") or _payload(message).get("sourceFile")
    image = _load_image(source_file)
    if image is not None:
        real_result = _try_camera_yolo_model(image)
        if real_result is not None:
            return real_result

    fallback = _camera_fallback_result(message, image)
    fallback["rawResult"] = {"mode": "mock_fallback", "sourceFileAvailable": image is not None}
    return fallback


def _try_camera_yolo_model(image: Any) -> dict[str, Any] | None:
    try:
        from app.ppe_service import PpeDetectionService
    except Exception:
        return None

    try:
        result = PpeDetectionService().detect(image)
    except Exception:
        return None

    identities = _identify_faces_for_violations(image)
    detections = []
    for item in result.get("violations", []):
        missing = item.get("missing") or []
        detect_type = "ppe_violation"
        if "helmet" in missing:
            detect_type = "no_helmet"
        elif "vest" in missing:
            detect_type = "no_vest"
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
    payload = _payload(message)
    configured = payload.get("mockDetections") or payload.get("detections")
    if isinstance(configured, list):
        detections = [_normalize_detection(item) for item in configured if isinstance(item, dict)]
    elif payload.get("mockViolation") or payload.get("violationType"):
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


def _process_sequence_prediction(message: dict[str, Any]) -> dict[str, Any]:
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


def _process_face_register(message: dict[str, Any]) -> dict[str, Any]:
    payload = _payload(message)
    personnel_id = payload.get("personnelId")
    if personnel_id is None:
        return _failed("face_register", "personnelId is required")

    image = _load_image(message.get("sourceFile") or payload.get("sourceFile"))
    if image is None:
        return _failed("face_register", "source image is not available")

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


def _identify_faces_for_violations(image: Any) -> list[dict[str, Any]]:
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
    identities.sort(key=lambda item: item.get("matchScore") or 0.0, reverse=True)
    return identities


def _identity_for_bbox(bbox: Any, identities: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not identities:
        return None
    target = _bbox_values(bbox)
    if target is None:
        return identities[0] if identities[0].get("recognized") else None

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

    recognized = [item for item in identities if item.get("recognized")]
    if len(recognized) == 1:
        return recognized[0]
    return None


def _bbox_values(value: Any) -> tuple[float, float, float, float] | None:
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
    bbox = _bbox_values(value)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def _point_in_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _weighted_average_prediction(samples: list[dict[str, Any]], payload: dict[str, Any], device_type: str) -> dict[str, Any]:
    thresholds = _thresholds(device_type, payload)
    predicted: dict[str, float] = {}
    for metric in _numeric_metric_keys(samples):
        values = [_number(sample.get(metric), None) for sample in samples]
        numeric_values = [value for value in values if value is not None]
        if numeric_values:
            predicted[metric] = round(_weighted_average(numeric_values), 4)

    factors = []
    ratios = {}
    for metric, value in predicted.items():
        warn = thresholds.get(metric, {}).get("warn")
        alarm = thresholds.get(metric, {}).get("alarm")
        limit = alarm or warn
        if limit:
            ratios[metric] = round(_ratio(abs(value), float(limit)), 4)
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
        "confidence": 0.72 if len(samples) >= 3 else 0.52,
        "horizonSeconds": int(_number(payload.get("horizonSeconds") or payload.get("horizon_seconds"), 300) or 300),
        "factors": factors,
        "metrics": predicted,
        "ratios": ratios,
        "reason": "weighted average prediction within limits" if risk_level == "normal" else "predicted risk raised by " + ", ".join(factors),
    }


def _sequence_samples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("telemetry", "samples", "records", "window"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if any(isinstance(value, (int, float)) and not isinstance(value, bool) for value in payload.values()):
        return [payload]
    latest = payload.get("latest")
    if isinstance(latest, dict):
        return [latest]
    return []


def _numeric_metric_keys(samples: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for sample in samples:
        for key, value in sample.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                keys.add(str(key))
    return sorted(keys)


def _weighted_average(values: list[float]) -> float:
    weights = list(range(1, len(values) + 1))
    return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)


def _thresholds(device_type: str, payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    defaults: dict[str, dict[str, dict[str, float]]] = {
        "tower_crane": {
            "weight": {"warn": 8.0, "alarm": 10.0},
            "moment": {"warn": 680.0, "alarm": 760.0},
            "windSpeed": {"warn": 10.8, "alarm": 13.8},
            "obliquity": {"warn": 4.0, "alarm": 6.0},
        },
        "environment_sensor": {
            "pm25": {"warn": 60.0, "alarm": 75.0},
            "pm10": {"warn": 120.0, "alarm": 150.0},
            "tsp": {"warn": 240.0, "alarm": 300.0},
            "noise": {"warn": 68.0, "alarm": 75.0},
        },
        "elevator": {
            "loadWeight": {"warn": 800.0, "alarm": 950.0},
            "peopleCount": {"warn": 12.0, "alarm": 15.0},
            "tilt": {"warn": 2.0, "alarm": 3.0},
        },
        "formwork": {
            "settlement": {"warn": 5.0, "alarm": 8.0},
            "pressure": {"warn": 20.0, "alarm": 25.0},
            "xAngle": {"warn": 2.0, "alarm": 3.0},
            "yAngle": {"warn": 2.0, "alarm": 3.0},
        },
        "deep_pit": {
            "waterLevel": {"warn": 2.8, "alarm": 3.2},
            "settlement": {"warn": 6.0, "alarm": 8.0},
            "strain": {"warn": 90.0, "alarm": 110.0},
            "soilPressure": {"warn": 55.0, "alarm": 65.0},
        },
    }
    thresholds = {key: dict(value) for key, value in defaults.get(device_type, {}).items()}
    payload_thresholds = payload.get("thresholds")
    if isinstance(payload_thresholds, dict):
        for metric, config in payload_thresholds.items():
            if isinstance(config, dict):
                thresholds[str(metric)] = {
                    "warn": _number(config.get("warn"), thresholds.get(str(metric), {}).get("warn")),
                    "alarm": _number(config.get("alarm"), thresholds.get(str(metric), {}).get("alarm")),
                }
    return thresholds


def _annotated_image_payload(image: Any | None, detections: list[dict[str, Any]]) -> dict[str, Any] | None:
    if image is None or not detections:
        return None
    try:
        import cv2
    except Exception:
        return None

    try:
        canvas = image.copy()
        height, width = canvas.shape[:2]
        thickness = max(2, round(min(width, height) / 220))
        font_scale = max(0.55, min(width, height) / 1050)
        for detection in detections:
            box = _bbox_xyxy(detection.get("bbox"), width, height)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            color = _annotation_color(str(detection.get("detectType") or detection.get("type") or "unknown"))
            label = _annotation_label(detection)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

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
            overlay = canvas.copy()
            cv2.rectangle(overlay, rect_top_left, rect_bottom_right, (16, 24, 40), -1)
            cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
            cv2.rectangle(canvas, rect_top_left, rect_bottom_right, color, max(1, thickness - 1))
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
    if not isinstance(value, list | tuple) or len(value) < 4:
        return None
    x1 = _number(value[0], None)
    y1 = _number(value[1], None)
    x2 = _number(value[2], None)
    y2 = _number(value[3], None)
    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None
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
    detect_type = str(detection.get("detectType") or detection.get("type") or detection.get("className") or "unknown")
    name = {
        "no_helmet": "no helmet",
        "helmet_missing": "no helmet",
        "no_vest": "no vest",
        "vest_missing": "no vest",
        "smoke": "smoking",
        "smoking": "smoking",
        "fire": "fire",
        "open_fire": "fire",
    }.get(detect_type, detect_type.replace("_", " "))
    confidence = _number(detection.get("confidence"), None)
    if confidence is None:
        return name
    return f"{name} {confidence * 100:.0f}%"


def _annotation_color(detect_type: str) -> tuple[int, int, int]:
    normalized = detect_type.strip().lower()
    if normalized in {"no_helmet", "helmet_missing"}:
        return 68, 68, 239
    if normalized in {"no_vest", "vest_missing"}:
        return 11, 158, 245
    return 246, 130, 59


def _normalize_detection(item: dict[str, Any]) -> dict[str, Any]:
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


def _load_image(source_file: Any) -> Any | None:
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
    if value.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(value, timeout=10) as response:
                data = response.read()
            array = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(array, cv2.IMREAD_COLOR)
        except Exception:
            return None

    path = _local_image_path(value)
    if path is None or not path.exists():
        return None
    return cv2.imread(str(path))


def _local_image_path(source_file: Any) -> Path | None:
    value: Any = source_file
    if isinstance(source_file, dict):
        value = source_file.get("localPath") or source_file.get("path") or source_file.get("url")
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if value.startswith(("http://", "https://", "oss://")):
        return None
    if value.startswith("file://"):
        parsed = urlparse(value)
        return Path(unquote(parsed.path)).expanduser()
    return Path(value).expanduser()


def _payload(message: dict[str, Any]) -> dict[str, Any]:
    payload = message.get("payload")
    return payload if isinstance(payload, dict) else {}


def _model_payload(code: str, version: str, mode: str) -> dict[str, str]:
    return {"code": code, "version": version, "mode": mode, "workerVersion": MODEL_VERSION}


def _number(value: Any, default: float | None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _ratio(value: float, limit: float) -> float:
    if limit <= 0:
        return 0.0
    return max(0.0, value / limit)


def _round_float(value: Any, default: float) -> float:
    parsed = _number(value, default)
    return round(float(parsed if parsed is not None else default), 4)


def _personnel_id(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _failed(model_code: str, message: str, mode: str = "failed") -> dict[str, Any]:
    return {
        "resultStatus": "failed",
        "detections": [],
        "prediction": None,
        "model": _model_payload(model_code, "none", mode),
        "errorMessage": message,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_task_hash(message: dict[str, Any]) -> str:
    text = "|".join(
        str(message.get(key) or "")
        for key in ("messageId", "taskId", "taskType", "deviceCode", "occurredAt")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
