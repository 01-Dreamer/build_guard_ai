from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        return False


load_dotenv()


def _parse_det_size(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("INSIGHTFACE_DET_SIZE must look like '640,640'")
    return int(parts[0]), int(parts[1])


BASE_DIR = Path(__file__).resolve().parent.parent

FACE_DATA_DIR = Path(os.getenv("FACE_DATA_DIR", BASE_DIR / "data" / "faces"))
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.55"))
FACE_VIOLATION_MATCH_THRESHOLD = float(os.getenv("FACE_VIOLATION_MATCH_THRESHOLD", "0.28"))

PPE_MODEL_PATH = os.getenv(
    "PPE_MODEL_PATH",
    str(BASE_DIR / "models" / "ppe" / "ppe-detection-yolo" / "ppe.pt"),
)
PPE_CONF_THRESHOLD = float(os.getenv("PPE_CONF_THRESHOLD", "0.35"))
PPE_IMAGE_SIZE = int(os.getenv("PPE_IMAGE_SIZE", "640"))
PPE_NO_HELMET_CLASSES = {
    item.strip().lower()
    for item in os.getenv("PPE_NO_HELMET_CLASSES", "NO-Hardhat,nohat,NO-Helmet").split(",")
    if item.strip()
}
PPE_WITH_HELMET_CLASSES = {
    item.strip().lower()
    for item in os.getenv("PPE_WITH_HELMET_CLASSES", "Hardhat,hat,Helmet").split(",")
    if item.strip()
}
PPE_NO_VEST_CLASSES = {
    item.strip().lower()
    for item in os.getenv("PPE_NO_VEST_CLASSES", "NO-Safety Vest,novest,NO-Vest").split(",")
    if item.strip()
}
PPE_WITH_VEST_CLASSES = {
    item.strip().lower()
    for item in os.getenv("PPE_WITH_VEST_CLASSES", "Safety Vest,vest").split(",")
    if item.strip()
}
PPE_PERSON_CLASSES = {
    item.strip().lower()
    for item in os.getenv("PPE_PERSON_CLASSES", "Person,person").split(",")
    if item.strip()
}

INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")
INSIGHTFACE_ROOT = os.getenv("INSIGHTFACE_ROOT", "~/.insightface")
INSIGHTFACE_CTX_ID = int(os.getenv("INSIGHTFACE_CTX_ID", "-1"))
INSIGHTFACE_DET_SIZE = _parse_det_size(os.getenv("INSIGHTFACE_DET_SIZE", "640,640"))
INSIGHTFACE_SMALL_DET_SIZE = _parse_det_size(os.getenv("INSIGHTFACE_SMALL_DET_SIZE", "320,320"))
INSIGHTFACE_PROVIDERS = [
    provider.strip()
    for provider in os.getenv("INSIGHTFACE_PROVIDERS", "CPUExecutionProvider").split(",")
    if provider.strip()
]

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@110.41.166.11:5672/%2F")
AI_EXCHANGE = os.getenv("AI_EXCHANGE", "buildguard.ai")
AI_REQUEST_QUEUE = os.getenv("AI_REQUEST_QUEUE", "buildguard.ai.request")
AI_RESULT_QUEUE = os.getenv("AI_RESULT_QUEUE", "buildguard.ai.result")
AI_RESULT_ROUTING_KEY = os.getenv("AI_RESULT_ROUTING_KEY", "ai.result")
AI_WORKER_PREFETCH = int(os.getenv("AI_WORKER_PREFETCH", "1"))
AI_WORKER_RECONNECT_SECONDS = float(os.getenv("AI_WORKER_RECONNECT_SECONDS", "5"))
