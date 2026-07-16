from __future__ import annotations

import os
from pathlib import Path


def _parse_det_size(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("INSIGHTFACE_DET_SIZE must look like '640,640'")
    return int(parts[0]), int(parts[1])


BASE_DIR = Path(__file__).resolve().parent.parent

FACE_DATA_DIR = Path(os.getenv("FACE_DATA_DIR", BASE_DIR / "data" / "faces"))
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.55"))

HELMET_PPE_MODEL_PATH = os.getenv(
    "HELMET_PPE_MODEL_PATH",
    str(BASE_DIR / "models" / "helmet" / "yolov8m-hard-hat-detection" / "best.pt"),
)
HELMET_PERSON_MODEL_PATH = os.getenv("HELMET_PERSON_MODEL_PATH", "yolov8n.pt")
HELMET_CONF_THRESHOLD = float(os.getenv("HELMET_CONF_THRESHOLD", "0.35"))
HELMET_PERSON_CONF_THRESHOLD = float(os.getenv("HELMET_PERSON_CONF_THRESHOLD", "0.35"))
HELMET_IMAGE_SIZE = int(os.getenv("HELMET_IMAGE_SIZE", "640"))
HELMET_NO_HELMET_CLASSES = {
    item.strip().lower()
    for item in os.getenv("HELMET_NO_HELMET_CLASSES", "NO-Hardhat,NO-Helmet,nohat").split(",")
    if item.strip()
}
HELMET_WITH_HELMET_CLASSES = {
    item.strip().lower()
    for item in os.getenv("HELMET_WITH_HELMET_CLASSES", "Hardhat,Helmet,hat").split(",")
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
