# BuildGuard AI 服务配置
# 所有配置项均支持通过环境变量覆盖，方便在不同环境下部署

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    # 运行时如果没有安装 python-dotenv 就跳过，不影响主流程
    def load_dotenv() -> bool:
        return False


# 加载 .env 文件中的环境变量
load_dotenv()


def _parse_det_size(value: str) -> tuple[int, int]:
    """解析人脸检测尺寸，格式为 "640,640"。"""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("INSIGHTFACE_DET_SIZE 格式必须为 '宽,高'，例如 '640,640'")
    return int(parts[0]), int(parts[1])


# ---- 项目根目录 ----
BASE_DIR = Path(__file__).resolve().parent.parent

# =============================================================================
# 人脸识别相关配置
# =============================================================================

# 人脸数据持久化目录，存储 FAISS 索引和元数据
FACE_DATA_DIR = Path(os.getenv("FACE_DATA_DIR", BASE_DIR / "data" / "faces"))

# 人脸匹配余弦相似度阈值，高于此值视为识别成功
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.55"))

# 违规检测中的人脸匹配阈值，比常规识别阈值更低，因为摄像头截图质量较差
FACE_VIOLATION_MATCH_THRESHOLD = float(os.getenv("FACE_VIOLATION_MATCH_THRESHOLD", "0.28"))

# =============================================================================
# InsightFace 引擎配置
# =============================================================================

# 模型包名称，buffalo_l 是轻量级通用模型，平衡精度与速度
INSIGHTFACE_MODEL = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")

# 模型缓存目录
INSIGHTFACE_ROOT = os.getenv("INSIGHTFACE_ROOT", "~/.insightface")

# 设备上下文：-1 表示 CPU，0 表示 GPU
INSIGHTFACE_CTX_ID = int(os.getenv("INSIGHTFACE_CTX_ID", "-1"))

# 默认检测尺寸
INSIGHTFACE_DET_SIZE = _parse_det_size(os.getenv("INSIGHTFACE_DET_SIZE", "640,640"))

# 小图模式检测尺寸（宽或高不超过 320px 时使用）
INSIGHTFACE_SMALL_DET_SIZE = _parse_det_size(os.getenv("INSIGHTFACE_SMALL_DET_SIZE", "320,320"))

# ONNX Runtime 推理提供器，CPUExecutionProvider 为 CPU 推理
INSIGHTFACE_PROVIDERS = [
    provider.strip()
    for provider in os.getenv("INSIGHTFACE_PROVIDERS", "CPUExecutionProvider").split(",")
    if provider.strip()
]

# =============================================================================
# PPE（个人防护装备）YOLO 检测配置
# =============================================================================

# YOLO 模型权重文件路径
PPE_MODEL_PATH = os.getenv(
    "PPE_MODEL_PATH",
    str(BASE_DIR / "models" / "ppe" / "ppe-detection-yolo" / "ppe.pt"),
)

# YOLO 检测置信度阈值，低于此值的检测结果会被过滤
PPE_CONF_THRESHOLD = float(os.getenv("PPE_CONF_THRESHOLD", "0.35"))

# YOLO 推理时的输入图像尺寸
PPE_IMAGE_SIZE = int(os.getenv("PPE_IMAGE_SIZE", "640"))

# YOLO 模型中“未戴安全帽”对应的类别名集合
PPE_NO_HELMET_CLASSES = {
    item.strip().lower()
    for item in os.getenv("PPE_NO_HELMET_CLASSES", "NO-Hardhat,nohat,NO-Helmet").split(",")
    if item.strip()
}

# YOLO 模型中“已戴安全帽”对应的类别名集合
PPE_WITH_HELMET_CLASSES = {
    item.strip().lower()
    for item in os.getenv("PPE_WITH_HELMET_CLASSES", "Hardhat,hat,Helmet").split(",")
    if item.strip()
}

# YOLO 模型中“未穿反光衣”对应的类别名集合
PPE_NO_VEST_CLASSES = {
    item.strip().lower()
    for item in os.getenv("PPE_NO_VEST_CLASSES", "NO-Safety Vest,novest,NO-Vest").split(",")
    if item.strip()
}

# YOLO 模型中“已穿反光衣”对应的类别名集合
PPE_WITH_VEST_CLASSES = {
    item.strip().lower()
    for item in os.getenv("PPE_WITH_VEST_CLASSES", "Safety Vest,vest").split(",")
    if item.strip()
}

# YOLO 模型中“人员”对应的类别名集合
PPE_PERSON_CLASSES = {
    item.strip().lower()
    for item in os.getenv("PPE_PERSON_CLASSES", "Person,person").split(",")
    if item.strip()
}

# =============================================================================
# RabbitMQ 消息队列配置
# =============================================================================

# RabbitMQ 连接地址
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@110.41.166.11:5672/%2F")

# 交换机名称
AI_EXCHANGE = os.getenv("AI_EXCHANGE", "buildguard.ai")

# 请求队列：Java 后端 → Python AI worker
AI_REQUEST_QUEUE = os.getenv("AI_REQUEST_QUEUE", "buildguard.ai.request")

# 结果队列：Python AI worker → Java 后端
AI_RESULT_QUEUE = os.getenv("AI_RESULT_QUEUE", "buildguard.ai.result")

# 结果路由键
AI_RESULT_ROUTING_KEY = os.getenv("AI_RESULT_ROUTING_KEY", "ai.result")

# 每次从队列预取的消息数量，1 表示每次只处理一条
AI_WORKER_PREFETCH = int(os.getenv("AI_WORKER_PREFETCH", "1"))

# 与 RabbitMQ 断开连接后的重连间隔（秒）
AI_WORKER_RECONNECT_SECONDS = float(os.getenv("AI_WORKER_RECONNECT_SECONDS", "5"))
