# BuildGuard AI Service

RabbitMQ AI worker for BuildGuard. This project does not expose an HTTP service; Java sends AI tasks through MQ and consumes the result through MQ.

The production integration path is MQ:

```text
Java server -> buildguard.ai.request -> Python worker -> buildguard.ai.result -> Java server
```

The worker supports `camera_yolo`, `yolo_detection`, `ppe_detection`, `face_recognition`, and `tower_prediction`. If a local YOLO or tower model is not available, it returns stable mock/rule results with the same response shape so the Java side can test the full async flow.

## Run

Use the existing conda environment:

```bash
conda activate BuildGuard
pip install -r requirements.txt
python -m app.worker
```

Run a worker-only smoke test without RabbitMQ:

```bash
conda activate BuildGuard
python -m app.worker --dry-run
```

The first task that uses InsightFace may download the configured model pack into `~/.insightface`.
The PPE detector uses the `Vinayakmane47/PPE_detection_YOLO` weight at:

```text
models/ppe/ppe-detection-yolo/ppe.pt
```

Download source:

```text
https://github.com/Vinayakmane47/PPE_detection_YOLO
```

The model classes include `Hardhat`, `NO-Hardhat`, `Safety Vest`, `NO-Safety Vest`, `Person`, `Mask`, `NO-Mask`, `Safety Cone`, `machinery`, and `vehicle`.

## RabbitMQ messages

The worker consumes JSON strings from `buildguard.ai.request` and publishes JSON strings to `buildguard.ai.result`.

Example request:

```json
{
  "messageId": "msg-001",
  "eventType": "ai.request",
  "taskId": "task-001",
  "taskType": "tower_prediction",
  "deviceCode": "TC-001",
  "deviceType": "tower_crane",
  "occurredAt": "2026-07-17T10:00:00+08:00",
  "sentAt": "2026-07-17T10:00:01+08:00",
  "payload": {
    "ratedLoad": 10,
    "ratedMoment": 80,
    "telemetry": [
      {
        "weight": 8.4,
        "amplitude": 9.8,
        "moment": 82.3,
        "windSpeed": 13.2,
        "obliquity": 2.1,
        "height": 31.5,
        "rotation": 120
      }
    ]
  }
}
```

Result fields include `taskId`, `resultStatus`, `detections`, `prediction`, `model`, and `errorMessage`.

## Persistence

FAISS and identity metadata are persisted under `data/faces` by default:

- `data/faces/faces.faiss`
- `data/faces/faces_meta.json`

Only embeddings and metadata are stored. Uploaded source images are not saved.

Useful environment variables:

- `FACE_DATA_DIR`: persistence directory, default `data/faces`
- `FACE_MATCH_THRESHOLD`: cosine similarity threshold, default `0.55`
- `INSIGHTFACE_MODEL`: model pack name, default `buffalo_l`
- `INSIGHTFACE_ROOT`: model cache directory, default `~/.insightface`
- `INSIGHTFACE_CTX_ID`: `-1` for CPU, `0` for GPU, default `-1`
- `INSIGHTFACE_DET_SIZE`: detection size, default `640,640`
- `INSIGHTFACE_SMALL_DET_SIZE`: detection size for images up to 320px, default `320,320`
- `INSIGHTFACE_PROVIDERS`: ONNX Runtime providers, default `CPUExecutionProvider`
- `PPE_MODEL_PATH`: PPE model path, default `models/ppe/ppe-detection-yolo/ppe.pt`
- `PPE_CONF_THRESHOLD`: PPE model confidence threshold, default `0.35`
- `PPE_IMAGE_SIZE`: YOLO inference image size, default `640`
- `RABBITMQ_URL`: RabbitMQ connection URL, default guest connection to `110.41.166.11:5672`
- `AI_REQUEST_QUEUE`: request queue, default `buildguard.ai.request`
- `AI_RESULT_QUEUE`: result queue, default `buildguard.ai.result`
- `AI_WORKER_PREFETCH`: worker prefetch count, default `1`
- `AI_WORKER_RECONNECT_SECONDS`: reconnect delay, default `5`
