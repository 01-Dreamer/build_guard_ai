# BuildGuard AI Service

FastAPI service for registering worker faces and detecting PPE status in site images.

## Run

Use the existing conda environment:

```bash
conda activate BuildGuard
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

The first request that uses InsightFace may download the configured model pack into `~/.insightface`.
The PPE detector uses the `Vinayakmane47/PPE_detection_YOLO` weight at:

```text
models/helmet/ppe-detection-yolo/ppe.pt
```

Download source:

```text
https://github.com/Vinayakmane47/PPE_detection_YOLO
```

The model classes include `Hardhat`, `NO-Hardhat`, `Safety Vest`, `NO-Safety Vest`, `Person`, `Mask`, `NO-Mask`, `Safety Cone`, `machinery`, and `vehicle`.

## APIs

Open the browser test page:

```text
http://127.0.0.1:8080/
```

1. Register or overwrite a face:

```bash
curl -X POST http://127.0.0.1:8080/faces/register \
  -F "id=worker001" \
  -F "name=张三" \
  -F "img=@/path/to/face.jpg"
```

`name` is optional. If omitted, it defaults to the same value as `id`.

2. Delete a registered face:

```bash
curl -X DELETE http://127.0.0.1:8080/faces/worker001
```

3. Detect people, safety helmets, safety vests, and identify faces when possible:

```bash
curl -X POST http://127.0.0.1:8080/safety/detect \
  -F "img=@/path/to/site.jpg"
```

Health check:

```bash
curl http://127.0.0.1:8080/health
```

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
- `HELMET_PPE_MODEL_PATH`: PPE model path, default `models/helmet/ppe-detection-yolo/ppe.pt`
- `HELMET_CONF_THRESHOLD`: helmet model confidence threshold, default `0.35`
