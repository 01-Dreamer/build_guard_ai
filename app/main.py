from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from app.config import FACE_DATA_DIR
from app.face_service import FaceRecognitionService, FaceServiceError, decode_image
from app.face_store import FaissFaceStore
from app.helmet_service import HelmetDetectionService, HelmetServiceError
from app.safety_service import NoHelmetSafetyService
from app.web import TEST_PAGE_HTML


store = FaissFaceStore(FACE_DATA_DIR)
service = FaceRecognitionService(store)
helmet_service = HelmetDetectionService()
safety_service = NoHelmetSafetyService(helmet_service=helmet_service, face_service=service)

app = FastAPI(title="BuildGuard Face Recognition API")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    return HTMLResponse(
        content=TEST_PAGE_HTML,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/health", include_in_schema=False)
async def health() -> dict[str, int | str]:
    return {"status": "ok", "registered_faces": store.count}


@app.post("/faces/register")
async def register_face(
    person_id: Annotated[str, Form(alias="id")],
    img: Annotated[UploadFile, File()],
    name: Annotated[str | None, Form()] = None,
) -> dict:
    try:
        image = decode_image(await img.read())
        return service.register(person_id=person_id, name=name, image=image)
    except FaceServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/faces/{person_id}")
async def delete_face(person_id: str) -> dict:
    return service.delete(person_id)


@app.post("/safety/detect")
async def detect_ppe_people(img: Annotated[UploadFile, File()]) -> dict:
    try:
        image = decode_image(await img.read())
        return safety_service.detect(image)
    except (FaceServiceError, HelmetServiceError) as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
