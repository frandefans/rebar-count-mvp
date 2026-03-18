from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .vision import assess_quality, decode_image, detect_rebar_heads, normalize_spec

ROOT = Path(__file__).resolve().parents[3]
WEB_DIR = ROOT / "rebar_mvp" / "collector-web"
STORAGE = ROOT / "rebar_mvp" / "counting-api" / "storage"
STORAGE.mkdir(parents=True, exist_ok=True)


class TaskMetadata(BaseModel):
    project_name: str
    batch: str
    yard: str
    photographer: str
    captured_at: datetime


app = FastAPI(title="rebar-counting-mvp")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TASKS: dict[str, dict] = {}


@app.get("/")
def root() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/v1/tasks")
async def create_task(metadata: str = Form(...), image: UploadFile = File(...), tag_image: UploadFile | None = File(default=None)):
    try:
        meta_obj = TaskMetadata.model_validate_json(metadata)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"metadata invalid: {exc}")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="image is empty")

    tag_bytes = await tag_image.read() if tag_image else None

    task_id = str(uuid.uuid4())
    task_dir = STORAGE / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    img_path = task_dir / f"image_{image.filename or 'upload.jpg'}"
    img_path.write_bytes(image_bytes)
    if tag_bytes:
        (task_dir / f"tag_{tag_image.filename or 'tag.jpg'}").write_bytes(tag_bytes)

    TASKS[task_id] = {"task_id": task_id, "status": "PROCESSING", "error_message": None, "result": None}

    try:
        image_np = decode_image(image_bytes)
        quality = assess_quality(image_np)
        det = detect_rebar_heads(image_np)

        ocr_fields = {
            "spec_text": None,
            "manufacturer": None,
            "batch": None,
            "weight_kg": None,
            "quantity": None,
        }

        norm = normalize_spec(ocr_fields.get("spec_text"))
        has_tag = bool(tag_bytes)
        score = float(quality["quality_score"])
        if not has_tag:
            score = min(score, 0.82)

        anomaly_codes = []
        if not has_tag:
            anomaly_codes.append("NO_TAG_RISK")
        if not all([ocr_fields["spec_text"], ocr_fields["batch"], ocr_fields["quantity"]]):
            anomaly_codes.append("REVIEW_HINT")

        retake = quality["retake_advice"]
        if score < 0.6 and not retake:
            retake = "请重新拍摄，优先保证清晰度和吊牌完整性。"

        estimated_weight = 0.0
        if norm.get("unit_weight_kg_m"):
            estimated_weight = round(det["head_count"] * norm["unit_weight_kg_m"] * 12.0, 3)

        result = {
            "bundle_count": int(det["bundle_count"]),
            "rebar_head_count": int(det["head_count"]),
            "estimated_weight_kg": float(estimated_weight),
            "ocr_fields": ocr_fields,
            "normalized_spec_code": norm.get("normalized_spec_code"),
            "confidence_score": score,
            "anomaly_codes": anomaly_codes,
            "retake_advice": retake,
            "detection_method": det["method"],
            "detected_head_count": int(det["head_count"]),
            "detection_regions": det["regions"],
            "overlay_image_b64": det["overlay_image_b64"],
            "model_path": det["model_path"],
        }
        TASKS[task_id]["status"] = "DONE"
        TASKS[task_id]["result"] = result
        (task_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        TASKS[task_id]["status"] = "FAILED"
        TASKS[task_id]["error_message"] = str(exc)

    return {"task_id": task_id, "status": TASKS[task_id]["status"]}


@app.get("/api/v1/tasks/{task_id}")
def get_task(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return {
        "task_id": task_id,
        "status": task["status"],
        "error_message": task["error_message"],
        "result_preview": None if task["status"] != "DONE" else {
            "bundle_count": task["result"].get("bundle_count"),
            "confidence_score": task["result"].get("confidence_score"),
        },
    }


@app.get("/api/v1/tasks/{task_id}/result")
def get_result(task_id: str):
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    if task["status"] != "DONE":
        raise HTTPException(status_code=409, detail=f"task status is {task['status']}")
    return task["result"]


class SpecReq(BaseModel):
    raw_spec: str


@app.post("/api/v1/specs/normalize")
def normalize_spec_api(req: SpecReq):
    x = normalize_spec(req.raw_spec)
    return {"raw_spec": req.raw_spec, **x}


app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


