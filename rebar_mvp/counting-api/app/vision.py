from __future__ import annotations

import base64
import math
import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as tvf


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = ROOT / "model_full" / "model_final.pth"


def _resolve_device() -> torch.device:
    device = os.getenv("VISION_DEVICE", "auto").lower()
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_path() -> Path:
    raw = os.getenv("VISION_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    return Path(raw).expanduser().resolve()


@lru_cache(maxsize=1)
def get_model() -> tuple[torch.nn.Module, torch.device, str]:
    model_path = _model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")

    device = _resolve_device()
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.roi_heads.detections_per_img = int(os.getenv("VISION_MAX_DETECTIONS", "500"))

    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device, str(model_path)


def decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("invalid image")
    return img


def assess_quality(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    exposure = float(np.mean(gray))
    blur_score = min(1.0, blur_var / 320.0)
    exposure_score = 1.0 - min(1.0, abs(exposure - 130.0) / 130.0)
    score = round(float(0.65 * blur_score + 0.35 * exposure_score), 3)
    occlusion = "HIGH" if score < 0.55 else "MEDIUM" if score < 0.75 else "LOW"
    return {
        "quality_score": score,
        "occlusion_level": occlusion,
        "recognizable": score >= 0.6,
        "retake_advice": "请补拍：退后半步完整入镜，避免逆光，保证清晰对焦。" if score < 0.6 else "",
    }


def _boxes_to_tensor(boxes: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32)
    t_boxes = torch.tensor([[b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]] for b in boxes], dtype=torch.float32)
    t_scores = torch.tensor([b["score"] for b in boxes], dtype=torch.float32)
    return t_boxes, t_scores


def dedup_center_distance(boxes: list[dict], shape: tuple[int, int]) -> list[dict]:
    if not boxes:
        return []
    h, w = shape
    short_side = min(h, w)
    radii = [max(2.0, 0.25 * (b["w"] + b["h"])) for b in boxes]
    median_r = float(np.median(np.array(radii, dtype=np.float32))) if radii else 8.0
    min_dist = max(6.0, min(short_side * 0.04, median_r * 0.95))

    accepted: list[dict] = []
    for cand in sorted(boxes, key=lambda b: b.get("score", 0.0), reverse=True):
        cx = cand["x"] + 0.5 * cand["w"]
        cy = cand["y"] + 0.5 * cand["h"]
        too_close = False
        for existing in accepted:
            ex = existing["x"] + 0.5 * existing["w"]
            ey = existing["y"] + 0.5 * existing["h"]
            if math.hypot(cx - ex, cy - ey) < min_dist:
                too_close = True
                break
        if not too_close:
            accepted.append(cand)
    return accepted


def cluster_regions(shape: tuple[int, int], boxes: list[dict]) -> list[dict]:
    if not boxes:
        return []
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    radius = max(18, int(min(h, w) * 0.06))
    centers = []
    for b in boxes:
        cx = int(b["x"] + b["w"] / 2)
        cy = int(b["y"] + b["h"] / 2)
        centers.append((cx, cy))
        cv2.circle(mask, (cx, cy), radius, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        in_count = 0
        for cx, cy in centers:
            if x <= cx <= x + bw and y <= cy <= y + bh:
                in_count += 1
        if in_count < 6:
            continue
        regions.append({"x": int(x), "y": int(y), "w": int(bw), "h": int(bh), "count": in_count})

    regions.sort(key=lambda r: r["count"], reverse=True)
    return [{"x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"]} for r in regions]


def _center(box: dict) -> tuple[float, float]:
    return box["x"] + 0.5 * box["w"], box["y"] + 0.5 * box["h"]


def _in_region(cx: float, cy: float, region: dict, pad_ratio: float = 0.08) -> bool:
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    pad_x = w * pad_ratio
    pad_y = h * pad_ratio
    return (x - pad_x) <= cx <= (x + w + pad_x) and (y - pad_y) <= cy <= (y + h + pad_y)


def filter_primary_cluster_boxes(boxes: list[dict], shape: tuple[int, int]) -> list[dict]:
    if len(boxes) < 8:
        return boxes
    regions = cluster_regions(shape, boxes)
    if not regions:
        return boxes

    primary = regions[0]
    kept = []
    for b in boxes:
        cx, cy = _center(b)
        if _in_region(cx, cy, primary, pad_ratio=0.08):
            kept.append(b)

    return kept if len(kept) >= max(6, int(0.6 * len(boxes))) else boxes


def build_overlay(image: np.ndarray, head_boxes: list[dict], regions: list[dict]) -> str | None:
    canvas = image.copy()
    for box in head_boxes[:800]:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 165, 255), 2)

    for i, box in enumerate(regions):
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(canvas, f"B{i+1}", (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    ok, encoded = cv2.imencode(".jpg", canvas)
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def detect_rebar_heads(image: np.ndarray) -> dict:
    model, device, model_path = get_model()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = tvf.to_tensor(rgb).to(device)
    threshold = float(os.getenv("VISION_SCORE_THRESHOLD", "0.74"))

    with torch.no_grad():
        output = model([tensor])[0]

    boxes = output.get("boxes", torch.zeros((0, 4))).detach().cpu().numpy()
    scores = output.get("scores", torch.zeros((0,))).detach().cpu().numpy()

    raw_boxes = []
    for box, score in zip(boxes, scores):
        sc = float(score)
        if sc < threshold:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        raw_boxes.append({
            "x": max(0, x1),
            "y": max(0, y1),
            "w": max(1, x2 - x1),
            "h": max(1, y2 - y1),
            "score": sc,
        })

    t_boxes, t_scores = _boxes_to_tensor(raw_boxes)
    if len(raw_boxes) > 0:
        kept_idx = torchvision.ops.nms(t_boxes, t_scores, float(os.getenv("VISION_NMS_IOU", "0.35"))).tolist()
        nms_boxes = [raw_boxes[i] for i in kept_idx]
    else:
        nms_boxes = []

    dedup_boxes = dedup_center_distance(nms_boxes, image.shape[:2])
    cluster_boxes = filter_primary_cluster_boxes(dedup_boxes, image.shape[:2])

    regions = cluster_regions(image.shape[:2], cluster_boxes)
    if not regions and cluster_boxes:
        xs = [b["x"] for b in cluster_boxes]
        ys = [b["y"] for b in cluster_boxes]
        x2s = [b["x"] + b["w"] for b in cluster_boxes]
        y2s = [b["y"] + b["h"] for b in cluster_boxes]
        regions = [{"x": min(xs), "y": min(ys), "w": max(x2s) - min(xs), "h": max(y2s) - min(ys)}]

    return {
        "head_count": len(cluster_boxes),
        "bundle_count": len(regions),
        "regions": regions,
        "overlay_image_b64": build_overlay(image, cluster_boxes, regions),
        "method": (
            "torch_frcnn_nms_clustered"
            f"(raw={len(raw_boxes)},nms={len(nms_boxes)},dedup={len(dedup_boxes)},cluster={len(cluster_boxes)})"
        ),
        "model_path": model_path,
    }


def normalize_spec(raw_spec: str | None) -> dict:
    if not raw_spec:
        return {"normalized_spec_code": None, "steel_grade": None, "diameter_mm": None, "unit_weight_kg_m": None}
    s = raw_spec.strip().upper().replace(" ", "")
    grade = "HRB400E" if "400" in s else "HRB500E" if "500" in s else None
    digits = "".join(ch for ch in s if ch.isdigit())
    diameter = int(digits[-2:]) if len(digits) >= 2 else None
    unit = None
    if diameter:
        unit = round(0.00617 * diameter * diameter, 4)
    code = f"{grade}-{diameter}" if grade and diameter else None
    return {"normalized_spec_code": code, "steel_grade": grade, "diameter_mm": diameter, "unit_weight_kg_m": unit}
