import argparse
import csv
import random
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as tvf

from app.vision import cluster_regions, dedup_center_distance


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0


def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def read_gt(xml_path: Path, cls_name: str) -> list[Box]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    out: list[Box] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        if name != cls_name:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        out.append(
            Box(
                x1=float(bb.findtext("xmin", 0)),
                y1=float(bb.findtext("ymin", 0)),
                x2=float(bb.findtext("xmax", 0)),
                y2=float(bb.findtext("ymax", 0)),
            )
        )
    return out


def load_model(weight_path: Path, device: torch.device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.roi_heads.detections_per_img = 500
    state = torch.load(str(weight_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def image_ids(root: Path) -> list[str]:
    ids = [p.stem for p in (root / "JPEGImages").glob("*.jpg")]
    ids += [p.stem for p in (root / "JPEGImages").glob("*.png")]
    return sorted(set(ids))


def image_path(root: Path, image_id: str) -> Path:
    p = root / "JPEGImages" / f"{image_id}.jpg"
    if p.exists():
        return p
    return root / "JPEGImages" / f"{image_id}.png"


def _center(box: dict) -> tuple[float, float]:
    return box["x"] + 0.5 * box["w"], box["y"] + 0.5 * box["h"]


def _in_region(cx: float, cy: float, region: dict, pad_ratio: float) -> bool:
    x, y, w, h = region["x"], region["y"], region["w"], region["h"]
    px = w * pad_ratio
    py = h * pad_ratio
    return (x - px) <= cx <= (x + w + px) and (y - py) <= cy <= (y + h + py)


def filter_primary_cluster_boxes_custom(boxes: list[dict], shape: tuple[int, int], pad_ratio: float) -> list[dict]:
    if len(boxes) < 8:
        return boxes
    regions = cluster_regions(shape, boxes)
    if not regions:
        return boxes
    primary = regions[0]
    kept = []
    for b in boxes:
        cx, cy = _center(b)
        if _in_region(cx, cy, primary, pad_ratio=pad_ratio):
            kept.append(b)
    return kept if len(kept) >= max(6, int(0.6 * len(boxes))) else boxes


def _tile_starts(total: int, size: int, overlap: float) -> list[int]:
    if total <= size:
        return [0]
    step = max(1, int(size * (1.0 - overlap)))
    starts = list(range(0, total - size + 1, step))
    last = total - size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _iter_tiles(width: int, height: int, tile_size: int, overlap: float):
    xs = _tile_starts(width, tile_size, overlap)
    ys = _tile_starts(height, tile_size, overlap)
    for y in ys:
        for x in xs:
            yield x, y, min(tile_size, width - x), min(tile_size, height - y)


def _infer_raw_boxes(model, device, rgb_img, score_thr: float, offset_x: int = 0, offset_y: int = 0):
    t = tvf.to_tensor(rgb_img).to(device)
    with torch.no_grad():
        out = model([t])[0]

    boxes = out.get("boxes", torch.zeros((0, 4))).detach().cpu().numpy()
    scores = out.get("scores", torch.zeros((0,))).detach().cpu().numpy()

    raw = []
    for b, s in zip(boxes, scores):
        sc = float(s)
        if sc < score_thr:
            continue
        x1, y1, x2, y2 = [float(v) for v in b.tolist()]
        raw.append(
            {
                "x": max(0.0, x1 + offset_x),
                "y": max(0.0, y1 + offset_y),
                "w": max(1.0, x2 - x1),
                "h": max(1.0, y2 - y1),
                "score": sc,
            }
        )
    return raw


def predict_current_pipeline(
    model,
    device,
    img: Image.Image,
    score_thr: float,
    nms_iou: float,
    cluster_pad: float,
    enable_tile: bool,
    tile_size: int,
    tile_overlap: float,
    tile_score_delta: float,
):
    rgb = np.array(img)

    full_raw = _infer_raw_boxes(model, device, rgb, score_thr, 0, 0)
    raw = list(full_raw)

    tile_raw_count = 0
    h, w = rgb.shape[:2]
    if enable_tile and max(h, w) > tile_size:
        tile_thr = max(0.01, min(0.99, score_thr + tile_score_delta))
        for x, y, tw, th in _iter_tiles(w, h, tile_size, tile_overlap):
            crop = rgb[y:y + th, x:x + tw]
            tile_raw = _infer_raw_boxes(model, device, crop, tile_thr, x, y)
            raw.extend(tile_raw)
            tile_raw_count += len(tile_raw)

    if raw:
        t_boxes = torch.tensor([[r["x"], r["y"], r["x"] + r["w"], r["y"] + r["h"]] for r in raw], dtype=torch.float32)
        t_scores = torch.tensor([r["score"] for r in raw], dtype=torch.float32)
        keep = torchvision.ops.nms(t_boxes, t_scores, nms_iou).tolist()
        nms = [raw[i] for i in keep]
    else:
        nms = []

    dedup = dedup_center_distance(nms, (h, w))
    cluster = filter_primary_cluster_boxes_custom(dedup, (h, w), pad_ratio=cluster_pad)

    pred = [Box(x1=b["x"], y1=b["y"], x2=b["x"] + b["w"], y2=b["y"] + b["h"], score=b["score"]) for b in cluster]
    return pred, {
        "full_raw": len(full_raw),
        "tile_raw": tile_raw_count,
        "raw": len(raw),
        "nms": len(nms),
        "dedup": len(dedup),
        "cluster": len(cluster),
    }


def match_counts(pred: list[Box], gt: list[Box], iou_thr: float):
    if not pred:
        return 0, 0, len(gt)
    used = [False] * len(gt)
    tp = 0
    fp = 0

    pred_sorted = sorted(pred, key=lambda b: b.score, reverse=True)
    for p in pred_sorted:
        best_i = -1
        best_iou = 0.0
        for i, g in enumerate(gt):
            if used[i]:
                continue
            v = iou(p, g)
            if v > best_iou:
                best_iou = v
                best_i = i
        if best_i >= 0 and best_iou >= iou_thr:
            used[best_i] = True
            tp += 1
        else:
            fp += 1
    fn = len(gt) - tp
    return tp, fp, fn


def evaluate(
    model,
    device,
    root: Path,
    ids: list[str],
    target_class: str,
    score_thr: float,
    nms_iou: float,
    iou_thr: float,
    cluster_pad: float,
    enable_tile: bool,
    tile_size: int,
    tile_overlap: float,
    tile_score_delta: float,
):
    tp = fp = fn = 0
    abs_err = []
    sq_err = []
    ape = []
    stage_full_raw = []
    stage_tile_raw = []
    stage_raw = []
    stage_nms = []
    stage_dedup = []
    stage_cluster = []
    per_image_rows = []

    for image_id in ids:
        img = Image.open(image_path(root, image_id)).convert("RGB")
        gt = read_gt(root / "Annotations" / f"{image_id}.xml", target_class)
        pred, stage = predict_current_pipeline(
            model,
            device,
            img,
            score_thr,
            nms_iou,
            cluster_pad,
            enable_tile,
            tile_size,
            tile_overlap,
            tile_score_delta,
        )

        _tp, _fp, _fn = match_counts(pred, gt, iou_thr)
        tp += _tp
        fp += _fp
        fn += _fn

        pc = len(pred)
        gc = len(gt)
        err = pc - gc
        abs_err.append(abs(err))
        sq_err.append(err * err)
        if gc > 0:
            ape.append(abs(err) / gc)

        stage_full_raw.append(stage["full_raw"])
        stage_tile_raw.append(stage["tile_raw"])
        stage_raw.append(stage["raw"])
        stage_nms.append(stage["nms"])
        stage_dedup.append(stage["dedup"])
        stage_cluster.append(stage["cluster"])

        per_image_rows.append(
            {
                "image_id": image_id,
                "gt_count": gc,
                "pred_count": pc,
                "err": err,
                "abs_err": abs(err),
                "tp": _tp,
                "fp": _fp,
                "fn": _fn,
                "full_raw": stage["full_raw"],
                "tile_raw": stage["tile_raw"],
                "raw": stage["raw"],
                "nms": stage["nms"],
                "dedup": stage["dedup"],
                "cluster": stage["cluster"],
            }
        )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    acc_like = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

    mae = sum(abs_err) / len(abs_err)
    rmse = (sum(sq_err) / len(sq_err)) ** 0.5
    mape = (sum(ape) / len(ape)) if ape else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy_like": acc_like,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "full_raw_avg": statistics.mean(stage_full_raw),
        "tile_raw_avg": statistics.mean(stage_tile_raw),
        "raw_avg": statistics.mean(stage_raw),
        "nms_avg": statistics.mean(stage_nms),
        "dedup_avg": statistics.mean(stage_dedup),
        "cluster_avg": statistics.mean(stage_cluster),
        "per_image": per_image_rows,
    }


def write_per_image_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image_id",
        "gt_count",
        "pred_count",
        "err",
        "abs_err",
        "tp",
        "fp",
        "fn",
        "full_raw",
        "tile_raw",
        "raw",
        "nms",
        "dedup",
        "cluster",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--target-class", default="rebar")
    ap.add_argument("--score-thr", type=float, default=0.74)
    ap.add_argument("--nms-iou", type=float, default=0.35)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--cluster-pad", type=float, default=0.08)
    ap.add_argument("--enable-tile", action="store_true")
    ap.add_argument("--tile-size", type=int, default=1280)
    ap.add_argument("--tile-overlap", type=float, default=0.25)
    ap.add_argument("--tile-score-delta", type=float, default=-0.05)
    ap.add_argument("--split", choices=["all", "holdout20"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--per-image-csv", default="")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    weight = Path(args.weight)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ids = image_ids(root)
    if args.split == "holdout20":
        rng = random.Random(args.seed)
        rng.shuffle(ids)
        ids = ids[: max(1, int(len(ids) * 0.2))]

    model = load_model(weight, device)
    out = evaluate(
        model=model,
        device=device,
        root=root,
        ids=ids,
        target_class=args.target_class,
        score_thr=args.score_thr,
        nms_iou=args.nms_iou,
        iou_thr=args.iou_thr,
        cluster_pad=args.cluster_pad,
        enable_tile=args.enable_tile,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        tile_score_delta=args.tile_score_delta,
    )

    print("==== Evaluation (current pipeline) ====")
    print(f"images={len(ids)} split={args.split} class={args.target_class}")
    print(f"weight={weight}")
    print(
        f"score_thr={args.score_thr} nms_iou={args.nms_iou} iou_thr={args.iou_thr} "
        f"cluster_pad={args.cluster_pad} enable_tile={args.enable_tile} "
        f"tile_size={args.tile_size} tile_overlap={args.tile_overlap} tile_score_delta={args.tile_score_delta}"
    )
    print("---- detection metrics ----")
    print(f"TP={out['tp']} FP={out['fp']} FN={out['fn']}")
    print(f"precision={out['precision']:.4f}")
    print(f"recall={out['recall']:.4f}")
    print(f"f1={out['f1']:.4f}")
    print(f"accuracy_like={out['accuracy_like']:.4f}")
    print("---- counting metrics ----")
    print(f"mae={out['mae']:.4f}")
    print(f"rmse={out['rmse']:.4f}")
    print(f"mape={out['mape']:.4f}")
    print("---- stage counts (per-image avg) ----")
    print(f"full_raw_avg={out['full_raw_avg']:.2f}")
    print(f"tile_raw_avg={out['tile_raw_avg']:.2f}")
    print(f"raw_avg={out['raw_avg']:.2f}")
    print(f"nms_avg={out['nms_avg']:.2f}")
    print(f"dedup_avg={out['dedup_avg']:.2f}")
    print(f"cluster_avg={out['cluster_avg']:.2f}")

    if args.per_image_csv:
        csv_path = Path(args.per_image_csv)
        write_per_image_csv(csv_path, out["per_image"])
        print(f"per_image_csv={csv_path}")


if __name__ == "__main__":
    main()
