import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as tvf


def read_gt_count(xml_path: Path, target_class: str = "rebar") -> int:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    c = 0
    for obj in root.findall("object"):
        name = (obj.find("name").text or "").strip()
        if name == target_class:
            c += 1
    return c


def load_model(weight_path: Path, device: torch.device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(str(weight_path), map_location=device))
    model.to(device)
    model.eval()
    return model


def build_val_ids(dataset_root: Path, val_ratio: float = 0.2, seed: int = 42):
    ids = sorted([p.stem for p in (dataset_root / "JPEGImages").glob("*.jpg")])
    ids += sorted([p.stem for p in (dataset_root / "JPEGImages").glob("*.png")])
    ids = sorted(set(ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    val_n = max(1, int(len(ids) * val_ratio))
    return ids[:val_n]


def image_path(dataset_root: Path, image_id: str) -> Path:
    p = dataset_root / "JPEGImages" / f"{image_id}.jpg"
    if p.exists():
        return p
    return dataset_root / "JPEGImages" / f"{image_id}.png"


def evaluate_threshold(model, device, dataset_root: Path, val_ids: list[str], threshold: float):
    abs_err = 0
    sq_err = 0
    n = 0
    for image_id in val_ids:
        img = Image.open(image_path(dataset_root, image_id)).convert("RGB")
        tensor = tvf.to_tensor(img).to(device)
        with torch.no_grad():
            out = model([tensor])[0]
        scores = out["scores"].detach().cpu()
        pred_count = int((scores >= threshold).sum().item())

        gt_count = read_gt_count(dataset_root / "Annotations" / f"{image_id}.xml")
        e = abs(pred_count - gt_count)
        abs_err += e
        sq_err += e * e
        n += 1

    mae = abs_err / max(1, n)
    rmse = (sq_err / max(1, n)) ** 0.5
    return mae, rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    weight = Path(args.weight)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(weight, device)
    val_ids = build_val_ids(dataset_root, val_ratio=0.2, seed=42)

    best = None
    thresholds = [round(x / 100, 2) for x in range(20, 91, 2)]
    for th in thresholds:
        mae, rmse = evaluate_threshold(model, device, dataset_root, val_ids, th)
        print(f"th={th:.2f} mae={mae:.3f} rmse={rmse:.3f}")
        if best is None or mae < best[1] or (mae == best[1] and rmse < best[2]):
            best = (th, mae, rmse)

    print(f"BEST threshold={best[0]:.2f} mae={best[1]:.3f} rmse={best[2]:.3f}")


if __name__ == "__main__":
    main()
