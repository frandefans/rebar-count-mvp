import argparse
import os
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import src.transforms as T
from src.engine import evaluate, train_one_epoch
from src.utils import utils


def get_transform(train: bool):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def read_xml(xml_path: Path, target_class: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    labels = []
    for obj in root.findall("object"):
        name = obj.find("name")
        if name is None:
            continue
        label = (name.text or "").strip()
        if label != target_class:
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(1)

    return boxes, labels


class RebarVOCDataset(Dataset):
    def __init__(self, root: Path, image_ids: list[str], target_class: str, transforms=None):
        self.root = root
        self.image_ids = image_ids
        self.target_class = target_class
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = self.root / "JPEGImages" / f"{image_id}.jpg"
        if not img_path.exists():
            img_path = self.root / "JPEGImages" / f"{image_id}.png"

        xml_path = self.root / "Annotations" / f"{image_id}.xml"
        img = Image.open(img_path).convert("RGB")
        boxes, labels = read_xml(xml_path, self.target_class)

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


def build_model(num_classes: int = 2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.detections_per_img = 500
    return model


@dataclass
class SplitResult:
    train_ids: list[str]
    val_ids: list[str]


def build_split(dataset_root: Path, val_ratio: float, seed: int) -> SplitResult:
    image_dir = dataset_root / "JPEGImages"
    ids = sorted([p.stem for p in image_dir.glob("*.jpg")] + [p.stem for p in image_dir.glob("*.png")])
    ids = sorted(set(ids))
    if not ids:
        raise RuntimeError(f"No images found in {image_dir}")

    rng = random.Random(seed)
    rng.shuffle(ids)
    val_count = max(1, int(len(ids) * val_ratio))
    val_ids = ids[:val_count]
    train_ids = ids[val_count:]
    if not train_ids:
        raise RuntimeError("Train split is empty. Reduce val_ratio.")
    return SplitResult(train_ids=train_ids, val_ids=val_ids)


def run(args):
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise RuntimeError(f"Dataset root not found: {dataset_root}")

    split = build_split(dataset_root, val_ratio=args.val_ratio, seed=args.seed)
    train_ds = RebarVOCDataset(dataset_root, split.train_ids, target_class=args.target_class, transforms=get_transform(True))
    val_ds = RebarVOCDataset(dataset_root, split.val_ids, target_class=args.target_class, transforms=get_transform(False))

    if args.max_train_samples > 0:
        train_ds = Subset(train_ds, list(range(min(args.max_train_samples, len(train_ds)))))
    if args.max_val_samples > 0:
        val_ds = Subset(val_ds, list(range(min(args.max_val_samples, len(val_ds)))))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(num_classes=2).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] device={device} train={len(train_ds)} val={len(val_ds)} target_class={args.target_class}")
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=20)
        scheduler.step()
        evaluate(model, val_loader, device=device)

        ckpt_path = out_dir / f"model_{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[INFO] saved checkpoint: {ckpt_path}")

    final_path = out_dir / "model_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"[INFO] saved final model: {final_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train rebar head detector on VOC-style dataset.")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--target-class", type=str, default="rebar")
    parser.add_argument("--output-dir", type=str, default="./model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
