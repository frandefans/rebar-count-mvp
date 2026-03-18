import argparse
import random
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import src.transforms as T
from src.engine import evaluate, train_one_epoch
from src.train_local import RebarVOCDataset, build_split
from src.utils import utils


class ResizeMaxSide:
    def __init__(self, max_side: int = 1600):
        self.max_side = max_side

    def __call__(self, image, target):
        w, h = image.size
        long_side = max(w, h)
        if long_side <= self.max_side:
            return image, target

        scale = self.max_side / float(long_side)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        image = image.resize((nw, nh), Image.BILINEAR)

        boxes = target.get("boxes")
        if boxes is not None and len(boxes) > 0:
            boxes = boxes.clone()
            boxes[:, [0, 2]] *= scale
            boxes[:, [1, 3]] *= scale
            target["boxes"] = boxes
            target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        return image, target


def get_transform(train: bool, resize_max_side: int):
    t = [ResizeMaxSide(max_side=resize_max_side), T.ToTensor()]
    if train:
        t.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(t)


def build_model(num_classes: int = 2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.detections_per_img = 500
    return model


def read_hardcase_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {x.strip() for x in path.read_text(encoding='utf-8').splitlines() if x.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--init-weight', default='')
    ap.add_argument('--hardcase-list', default='')
    ap.add_argument('--hardcase-multiplier', type=int, default=3)
    ap.add_argument('--target-class', default='rebar')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--val-ratio', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lr', type=float, default=0.003)
    ap.add_argument('--resize-max-side', type=int, default=1600)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    split = build_split(dataset_root, val_ratio=args.val_ratio, seed=args.seed)

    hardcase_ids = read_hardcase_ids(Path(args.hardcase_list)) if args.hardcase_list else set()
    train_ids = list(split.train_ids)

    # oversample hard cases in training split
    extra = []
    if hardcase_ids and args.hardcase_multiplier > 1:
        hard_train = [x for x in train_ids if x in hardcase_ids]
        for _ in range(args.hardcase_multiplier - 1):
            extra.extend(hard_train)
        train_ids.extend(extra)

    random.Random(args.seed).shuffle(train_ids)

    train_ds = RebarVOCDataset(
        dataset_root,
        train_ids,
        target_class=args.target_class,
        transforms=get_transform(True, resize_max_side=args.resize_max_side),
    )
    val_ds = RebarVOCDataset(
        dataset_root,
        split.val_ids,
        target_class=args.target_class,
        transforms=get_transform(False, resize_max_side=args.resize_max_side),
    )

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

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = build_model(num_classes=2)

    if args.init_weight:
        ckpt = Path(args.init_weight)
        if ckpt.exists():
            model.load_state_dict(torch.load(str(ckpt), map_location='cpu'))
            print(f'[INFO] loaded init weight: {ckpt}')

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[INFO] device={device} train={len(train_ds)} val={len(val_ds)} hardcase_in_train={len([x for x in split.train_ids if x in hardcase_ids])} extra={len(extra)}')

    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=20)
        scheduler.step()
        evaluate(model, val_loader, device=device)
        ckpt = out_dir / f'model_{epoch}.pth'
        torch.save(model.state_dict(), ckpt)
        print(f'[INFO] saved {ckpt}')

    final_path = out_dir / 'model_final.pth'
    torch.save(model.state_dict(), final_path)
    print(f'[INFO] saved final model: {final_path}')


if __name__ == '__main__':
    main()
