import argparse
import csv
import itertools
import re
import subprocess
from pathlib import Path


def parse_metrics(text: str) -> dict:
    keys = ["precision", "recall", "f1", "accuracy_like", "mae", "rmse", "mape"]
    out = {}
    for k in keys:
        m = re.search(rf"{k}=([0-9.]+)", text)
        out[k] = float(m.group(1)) if m else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="python")
    ap.add_argument("--eval-script", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--target-class", default="rebar")
    ap.add_argument("--split", default="holdout20")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--score-list", default="0.62,0.66,0.70,0.74")
    ap.add_argument("--nms-list", default="0.30,0.35,0.40,0.45")
    ap.add_argument("--pad-list", default="0.06,0.08,0.10,0.12")
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    scores = [float(x) for x in args.score_list.split(",")]
    nmses = [float(x) for x in args.nms_list.split(",")]
    pads = [float(x) for x in args.pad_list.split(",")]

    rows = []
    combos = list(itertools.product(scores, nmses, pads))

    for score_thr, nms_iou, pad in combos:
        cmd = [
            args.python,
            args.eval_script,
            "--dataset-root",
            args.dataset_root,
            "--weight",
            args.weight,
            "--target-class",
            args.target_class,
            "--score-thr",
            str(score_thr),
            "--nms-iou",
            str(nms_iou),
            "--cluster-pad",
            str(pad),
            "--iou-thr",
            str(args.iou_thr),
            "--split",
            args.split,
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        txt = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m = parse_metrics(txt)
        rows.append(
            {
                "score_thr": score_thr,
                "nms_iou": nms_iou,
                "cluster_pad": pad,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "accuracy_like": m["accuracy_like"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "mape": m["mape"],
                "ok": proc.returncode == 0,
            }
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "score_thr",
        "nms_iou",
        "cluster_pad",
        "precision",
        "recall",
        "f1",
        "accuracy_like",
        "mae",
        "rmse",
        "mape",
        "ok",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    good = [r for r in rows if r["ok"] and r["precision"] is not None]
    good.sort(key=lambda x: (-x["recall"], -x["precision"], x["mae"]))

    print(f"grid_done={len(rows)}")
    print(f"out_csv={out_path}")
    for i, r in enumerate(good[:5], 1):
        print(
            f"TOP{i} score={r['score_thr']:.2f} nms={r['nms_iou']:.2f} pad={r['cluster_pad']:.2f} "
            f"P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} MAE={r['mae']:.4f}"
        )


if __name__ == "__main__":
    main()
