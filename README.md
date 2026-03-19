# Rebar Count MVP (Single Image)

This repository contains two parts:

1. `rebar_mvp`: runnable H5 + FastAPI service for single-image rebar head counting.
2. `src`: model training and evaluation pipeline (including recall-boost retraining).

The current default runtime model is:

- `model_recall_boost/model_final.pth` (Git LFS tracked)

---

## Project Layout

```text
rebar_count/
  rebar_mvp/
    collector-web/
      index.html
    counting-api/
      app/
        main.py
        vision.py
      scripts/
        eval_current_pipeline.py
        grid_search_current_pipeline.py
      requirements-mvp.txt
      .env.example
  src/
    train_local.py
    tune_threshold.py
    build_hardcase_list.py
    train_recall_boost.py
    engine.py
    coco/
      coco_eval.py
      coco_utils.py
  model_full/model_final.pth
  model_recall_boost/model_final.pth
  artifacts/hardcase_ids.txt
```

---

## 1) Run MVP Service (H5 + API)

### Install dependencies

```powershell
cd D:\0\PythonDemo\rebar_count\rebar_mvp\counting-api
python -m pip install -r requirements-mvp.txt
```

### Start service

```powershell
cd D:\0\PythonDemo\rebar_count\rebar_mvp\counting-api
$env:VISION_MODEL_PATH="D:\0\PythonDemo\rebar_count\model_recall_boost\model_final.pth"
$env:VISION_SCORE_THRESHOLD="0.62"
$env:VISION_NMS_IOU="0.35"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:

- `http://127.0.0.1:8000/`

### MVP APIs

- `POST /api/v1/tasks` (multipart: `metadata`, `image`, optional `tag_image`)
- `GET /api/v1/tasks/{task_id}`
- `GET /api/v1/tasks/{task_id}/result`
- `POST /api/v1/specs/normalize`

---

## 2) Dataset

Current evaluation/training dataset path:

- `C:\Users\76884\Downloads\dataset_reinforcing_steel_bar_counting`

Expected structure:

```text
dataset_reinforcing_steel_bar_counting/
  JPEGImages/
  Annotations/
```

---

## 3) Evaluate Current Pipeline

The evaluation matches runtime post-processing (`nms + dedup + primary-cluster filter`).

### Full-set evaluation

```powershell
$env:PYTHONPATH="D:\0\PythonDemo\rebar_count\rebar_mvp\counting-api"
python D:\0\PythonDemo\rebar_count\rebar_mvp\counting-api\scripts\eval_current_pipeline.py `
  --dataset-root "C:\Users\76884\Downloads\dataset_reinforcing_steel_bar_counting" `
  --weight "D:\0\PythonDemo\rebar_count\model_recall_boost\model_final.pth" `
  --target-class rebar `
  --score-thr 0.62 `
  --nms-iou 0.35 `
  --cluster-pad 0.12 `
  --iou-thr 0.5 `
  --split all `
  --device auto
```

### Export per-image metrics

```powershell
$env:PYTHONPATH="D:\0\PythonDemo\rebar_count\rebar_mvp\counting-api"
python D:\0\PythonDemo\rebar_count\rebar_mvp\counting-api\scripts\eval_current_pipeline.py `
  --dataset-root "C:\Users\76884\Downloads\dataset_reinforcing_steel_bar_counting" `
  --weight "D:\0\PythonDemo\rebar_count\model_recall_boost\model_final.pth" `
  --target-class rebar `
  --score-thr 0.62 `
  --nms-iou 0.35 `
  --cluster-pad 0.12 `
  --iou-thr 0.5 `
  --split all `
  --device auto `
  --per-image-csv "D:\0\PythonDemo\rebar_count\rebar_mvp\counting-api\artifacts\eval_all_per_image.csv"
```

---

## 4) Parameter Grid Search

```powershell
$env:PYTHONPATH="D:\0\PythonDemo\rebar_count\rebar_mvp\counting-api"
python D:\0\PythonDemo\rebar_count\rebar_mvp\counting-api\scripts\grid_search_current_pipeline.py `
  --python python `
  --eval-script D:\0\PythonDemo\rebar_count\rebar_mvp\counting-api\scripts\eval_current_pipeline.py `
  --dataset-root "C:\Users\76884\Downloads\dataset_reinforcing_steel_bar_counting" `
  --weight "D:\0\PythonDemo\rebar_count\model_recall_boost\model_final.pth" `
  --target-class rebar `
  --split holdout20 `
  --seed 42 `
  --device auto `
  --iou-thr 0.5 `
  --score-list 0.62,0.66,0.70,0.74 `
  --nms-list 0.30,0.35,0.40,0.45 `
  --pad-list 0.06,0.08,0.10,0.12 `
  --out-csv D:\0\PythonDemo\rebar_count\rebar_mvp\counting-api\artifacts\grid_holdout20.csv
```

---

## 5) Retraining (Recall Boost)

### Build hard-case list from per-image eval

```powershell
python D:\0\PythonDemo\rebar_count\src\build_hardcase_list.py `
  --per-image-csv D:\0\PythonDemo\rebar_count\rebar_mvp\counting-api\artifacts\eval_all_per_image.csv `
  --out-txt D:\0\PythonDemo\rebar_count\artifacts\hardcase_ids.txt `
  --fn-threshold 30 `
  --topk 120
```

### Fine-tune from baseline model

```powershell
$env:PYTHONPATH="D:\0\PythonDemo\rebar_count"
python D:\0\PythonDemo\rebar_count\src\train_recall_boost.py `
  --dataset-root "C:\Users\76884\Downloads\dataset_reinforcing_steel_bar_counting" `
  --output-dir D:\0\PythonDemo\rebar_count\model_recall_boost `
  --init-weight D:\0\PythonDemo\rebar_count\model_full\model_final.pth `
  --hardcase-list D:\0\PythonDemo\rebar_count\artifacts\hardcase_ids.txt `
  --hardcase-multiplier 3 `
  --epochs 4 `
  --batch-size 1 `
  --num-workers 0 `
  --val-ratio 0.2 `
  --seed 42 `
  --lr 0.003 `
  --resize-max-side 1600
```

---

## 6) Model Files

Tracked by Git LFS:

- `model_full/model_final.pth`
- `model_smoke/model_final.pth`
- `model_recall_boost/model_final.pth`

If clone is missing weights:

```powershell
git lfs pull
```

---

## 7) Versioning and Rollback

- Baseline runnable tag: `v0.2-baseline-runnable`
- To rollback:

```powershell
git checkout v0.2-baseline-runnable
```

---

## Notes

- Runtime artifacts are ignored:
  - `rebar_mvp/counting-api/storage/`
  - `rebar_mvp/counting-api/artifacts/`
- Training artifacts are partially ignored; only final model files are tracked.
