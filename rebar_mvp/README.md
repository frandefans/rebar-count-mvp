# Rebar MVP (Single Image)

## Start

```powershell
cd D:\0\PythonDemo\rebar_count\rebar_mvp\counting-api
python -m pip install -r requirements-mvp.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open: `http://127.0.0.1:8000/`

## API

- `POST /api/v1/tasks` (multipart: `metadata`, `image`, optional `tag_image`)
- `GET /api/v1/tasks/{task_id}`
- `GET /api/v1/tasks/{task_id}/result`
- `POST /api/v1/specs/normalize`

## Model Weights

- `model_full/model_final.pth`
- `model_smoke/model_final.pth`

Both are tracked by Git LFS.
