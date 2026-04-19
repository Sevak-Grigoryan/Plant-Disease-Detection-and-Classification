# Plant Disease Classifier — Web App

This is a small web app that takes a picture of a leaf and tells you which
plant disease it has (40 classes). It uses my fine-tuned DINOv2-Large + LoRA
model, pulled straight from Hugging Face.

## Before you start: the model is big

The model is about **1.2 GB**, so the first time you start the app it needs to
download it from Hugging Face. On a normal home connection this takes a
**couple of minutes**. After that it's cached on your machine and starts in
a few seconds.

Even once it's downloaded, loading it into memory takes another
**10–30 seconds** on CPU. The app isn't ready until you see
`Application startup complete.` in the terminal. If you try to use it before
then, you'll get a 503 — that just means "still loading, wait a bit".

You need roughly **2 GB of free RAM** and **2 GB of free disk** for the HF
cache. It works on CPU; a GPU just makes predictions faster.

## How to run it

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r app/requirements.txt
python -m uvicorn app.app:app --host 127.0.0.1 --port 8000
```

The first run will spend a while downloading the model. Just let it finish.

Once it says `Application startup complete.`, open
**http://127.0.0.1:8000/** in your browser, drop in a leaf photo, and click
Predict.

## Checking it's ready

If predictions are giving you 503 errors, check
**http://127.0.0.1:8000/health**. You'll see something like:

```json
{"status":"ok","model_loaded":true,"device":"cpu","classes_count":40,"error":null}
```

- `status: ok` → ready to go.
- `status: degraded` → the model didn't load. The `error` field tells you why.

## Using the API directly

Full interactive docs: **http://127.0.0.1:8000/docs**

Quick example with curl:

```powershell
curl -X POST "http://127.0.0.1:8000/api/predict?top_k=3" -F "file=@leaf.jpg"
```

You get back the top class, its confidence, and the top-k runners-up.

## If something goes wrong

- **503 on predict** → model is still loading (or failed). Check `/health`.
- **First download stuck** → delete
  `~/.cache/huggingface/hub/models--SevakGrigoryan--dinov2-large-plant-disease`
  and try again.
- **ImportError** → `pip install -r app/requirements.txt` again.
- **Out of memory** → close other heavy apps, DINOv2-Large wants ~2 GB RAM.

That's it. The default HF repo is public, so no token or login is needed.
