# Plant Disease Classification Project

A complete deep-learning pipeline for multi-class plant disease classification, including:
- data preprocessing and split validation,
- multiple model baselines (EfficientNet, GoogLeNet, MobileNetV3),
- a DINOv2 + LoRA advanced model,
- and a FastAPI web app for inference.

This repository is organized so you can move from data preparation to training, benchmarking, and deployment in one place.

## 1) Project Overview

- Task: Multi-class image classification of plant diseases.
- Number of classes: 40.
- Input image size: 224 x 224.
- Final prepared split (`train_ready_data`):
  - Train: 6000 images
  - Validation: 1288 images
  - Test: 1284 images
- Data leakage check (train/val/test): 0 overlaps.

Source: `train_ready_data/preprocessing_report.json`.

## 2) Data & Preprocessing

### Raw data

Raw image folders (`data/`, `plant_data_clean/`, `Resized_Data/`, `train_ready_data/`) are **not** tracked in git — they total ~2.6 GB and exceed GitHub's file-size limits. They are produced from public plant-disease datasets (PlantVillage, PlantDoc, and related leaf-disease collections), consolidated into a unified 40-class schema.

To reproduce the pipeline locally, either:
- Place the raw per-class folders under `data/train` and `data/val`, then run the notebook below, or
- Download the prepared final split directly (see "Prepared split download" further down) and skip preprocessing.

### `Data_Processing.ipynb`

All data cleaning and split construction is performed in the root-level notebook [`Data_Processing.ipynb`](Data_Processing.ipynb). It is the single entry point that turns the raw sources into the `train_ready_data/` directory consumed by every training script.

The notebook performs, in order:

1. **Class-name normalization** — lower-case, underscore-separated, unified 40-class taxonomy across heterogeneous source datasets (e.g. merging equivalent labels and resolving naming inconsistencies).
2. **Deduplication & leakage prevention** — perceptual-hash / file-hash based duplicate removal across `train` / `val` / `test` so no image appears in more than one split. The final leakage check is logged in `preprocessing_report.json` (reported overlaps: `0`).
3. **Per-class cleaning** — removes corrupted, zero-byte, or non-image files; drops classes that fall below a minimum sample threshold.
4. **Resizing** — all images resized to `224 x 224` (model input size) and written to `Resized_Data/`.
5. **Stratified splitting** — produces the final `train_ready_data/{train,val,test}/<class>/` structure with per-class balance preserved:
   - Train: **6000**
   - Validation: **1288**
   - Test: **1284**
6. **Reporting** — writes `train_ready_data/preprocessing_report.json` with per-step counts, class distribution, and the leakage check result.

### Running the notebook

```powershell
jupyter notebook Data_Processing.ipynb
```

Run the cells top-to-bottom. Expected output is a populated `train_ready_data/` directory plus `preprocessing_report.json`. After this, every `train.py` in the project can be launched directly — they resolve paths relative to the repo root.

### Prepared split download

Because the data is excluded from git, a mirrored copy of the final `train_ready_data/` split will be made available on Hugging Face Datasets / Kaggle (links to be added after upload). Place the downloaded folder at the repo root so it sits at `./train_ready_data/`.

## 3) Structured Repository Layout

```text
.
|-- README.md
|-- requirements.txt
|-- Academic_Report.pdf
|-- Data_Processing.ipynb
|-- app/
|   |-- app.py
|   |-- index.html
|   |-- requirements.txt
|   `-- README.md
|-- data/
|   |-- train/
|   `-- val/
|-- plant_data_clean/
|   |-- train/
|   `-- val/
|-- Resized_Data/
|   |-- train/
|   `-- val/
|-- train_ready_data/
|   |-- train/
|   |-- val/
|   |-- test/
|   `-- preprocessing_report.json
|-- SOTA/
|   |-- EfficientNET/
|   |   |-- model.py
|   |   |-- train.py
|   |   |-- eval_best.py
|   |   |-- checkpoints/
|   |   `-- artifacts/
|   |-- GoogleNET/
|   |   |-- model.py
|   |   |-- train.py
|   |   `-- artifacts/
|   `-- MobileNET/
|       |-- model.py
|       |-- train.py
|       |-- eval_best.py
|       `-- artifacts/
`-- VIT/
    |-- CLIP/
    |   |-- model.py
    |   |-- train.py
    |   `-- Results/
    `-- DINO_V2/
        |-- model.py
        |-- train.py
        `-- Results/
            |-- checkpoints/
            `-- reports/
```

## 4) Results Summary (All Available Benchmarks)

CNN and DINOv2 baselines are evaluated on the full project test split (`1284` images, `40` classes). CLIP + LoRA was evaluated on a class-balanced test subset (`400` images, `10` per class) drawn from the same pool.

| Model | Test Accuracy | Macro F1 | Weighted F1 | Test Loss | Best Val Accuracy | Test Size | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| DINOv2 + LoRA | **0.9019** | 0.8800 | 0.9000 | N/A | N/A | 1284 | Best overall model |
| EfficientNet-B0 (SOTA/EfficientNET) | 0.7835 | 0.7592 | 0.7803 | 1.3785 | 0.7873 | 1284 | `epoch=30` in test results |
| CLIP ViT-B/32 + LoRA (VIT/CLIP) | 0.7725 | 0.7709 | 0.7709 | 1.3962 | 0.7993 | 400 | ROC-AUC 0.9835; only 317K trainable params (of 151.6M) |
| MobileNetV3 (SOTA/MobileNET) | 0.7726 | 0.7457 | 0.7693 | 1.3710 | 0.8028 | 1284 | `best epoch=26` in metrics text |
| GoogLeNet (SOTA/GoogleNET) | 0.7383 | 0.7141 | 0.7361 | 1.4885 | N/A | 1284 | Device recorded as CPU in results |

### Checkpoints on Hugging Face

All experiment checkpoints (EfficientNet, GoogLeNet, MobileNetV3, CLIP + LoRA, DINOv2 + LoRA) are published on the Hugging Face Hub:

**[SevakGrigoryan/plant_diseases_checkpoints](https://huggingface.co/SevakGrigoryan/plant_diseases_checkpoints)**

Download an individual checkpoint with:

```python
from huggingface_hub import hf_hub_download
import torch

path = hf_hub_download("SevakGrigoryan/plant_diseases_checkpoints", "efficientnet/best.pth")
state = torch.load(path, map_location="cpu")
```

### Result Files by Model

- DINOv2 + LoRA:
  - `VIT/DINO_V2/Results/reports/report.txt`
  - `VIT/DINO_V2/Results/checkpoints/best_model.pth`
- CLIP + LoRA:
  - `VIT/CLIP/Results/results.txt`
  - `VIT/CLIP/Results/test_classification_report.csv`
  - `VIT/CLIP/Results/test_roc_auc.txt`
  - `VIT/CLIP/Results/plots/` (accuracy & loss curves)
- EfficientNet-B0:
  - `SOTA/EfficientNET/artifacts/test_results.json`
  - `SOTA/EfficientNET/artifacts/classification_report.txt`
  - `SOTA/EfficientNET/artifacts/classification_report.json`
- GoogLeNet:
  - `SOTA/GoogleNET/artifacts/test_results.json`
  - `SOTA/GoogleNET/artifacts/classification_report.txt`
  - `SOTA/GoogleNET/artifacts/classification_report.json`
- MobileNetV3:
  - `SOTA/MobileNET/artifacts/test_metrics.txt`
  - `SOTA/MobileNET/artifacts/classification_report.txt`

## 5) Environment Setup

Recommended Python version: 3.10+.

### Create a virtual environment and install dependencies

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you train from scratch on GPU, install the CUDA-enabled PyTorch build that matches your system.

## 6) Training Commands

Run from the repository root unless stated otherwise.

### EfficientNet

```powershell
python SOTA/EfficientNET/train.py
```

### GoogLeNet

```powershell
python SOTA/GoogleNET/train.py
```

### MobileNetV3

```powershell
python SOTA/MobileNET/train.py
```

### DINOv2 + LoRA

```powershell
python VIT/DINO_V2/train.py
```

### CLIP + LoRA

```powershell
python VIT/CLIP/train.py
```

## 7) Run the Inference Web App

```powershell
python -m uvicorn app.app:app --reload --host 127.0.0.1 --port 8000
```

Open: `http://127.0.0.1:8000`


## 8) Key Takeaways

- DINOv2 + LoRA currently gives the strongest test performance (`90.19%`).
- EfficientNet-B0 is the best CNN baseline (`78.35%`), narrowly ahead of MobileNetV3 (`77.26%`).
- CLIP ViT-B/32 + LoRA reaches `77.25%` while training only `317K` of `151.6M` parameters — the most parameter-efficient model in the benchmark.
- GoogLeNet provides a lower baseline (`73.83%`) useful for comparison.
- The prepared dataset split is clean and leakage-free, making model comparison consistent.
