# Plant Disease Classification Project

This repo is my end-to-end take on classifying plant diseases from leaf photos. It covers everything from cleaning the data to training a few different models, comparing them, and serving the best one through a small FastAPI web app.

The models I tried: EfficientNet, GoogLeNet, MobileNetV3, CLIP + LoRA, and DINOv2 + LoRA.

## 1) Project Overview

It's a 40-class image classification problem, with all images resized to 224 x 224. After cleaning and splitting, the final dataset has 6000 training images, 1288 for validation, and 1284 for testing — with no overlap between any of the splits.

## 2) Data & Preprocessing

The raw image folders aren't in git — together they're around 2.6 GB, which is too big for GitHub.

If you want to reproduce things locally, you have two options: either drop the raw class folders into `data/train` and `data/val` and run the notebook, or grab the already-prepared split and skip straight to training.

### `Data_Processing.ipynb`

All the cleaning happens in [`Data_Processing.ipynb`](Data_Processing.ipynb) at the repo root. Running it top-to-bottom does roughly this:

- normalizes class names into one consistent 40-class taxonomy,
- removes duplicates across train/val/test using image hashing (final overlap check: 0),
- throws away corrupted or empty files and drops classes that are too small,
- resizes everything to 224 x 224 into `Resized_Data/`,
- builds the stratified `train_ready_data/{train,val,test}/<class>/` split,
- writes a `preprocessing_report.json` with all the counts.

```powershell
jupyter notebook Data_Processing.ipynb
```

Once that's done, any `train.py` in the repo will just work — they all resolve paths from the repo root.

### Prepared split download

Since the data isn't in git, I'll be uploading a mirror of the final `train_ready_data/` split to Hugging Face / Kaggle (links coming once it's up). Just drop the folder at the repo root as `./train_ready_data/`.

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
