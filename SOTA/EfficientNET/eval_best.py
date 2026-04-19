import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model, calculate_parameters, print_parameter_summary

IMAGE_SIZE = 224
BATCH_SIZE = 48
NUM_WORKERS = 4
LABEL_SMOOTHING = 0.1

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "train_ready_data"
TEST_DIR = DATA_ROOT / "test"

CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best.pth"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_test_loader() -> tuple[DataLoader, list[str]]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_ds = datasets.ImageFolder(TEST_DIR, transform=test_tf)
    loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader, test_ds.classes

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)

        running_loss += loss.item()
        running_acc += (preds == targets).float().mean().item()
        n_batches += 1

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

    return (
        running_loss / n_batches,
        running_acc / n_batches,
        np.array(all_targets),
        np.array(all_preds),
    )

def load_checkpoint(model: nn.Module, device: torch.device) -> dict:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    return checkpoint if isinstance(checkpoint, dict) else {}

def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    device = get_device()
    test_loader, class_names = build_test_loader()

    model = build_model(
        num_classes=len(class_names),
        pretrained=False,
        dropout=0.3,
    ).to(device)
    checkpoint = load_checkpoint(model, device)

    print_parameter_summary(model)
    params_info = calculate_parameters(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, digits=4)
    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    with open(ARTIFACTS_DIR / "model_parameters.json", "w", encoding="utf-8") as f:
        json.dump(params_info, f, indent=2)

    with open(ARTIFACTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(ARTIFACTS_DIR / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    np.savetxt(ARTIFACTS_DIR / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    test_results = {
        "checkpoint_path": str(CHECKPOINT_PATH),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "num_test_samples": int(len(y_true)),
        "best_val_accuracy": float(checkpoint.get("best_val_acc", -1.0)),
        "epoch": int(checkpoint.get("epoch", -1)),
    }
    with open(ARTIFACTS_DIR / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc : {test_acc:.4f}")
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")

if __name__ == "__main__":
    main()
