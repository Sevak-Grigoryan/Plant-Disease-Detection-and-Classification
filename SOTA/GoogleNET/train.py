import copy
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model, calculate_parameters, print_parameter_summary

SEED = 42
NUM_EPOCHS = 30
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
IMAGE_SIZE = 224

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "train_ready_data"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_transforms() -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.08),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=8),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
            normalize,
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_tf, eval_tf, eval_tf

def build_loaders() -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    train_tf, val_tf, test_tf = build_transforms()

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_tf)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=test_tf)

    class_names = train_ds.classes
    if val_ds.classes != class_names or test_ds.classes != class_names:
        raise ValueError("Class order mismatch across train/val/test. Ensure same class folders in all splits.")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, class_names

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, targets)
        n_batches += 1

    return running_loss / n_batches, running_acc / n_batches

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    n_batches = 0

    all_preds = []
    all_targets = []

    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

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

def plot_curves(history: Dict[str, List[float]], save_path: Path) -> None:
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def save_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (Test)",
    )

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val_acc: float,
    history: Dict[str, List[float]],
    path: Path,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "history": history,
        },
        path,
    )

def main() -> None:
    set_seed(SEED)
    device = get_device()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_DIR.exists() or not VAL_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Expected dataset folders not found under: {DATA_ROOT}. "
            "Please create train/ val/ test directories."
        )

    train_loader, val_loader, test_loader, class_names = build_loaders()

    model = build_model(num_classes=len(class_names), pretrained=True, dropout=0.2).to(device)
    print_parameter_summary(model)

    params_info = calculate_parameters(model)
    with open(ARTIFACTS_DIR / "model_parameters.json", "w", encoding="utf-8") as f:
        json.dump(params_info, f, indent=2)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        latest_ckpt = CHECKPOINT_DIR / "latest.pth"
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, history, latest_ckpt)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            best_ckpt = CHECKPOINT_DIR / "best.pth"
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, history, best_ckpt)

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    model.load_state_dict(best_weights)

    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc : {test_acc:.4f}")

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, digits=4)
    report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    cm = confusion_matrix(y_true, y_pred)

    with open(ARTIFACTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(ARTIFACTS_DIR / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    np.savetxt(ARTIFACTS_DIR / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    save_confusion_matrix(cm, class_names, ARTIFACTS_DIR / "confusion_matrix.png")

    test_results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "num_test_samples": int(len(y_true)),
        "best_val_accuracy": float(best_val_acc),
    }
    with open(ARTIFACTS_DIR / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(test_results, f, indent=2)

    with open(ARTIFACTS_DIR / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_curves(history, ARTIFACTS_DIR / "train_val_curves.png")

    print("Training and evaluation complete.")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    print(f"Artifacts saved to  : {ARTIFACTS_DIR}")

if __name__ == "__main__":
    main()
