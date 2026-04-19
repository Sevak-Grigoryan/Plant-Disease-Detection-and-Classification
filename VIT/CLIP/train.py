import os
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model, count_parameters

warnings.filterwarnings("ignore")

class Config:

    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parents[1]

    data_root = project_root / "train_ready_data"
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    results_dir = base_dir / "Results"
    plots_dir = results_dir / "plots"
    ckpt_dir = results_dir / "checkpoints"

    project = "clip-lora-plant-disease"
    run_name = "clip-vit-base-patch32-lora"

    model_name = "openai/clip-vit-base-patch32"
    lora_r = 2
    lora_alpha = 8
    lora_dropout = 0.10
    target_modules = ("q_proj", "k_proj", "v_proj", "out_proj")

    img_size = 224
    batch_size = 32
    epochs = 100
    lr = 3e-4
    weight_decay = 1e-4
    label_smoothing = 0.1
    patience_early_stop = 12

    scheduler_factor = 0.5
    scheduler_patience = 5
    scheduler_threshold = 0.001
    scheduler_cooldown = 1
    scheduler_min_lr = 1e-6

    seed = 42
    num_workers = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = torch.cuda.is_available()

CFG = Config()

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_dirs() -> None:
    CFG.results_dir.mkdir(parents=True, exist_ok=True)
    CFG.plots_dir.mkdir(parents=True, exist_ok=True)
    CFG.ckpt_dir.mkdir(parents=True, exist_ok=True)

def save_model(state_dict: dict, path: Path) -> None:
    cpu_state = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    torch.save(cpu_state, str(path))

def compute_multiclass_auc(y_true, y_prob, num_classes: int) -> float:
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    try:
        return roc_auc_score(y_true_bin, y_prob, multi_class="ovr")
    except Exception:
        return float("nan")

def get_transforms(img_size: int):
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std),
    ])

    return train_tf, eval_tf

def build_dataloaders():
    train_tf, eval_tf = get_transforms(CFG.img_size)
    train_ds = datasets.ImageFolder(str(CFG.train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(CFG.val_dir), transform=eval_tf)
    test_ds = datasets.ImageFolder(str(CFG.test_dir), transform=eval_tf)

    pin = CFG.device == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=CFG.batch_size, shuffle=True,
        num_workers=CFG.num_workers, pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=CFG.batch_size, shuffle=False,
        num_workers=CFG.num_workers, pin_memory=pin,
    )
    return train_loader, val_loader, test_loader, train_ds.classes, train_ds.targets

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader)

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=CFG.use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        running_loss += loss.item()
        correct += (preds == y).sum().item()
        total += y.size(0)

        if (batch_idx + 1) % max(1, n_batches // 5) == 0 or (batch_idx + 1) == n_batches:
            print(
                f"  [Train {batch_idx+1}/{n_batches}] "
                f"loss={running_loss/(batch_idx+1):.4f}  acc={correct/total:.4f}",
                flush=True,
            )

    return running_loss / n_batches, correct / total

def validate_one_epoch(model, loader, criterion, device, num_classes: int):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    n_batches = len(loader)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=CFG.use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            running_loss += loss.item()
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    auc_score = compute_multiclass_auc(all_labels, all_probs, num_classes)
    return (
        running_loss / n_batches,
        correct / total,
        auc_score,
        np.array(all_labels),
        np.array(all_preds),
        all_probs,
    )

def save_confusion_matrix(test_labels, test_preds, path: Path):
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, cmap="Blues")
    plt.title("Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close()

def save_roc_curves(test_labels, test_probs, num_classes, class_names, path: Path):
    test_labels_bin = label_binarize(test_labels, classes=list(range(num_classes)))
    plt.figure(figsize=(12, 9))
    for i in range(num_classes):
        try:
            fpr, tpr, _ = roc_curve(test_labels_bin[:, i], test_probs[:, i])
            class_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={class_auc:.2f})")
        except Exception:
            continue
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Class ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close()

def save_history_plots(history_df: pd.DataFrame):
    loss_path = CFG.plots_dir / "loss_curve.png"
    plt.figure(figsize=(8, 6))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(str(loss_path), dpi=300, bbox_inches="tight"); plt.close()

    acc_path = CFG.plots_dir / "accuracy_curve.png"
    plt.figure(figsize=(8, 6))
    plt.plot(history_df["epoch"], history_df["train_acc"], label="Train Accuracy")
    plt.plot(history_df["epoch"], history_df["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(str(acc_path), dpi=300, bbox_inches="tight"); plt.close()

    auc_path = CFG.plots_dir / "val_auc_curve.png"
    plt.figure(figsize=(8, 6))
    plt.plot(history_df["epoch"], history_df["val_roc_auc"], label="Val ROC-AUC")
    plt.xlabel("Epoch"); plt.ylabel("ROC-AUC"); plt.title("Validation ROC-AUC Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(str(auc_path), dpi=300, bbox_inches="tight"); plt.close()

    lr_path = CFG.plots_dir / "lr_curve.png"
    plt.figure(figsize=(8, 6))
    plt.plot(history_df["epoch"], history_df["lr"], label="Learning Rate")
    plt.xlabel("Epoch"); plt.ylabel("LR"); plt.title("Learning Rate Curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(str(lr_path), dpi=300, bbox_inches="tight"); plt.close()

    return loss_path, acc_path, auc_path, lr_path

def main() -> None:
    set_seed(CFG.seed)
    create_dirs()

    print("=" * 57)
    print(f"Using device: {CFG.device}")
    print(f"Data root:    {CFG.data_root}")
    print(f"Results dir:  {CFG.results_dir}")
    print("=" * 57)

    train_loader, val_loader, test_loader, class_names, train_targets = build_dataloaders()
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_targets),
        y=train_targets,
    )
    weights = torch.tensor(weights, dtype=torch.float32).to(CFG.device)

    model = build_model(
        num_classes=num_classes,
        device=CFG.device,
        model_name=CFG.model_name,
        lora_r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        target_modules=CFG.target_modules,
    )

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    wandb.init(
        project=CFG.project,
        name=CFG.run_name,
        config={k: v for k, v in vars(CFG).items() if not k.startswith("_") and not callable(v)},
        reinit=True,
    )
    wandb.log({"total_params": total_params, "trainable_params": trainable_params})
    wandb.watch(model, log="all", log_freq=100)

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=CFG.label_smoothing)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=CFG.scheduler_factor,
        patience=CFG.scheduler_patience,
        threshold=CFG.scheduler_threshold,
        threshold_mode="rel",
        cooldown=CFG.scheduler_cooldown,
        min_lr=CFG.scheduler_min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=CFG.use_amp)

    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_roc_auc": [], "lr": [],
    }
    best_val_acc, best_val_auc, early_stop_counter = 0.0, 0.0, 0

    best_acc_path = CFG.ckpt_dir / "best_val_acc.pth"
    best_auc_path = CFG.ckpt_dir / "best_val_auc.pth"

    for epoch in range(CFG.epochs):
        print(f"\nEpoch [{epoch+1}/{CFG.epochs}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, CFG.device,
        )
        val_loss, val_acc, val_roc_auc, _, _, _ = validate_one_epoch(
            model, val_loader, criterion, CFG.device, num_classes,
        )

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Val ROC-AUC: {val_roc_auc:.4f} | LR: {current_lr:.7f}",
            flush=True,
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_roc_auc"].append(val_roc_auc)
        history["lr"].append(current_lr)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_roc_auc": val_roc_auc,
            "learning_rate": current_lr,
            "best_val_acc_so_far": best_val_acc,
            "best_val_auc_so_far": best_val_auc,
        })

        improved = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model.state_dict(), best_acc_path)
            print("  --> Saved BEST model by validation accuracy")
            improved = True
        if val_roc_auc > best_val_auc:
            best_val_auc = val_roc_auc
            save_model(model.state_dict(), best_auc_path)
            print("  --> Saved BEST model by validation ROC-AUC")
            improved = True

        early_stop_counter = 0 if improved else early_stop_counter + 1
        if not improved:
            print(f"  Early stopping counter: {early_stop_counter}/{CFG.patience_early_stop}")

        history_df = pd.DataFrame(history)
        history_df.to_csv(str(CFG.results_dir / "history.csv"), index=False)

        if early_stop_counter >= CFG.patience_early_stop:
            print("Early stopping triggered.")
            break

    model.load_state_dict(torch.load(str(best_acc_path), map_location=CFG.device, weights_only=True))
    model.eval()
    print(f"\nLoaded best model from: {best_acc_path}")

    test_loss, test_acc, test_roc_auc, test_labels, test_preds, test_probs = validate_one_epoch(
        model, test_loader, criterion, CFG.device, num_classes,
    )
    print(f"\nTest Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test ROC-AUC:  {test_roc_auc:.4f}")

    report_str = classification_report(test_labels, test_preds, target_names=class_names, digits=4)
    report_dict = classification_report(
        test_labels, test_preds, target_names=class_names, output_dict=True, digits=4,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = CFG.results_dir / "test_classification_report.csv"
    report_df.to_csv(str(report_csv_path))

    cm_path = CFG.plots_dir / "test_confusion_matrix.png"
    save_confusion_matrix(test_labels, test_preds, cm_path)

    roc_curve_path = CFG.plots_dir / "test_roc_auc_curve.png"
    save_roc_curves(test_labels, test_probs, num_classes, class_names, roc_curve_path)

    history_df = pd.DataFrame(history)
    loss_path, acc_path, auc_path, lr_path = save_history_plots(history_df)

    results_txt_path = CFG.results_dir / "results.txt"
    with open(results_txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("FINAL TEST RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total Parameters         : {total_params:,}\n")
        f.write(f"Trainable Parameters     : {trainable_params:,}\n\n")
        f.write(f"Best Validation Accuracy : {best_val_acc:.4f}\n")
        f.write(f"Best Validation ROC-AUC  : {best_val_auc:.4f}\n\n")
        f.write(f"Test Accuracy            : {test_acc:.4f}\n")
        f.write(f"Test Loss                : {test_loss:.4f}\n")
        f.write(f"Test ROC-AUC             : {test_roc_auc:.4f}\n\n")
        f.write("=" * 70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(report_str + "\n")

    with open(CFG.results_dir / "test_roc_auc.txt", "w") as f:
        f.write(str(test_roc_auc))
    with open(CFG.results_dir / "class_names.txt", "w") as f:
        for idx, name in enumerate(class_names):
            f.write(f"{idx}\t{name}\n")

    wandb.log({
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "test_roc_auc": test_roc_auc,
        "test_confusion_matrix": wandb.Image(str(cm_path)),
        "test_roc_curve": wandb.Image(str(roc_curve_path)),
        "loss_curve": wandb.Image(str(loss_path)),
        "accuracy_curve": wandb.Image(str(acc_path)),
        "val_auc_curve": wandb.Image(str(auc_path)),
        "lr_curve": wandb.Image(str(lr_path)),
    })

    for fpath in [
        results_txt_path, report_csv_path, CFG.results_dir / "history.csv",
        cm_path, roc_curve_path, loss_path, acc_path, auc_path, lr_path,
        CFG.results_dir / "test_roc_auc.txt", CFG.results_dir / "class_names.txt",
        best_acc_path, best_auc_path,
    ]:
        wandb.save(str(fpath))

    wandb.finish()

    print("\nDONE")
    print(f"Results folder:   {CFG.results_dir}")
    print(f"Best model:       {best_acc_path}")
    print(f"Trainable params: {trainable_params:,}")

if __name__ == "__main__":
    main()
