import gc
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import wandb

from model import build_model

class Config:
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parents[1]

    data_root = str((project_root / "train_ready_data").resolve())
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    image_size = 224
    batch_size = 32
    epochs = 50
    weight_decay = 1e-4
    seed = 42

    hidden_dim = 1024

    unfreeze_epoch = 10
    unfreeze_blocks = [8, 9, 10, 11]
    unfreeze_lr = 5e-6

    lora_lr = 2e-4
    head_lr = 1e-3
    other_lr = 5e-5

    exp_dir = str((base_dir / "Results").resolve())
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    report_dir = os.path.join(exp_dir, "reports")

    last_checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

    wandb_project = "DINO_V2"
    wandb_run_name = "DINO_V2_50_EPOCHS"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = torch.cuda.is_available()
    num_workers = 2

    resume_training = False

    save_last_every_epoch = True

CFG = Config()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_dirs() -> None:
    Path(CFG.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(CFG.report_dir).mkdir(parents=True, exist_ok=True)

def build_class_weights(train_loader, num_classes: int, device: str):
    label_counts = Counter(train_loader.dataset.targets)
    total = sum(label_counts.values())
    weights = torch.tensor(
        [total / (num_classes * label_counts[i]) for i in range(num_classes)],
        dtype=torch.float32,
        device=device,
    )
    return weights

def cleanup_temp_file(path: str):
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

def save_resume_checkpoint(path, epoch, model, optimizer, scheduler, scaler, history, best_val_acc):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "history": history,
        "best_val_acc": best_val_acc,
    }
    tmp_path = path + ".tmp"
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        print(f"Resume checkpoint save failed: {e}")
        cleanup_temp_file(tmp_path)
        return False

def save_model_only_checkpoint(path, epoch, model, best_val_acc):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_val_acc": best_val_acc,
    }
    tmp_path = path + ".tmp"
    try:
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        print(f"Best model save failed: {e}")
        cleanup_temp_file(tmp_path)
        return False

def load_resume_checkpoint(path, model, optimizer, scheduler, scaler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    history = checkpoint.get("history", {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []})
    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    return start_epoch, history, best_val_acc

def load_model_only_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint.get("epoch", -1)
    best_val_acc = checkpoint.get("best_val_acc", 0.0)
    return epoch, best_val_acc

def get_transforms(image_size: int):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.RandomRotation(degrees=25),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.08)
        ], p=0.8),
        transforms.RandomAffine(degrees=20, translate=(0.12, 0.12), scale=(0.85, 1.15), shear=12),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.35),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.35, scale=(0.02, 0.20), ratio=(0.3, 3.3), value="random"),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform

def build_dataloaders():
    train_transform, eval_transform = get_transforms(CFG.image_size)
    train_dataset = datasets.ImageFolder(CFG.train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(CFG.val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(CFG.test_dir, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_dataset.classes

def should_unfreeze_name(name: str, block_ids: list[int]) -> bool:
    return any(f".layer.{i}." in name for i in block_ids)

def set_unfreeze_blocks(model, block_ids: list[int]):
    changed = 0
    for name, p in model.named_parameters():
        if should_unfreeze_name(name, block_ids):
            if not p.requires_grad:
                p.requires_grad = True
                changed += 1
    return changed

def build_optimizer(model):
    lora_params, head_params, late_backbone_params, other_params = [], [], [], []
    for name, p in model.named_parameters():
        if "head" in name:
            head_params.append(p)
        elif "lora" in name.lower():
            lora_params.append(p)
        elif should_unfreeze_name(name, CFG.unfreeze_blocks):
            late_backbone_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": CFG.lora_lr},
            {"params": head_params, "lr": CFG.head_lr},
            {"params": late_backbone_params, "lr": CFG.unfreeze_lr},
            {"params": other_params, "lr": CFG.other_lr},
        ],
        weight_decay=CFG.weight_decay,
    )
    return optimizer

def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if CFG.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        total_loss += loss.item() * images.size(0)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset), accuracy_score(all_labels, all_preds)

@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Evaluating"):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        if CFG.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader.dataset), accuracy_score(all_labels, all_preds), all_labels, all_preds

@torch.no_grad()
def evaluate_tta(model, loader, criterion, device, n_aug=5, desc="Evaluating TTA"):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)
        logits_sum = None
        for _ in range(n_aug):
            if CFG.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
            else:
                logits = model(images)
            logits_sum = logits if logits_sum is None else logits_sum + logits

        logits_avg = logits_sum / n_aug
        if CFG.use_amp:
            logits_avg = logits_avg.float()

        loss = criterion(logits_avg, labels)
        total_loss += loss.item() * images.size(0)
        all_preds.extend(torch.argmax(logits_avg, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader.dataset), accuracy_score(all_labels, all_preds), all_labels, all_preds

def main():
    set_seed(CFG.seed)
    create_dirs()
    train_loader, val_loader, test_loader, class_names = build_dataloaders()

    num_classes = len(class_names)
    model = build_model(num_classes=num_classes, device=CFG.device)

    optimizer = build_optimizer(model)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[CFG.lora_lr, CFG.head_lr, CFG.unfreeze_lr, CFG.other_lr],
        steps_per_epoch=len(train_loader),
        epochs=CFG.epochs,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=CFG.use_amp)
    class_weights = build_class_weights(train_loader, num_classes, CFG.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, start_epoch = 0.0, 1

    if CFG.resume_training and os.path.exists(CFG.last_checkpoint_path):
        start_epoch, history, best_val_acc = load_resume_checkpoint(
            CFG.last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            scaler,
            CFG.device,
        )

    wandb.init(project=CFG.wandb_project, name=CFG.wandb_run_name)

    for epoch in range(start_epoch, CFG.epochs + 1):
        print(f"\nEpoch {epoch}/{CFG.epochs}")
        if epoch == CFG.unfreeze_epoch:
            set_unfreeze_blocks(model, CFG.unfreeze_blocks)
            gc.collect()
            torch.cuda.empty_cache()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, CFG.device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, CFG.device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
            }
        )

        if CFG.save_last_every_epoch:
            save_resume_checkpoint(
                CFG.last_checkpoint_path,
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
                history,
                best_val_acc,
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model_only_checkpoint(CFG.best_model_path, epoch, model, best_val_acc)
            print(f"New best val acc: {best_val_acc:.4f}")

    if os.path.exists(CFG.best_model_path):
        load_model_only_checkpoint(CFG.best_model_path, model, CFG.device)

    _, test_acc, y_true, y_pred = evaluate_tta(model, test_loader, criterion, CFG.device, n_aug=5)
    print(f"\nFinal Test Accuracy (TTA): {test_acc:.4f}")

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)
    wandb.finish()

if __name__ == "__main__":
    main()
