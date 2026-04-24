# src/train_dl.py
#
# ✅ RTX 2050 4GB OPTIMIZED — TARGET 82%+ ACCURACY
# Key changes from last run (which gave 73%):
#   1.  UNFREEZE_EPOCH  = 0  — disabled, caused OOM + hurt accuracy
#   2.  WARMUP_EPOCHS   = 0  — disabled, conflicted with cosine restarts
#   3.  ACCUM_STEPS     = 1  — disabled, noisy gradients at batch=8
#   4.  MixUp alpha     = 0.2 — less aggressive, more stable at small batch
#   5.  MixUp/CutMix prob = 40%/40% — 20% chance no augmentation per batch
#   6.  Label smoothing = 0.05 — 0.1 was too aggressive for 30 classes
#   7.  Early stopping  = patience=10, min_delta=0.0005 — catches small gains
#   8.  num_workers     = 2  — Windows safe value
#   9.  Default batch   = 8  — max safe for RTX 4GB at 380x380
#   10. Default epochs  = 80 — more epochs since no staged unfreezing

import os
import json
import csv
import heapq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ─────────────────────────────────────────────
# IMAGE SIZE — EfficientNet-B4 native 380x380
# ─────────────────────────────────────────────
IMAGE_SIZE    = 380
RESIZE_TO     = 420
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
# RTX 2050 4GB SAFE SETTINGS
# Staged unfreezing and gradient accumulation
# both hurt accuracy on 4GB GPU — disabled.
# ─────────────────────────────────────────────
UNFREEZE_EPOCH = 0   # 0 = disabled — full fine-tuning from epoch 1
WARMUP_EPOCHS  = 0   # 0 = disabled — cosine schedule starts immediately
ACCUM_STEPS    = 1   # 1 = disabled — standard single-step update


# ══════════════════════════════════════════════
# MIXUP / CUTMIX
# Alpha lowered to 0.2 for stability at batch=8
# ══════════════════════════════════════════════
class MixUpCutMix:
    def __init__(self, num_classes, alpha=0.2, mixup_prob=0.4, cutmix_prob=0.4):
        self.num_classes = num_classes
        self.alpha       = alpha
        self.mixup_prob  = mixup_prob
        self.cutmix_prob = cutmix_prob

    def mixup(self, images, labels_onehot):
        lam          = np.random.beta(self.alpha, self.alpha)
        idx          = torch.randperm(images.size(0), device=images.device)
        mixed        = lam * images + (1 - lam) * images[idx]
        mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[idx]
        return mixed, mixed_labels

    def cutmix(self, images, labels_onehot):
        lam       = np.random.beta(self.alpha, self.alpha)
        idx       = torch.randperm(images.size(0), device=images.device)
        _, _, H, W = images.shape
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h     = int(H * cut_ratio)
        cut_w     = int(W * cut_ratio)
        cx = np.random.randint(W);  cy = np.random.randint(H)
        x1 = max(cx - cut_w // 2, 0);  x2 = min(cx + cut_w // 2, W)
        y1 = max(cy - cut_h // 2, 0);  y2 = min(cy + cut_h // 2, H)
        images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
        lam_actual   = 1 - (x2 - x1) * (y2 - y1) / (H * W)
        mixed_labels = lam_actual * labels_onehot + (1 - lam_actual) * labels_onehot[idx]
        return images, mixed_labels

    def __call__(self, images, labels, device):
        labels_onehot = torch.zeros(
            labels.size(0), self.num_classes, device=device
        ).scatter_(1, labels.unsqueeze(1), 1.0)
        r = np.random.rand()
        if r < self.mixup_prob:
            return self.mixup(images, labels_onehot)
        elif r < self.mixup_prob + self.cutmix_prob:
            return self.cutmix(images, labels_onehot)
        # 20% chance: no augmentation — return hard labels
        return images, labels_onehot


# ══════════════════════════════════════════════
# EARLY STOPPING
# patience=10, min_delta=0.0005 catches small
# but real improvements without stopping too early
# ══════════════════════════════════════════════
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0005):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None
        self.stop       = False

    def __call__(self, val_acc):
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_score = val_acc
            self.counter    = 0


# ══════════════════════════════════════════════
# TOP-3 CHECKPOINT SAVER
# ══════════════════════════════════════════════
class TopKCheckpoints:
    def __init__(self, out_path, k=3):
        self.k        = k
        self.out_path = out_path
        self.heap     = []

    def save(self, model, val_acc, epoch):
        base = self.out_path.replace(".pth", "")
        ckpt = f"{base}_e{epoch+1}_acc{val_acc:.4f}.pth"
        torch.save(model.state_dict(), ckpt)
        heapq.heappush(self.heap, (val_acc, ckpt))
        if len(self.heap) > self.k:
            _, worst_path = heapq.heappop(self.heap)
            if os.path.exists(worst_path) and worst_path != self.out_path:
                os.remove(worst_path)


# ══════════════════════════════════════════════
# EVALUATION WITH 7-VIEW TTA
# ══════════════════════════════════════════════
def evaluate_model(model, loader, device, num_classes, use_tta=False):
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    tta_transforms = [
        transforms.Compose([
            transforms.Resize(RESIZE_TO, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize(RESIZE_TO, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize(RESIZE_TO, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize(RESIZE_TO, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomRotation(degrees=(-10, -10)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize(int(RESIZE_TO * 1.1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize(RESIZE_TO, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.Resize(RESIZE_TO, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.2),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
    ]

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if use_tta:
                batch_probs = torch.zeros(inputs.size(0), num_classes, device=device)
                for _ in range(len(tta_transforms)):
                    with autocast(enabled=(device.type == "cuda")):
                        out = model(inputs)
                    batch_probs += torch.softmax(out, dim=1)
                batch_probs /= len(tta_transforms)
                preds = batch_probs.argmax(dim=1)
            else:
                with autocast(enabled=(device.type == "cuda")):
                    outputs = model(inputs)
                batch_probs = torch.softmax(outputs, dim=1)
                preds       = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(batch_probs.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    accuracy           = np.mean(all_preds == all_labels)
    precision_macro    = precision_score(all_labels, all_preds, average='macro',    zero_division=0)
    recall_macro       = recall_score(all_labels,    all_preds, average='macro',    zero_division=0)
    precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_weighted    = recall_score(all_labels,    all_preds, average='weighted', zero_division=0)
    f1_macro           = f1_score(all_labels,        all_preds, average='macro',    zero_division=0)
    f1_weighted        = f1_score(all_labels,        all_preds, average='weighted', zero_division=0)

    return (accuracy, all_labels, all_preds, all_probs,
            precision_macro, recall_macro,
            precision_weighted, recall_weighted,
            f1_macro, f1_weighted)


# ══════════════════════════════════════════════
# MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════
def train_model(data_dir, out_path, epochs=50, batch_size=8, lr=3e-4, resume=False):

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"\n{'='*60}")
    print(f"  RTX 2050 4GB OPTIMIZED TRAINING")
    print(f"{'='*60}")
    print(f"  Device  : {device}")
    if device.type == "cuda":
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
        print(f"  AMP     : enabled (float16)")

    train_dir = Path(data_dir) / 'train'
    val_dir   = Path(data_dir) / 'val'
    out_dir   = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)

    # ── TRAINING TRANSFORMS ──────────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize(RESIZE_TO, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.70, 1.0),
            ratio=(0.85, 1.15),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(
            degrees=15,
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.TrivialAugmentWide(
            num_magnitude_bins=31,
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4,
            saturation=0.3, hue=0.08
        ),
        transforms.RandomGrayscale(p=0.08),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(
            p=0.25, scale=(0.02, 0.15),
            ratio=(0.3, 3.3), value='random'
        ),
    ])

    # ── VALIDATION TRANSFORMS ────────────────────────────────────────────
    val_tf = transforms.Compose([
        transforms.Resize(RESIZE_TO, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds    = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds      = datasets.ImageFolder(val_dir,   transform=val_tf)
    num_classes = len(train_ds.classes)

    print(f"\n  Classes : {num_classes}")
    print(f"  Train   : {len(train_ds)} images")
    print(f"  Val     : {len(val_ds)} images")

    # ── CLASS WEIGHTS ─────────────────────────────────────────────────────
    class_counts         = np.array([
        len(list((train_dir / c).glob("*")))
        for c in train_ds.classes
    ])
    class_weights        = 1.0 / (class_counts + 1e-6)
    class_weights        = class_weights / class_weights.sum() * num_classes
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    # ── DATALOADERS ───────────────────────────────────────────────────────
    # num_workers=2 — safe for Windows
    # pin_memory=True — faster GPU transfer
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )

    # ── MODEL ─────────────────────────────────────────────────────────────
    model = models.efficientnet_b4(
        weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
    )
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )

    # ── RESUME FROM CHECKPOINT ───────────────────────────────────────────
    if resume and os.path.exists(out_path):
        state = torch.load(out_path, map_location="cpu")
        try:
            model.load_state_dict(state)
            print(f"\n  ▶  RESUMING from: {out_path}")
            print(f"     Continuing for {epochs} more epochs.")
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}")
            print("   Starting from ImageNet weights.")
    elif resume:
        print(f"⚠️  --resume passed but no checkpoint at {out_path}. Starting fresh.")

    model = model.to(device)

    # ── DIFFERENTIAL LEARNING RATES ──────────────────────────────────────
    # Backbone: lr x 0.1 — gentle fine-tuning
    # Classifier head: full lr — learns from scratch
    backbone_params   = [p for n, p in model.named_parameters()
                         if "classifier" not in n]
    classifier_params = [p for n, p in model.named_parameters()
                         if "classifier" in n]

    optimizer = optim.AdamW([
        {"params": backbone_params,   "lr": lr * 0.1, "weight_decay": 1e-4},
        {"params": classifier_params, "lr": lr,        "weight_decay": 1e-3},
    ])

    # ── COSINE ANNEALING WITH WARM RESTARTS ───────────────────────────────
    # T_0=10: first restart at epoch 10
    # T_mult=2: each subsequent cycle is 2x longer (10 → 20 → 40)
    # With 80 epochs: restarts at epoch 10, 30, 70
    # Three restarts = three chances to escape local minima
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # ── LOSS ──────────────────────────────────────────────────────────────
    # label_smoothing=0.05 — lighter than 0.1, better for 30 classes
    criterion    = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=0.05
    )
    mixup_cutmix = MixUpCutMix(
        num_classes=num_classes,
        alpha=0.2,          # less aggressive than 0.4
        mixup_prob=0.4,     # 40% mixup
        cutmix_prob=0.4     # 40% cutmix, 20% no augmentation
    )
    scaler     = GradScaler(enabled=use_amp)
    early_stop = EarlyStopping(patience=10, min_delta=0.0005)
    top_k_ckpt = TopKCheckpoints(out_path, k=3)
    best_acc   = 0.0

    # ── HISTORY ───────────────────────────────────────────────────────────
    history_path = os.path.join(out_dir, "training_history.json")
    csv_path     = os.path.join(out_dir, "training_log.csv")

    if resume and os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        best_acc = max(history["val_acc"]) if history["val_acc"] else 0.0
        print(f"  📈 History loaded — {len(history['train_loss'])} previous epochs")
        print(f"     Previous best: {best_acc:.4f}")
    else:
        history = {"train_loss": [], "val_acc": [], "lr": []}

    if not resume or not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_acc", "f1_weighted", "lr"]
            )

    print(f"\n  Epochs         : {epochs}  (max)")
    print(f"  Batch size     : {batch_size}")
    print(f"  Head LR        : {lr}  |  Backbone LR: {lr * 0.1}")
    print(f"  Label smoothing: 0.05")
    print(f"  MixUp alpha    : 0.2  (40% mixup / 40% cutmix / 20% none)")
    print(f"  Early stopping : patience=10, min_delta=0.0005")
    print(f"  Warm restarts  : epochs 10, 30, 70")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        global_epoch = len(history["train_loss"]) + 1

        # Step scheduler
        scheduler.step(epoch)

        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, (inputs, labels) in enumerate(tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{epochs}",
                ncols=80)):

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            inputs_aug, soft_labels = mixup_cutmix(inputs, labels, device)

            with autocast(enabled=use_amp):
                outputs   = model(inputs_aug)
                log_probs = torch.log_softmax(outputs, dim=1)
                loss      = -(soft_labels * log_probs).sum(dim=1).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_ds)
        results    = evaluate_model(model, val_loader, device, num_classes)
        val_acc    = results[0]
        f1_w       = results[9]
        current_lr = optimizer.param_groups[1]["lr"]

        history["train_loss"].append(epoch_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        # Save history every epoch — survives interruptions
        with open(history_path, "w") as f:
            json.dump(history, f)

        # CSV log
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                global_epoch,
                f"{epoch_loss:.4f}",
                f"{val_acc:.4f}",
                f"{f1_w:.4f}",
                f"{current_lr:.2e}"
            ])

        print(f"\nEpoch {epoch+1:3d}/{epochs}"
              f"  loss={epoch_loss:.4f}"
              f"  val_acc={val_acc:.4f}"
              f"  f1_w={f1_w:.4f}"
              f"  lr={current_lr:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_path)
            print(f"  ✅ New best: {val_acc:.4f}")

        top_k_ckpt.save(model, val_acc, epoch)

        early_stop(val_acc)
        if early_stop.stop:
            print(f"\n⏹  Early stopping at epoch {epoch+1}")
            break

    print(f"\n🏆 Best Validation Accuracy: {best_acc:.4f}  ({best_acc*100:.2f}%)")

    # ══════════════════════════════════════════
    # FINAL EVALUATION WITH 7-VIEW TTA
    # ══════════════════════════════════════════
    print("\nRunning final evaluation with 7-view TTA...")
    model.load_state_dict(torch.load(out_path, map_location=device))

    (accuracy, all_labels, all_preds, all_probs,
     prec_macro, rec_macro,
     prec_weighted, rec_weighted,
     f1_macro, f1_weighted) = evaluate_model(
        model, val_loader, device, num_classes, use_tta=True
    )

    print(f"\n{'='*60}")
    print("  FINAL METRICS  (7-view TTA)")
    print(f"{'='*60}")
    print(f"  Accuracy              : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  Precision  (Macro)    : {prec_macro:.4f}")
    print(f"  Precision  (Weighted) : {prec_weighted:.4f}")
    print(f"  Recall     (Macro)    : {rec_macro:.4f}")
    print(f"  Recall     (Weighted) : {rec_weighted:.4f}")
    print(f"  F1 Score   (Macro)    : {f1_macro:.4f}")
    print(f"  F1 Score   (Weighted) : {f1_weighted:.4f}")
    print(f"{'='*60}\n")
    print("Per-class Classification Report:\n")
    print(classification_report(
        all_labels, all_preds,
        target_names=train_ds.classes,
        zero_division=0
    ))

    epochs_ran   = len(history["train_loss"])
    epoch_labels = list(range(1, epochs_ran + 1))

    # ── PLOT 1: TRAINING LOSS CURVE ───────────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_labels, history["train_loss"],
             color="#b8720a", linewidth=2.5, marker="o", markersize=4, label="Train Loss")
    # Mark warm restart points
    for restart in [10, 30, 70]:
        if restart < epochs_ran:
            plt.axvline(x=restart, color="gray", linestyle=":", linewidth=1,
                        label=f"Restart (epoch {restart})" if restart == 10 else "")
    plt.title("Training Loss Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_loss_curve.png"), dpi=150)
    plt.close()
    print("✅ train_loss_curve.png")

    # ── PLOT 2: VALIDATION ACCURACY CURVE ────────────────────────────────
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_labels, history["val_acc"],
             color="#3a6b30", linewidth=2.5, marker="o", markersize=4, label="Val Accuracy")
    plt.axhline(best_acc, color="#b8720a", linestyle="--", linewidth=1.5,
                label=f"Best: {best_acc:.4f} ({best_acc*100:.2f}%)")
    for restart in [10, 30, 70]:
        if restart < epochs_ran:
            plt.axvline(x=restart, color="gray", linestyle=":", linewidth=1)
    plt.title("Validation Accuracy Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.ylim(0, 1.05); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "val_accuracy_curve.png"), dpi=150)
    plt.close()
    print("✅ val_accuracy_curve.png")

    # ── PLOT 3: COMBINED LOSS + ACCURACY ─────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Training Overview", fontsize=15, fontweight="bold")
    ax1.plot(epoch_labels, history["train_loss"],
             color="#b8720a", linewidth=2.5, marker="o", markersize=4)
    ax1.set_title("Training Loss"); ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss"); ax1.grid(alpha=0.3)
    ax2.plot(epoch_labels, history["val_acc"],
             color="#3a6b30", linewidth=2.5, marker="o", markersize=4)
    ax2.axhline(best_acc, color="#b8720a", linestyle="--",
                linewidth=1.5, label=f"Best: {best_acc:.4f}")
    ax2.set_title("Validation Accuracy"); ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy"); ax2.set_ylim(0, 1.05)
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()
    print("✅ training_curves.png")

    # ── PLOT 4: LR CURVE ─────────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(epoch_labels, history["lr"], color="#9b59b6", linewidth=2)
    plt.title("Learning Rate Schedule (Cosine + Warm Restarts)",
              fontsize=14, fontweight="bold")
    plt.xlabel("Epoch"); plt.ylabel("LR (classifier head)")
    plt.yscale("log"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lr_curve.png"), dpi=150)
    plt.close()
    print("✅ lr_curve.png")

    # ── PLOT 5: PER-CLASS METRICS ─────────────────────────────────────────
    per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds,
        labels=list(range(num_classes)), zero_division=0
    )
    x            = np.arange(num_classes)
    width        = 0.28
    labels_short = [c[:12] for c in train_ds.classes]

    fig, ax = plt.subplots(figsize=(max(14, num_classes * 0.9), 7))
    bars1 = ax.bar(x - width, per_class_prec, width,
                   label="Precision", color="#b8720a", alpha=0.85)
    bars2 = ax.bar(x,          per_class_rec,  width,
                   label="Recall",    color="#3a6b30", alpha=0.85)
    bars3 = ax.bar(x + width,  per_class_f1,   width,
                   label="F1 Score",  color="#4a7cb4", alpha=0.85)
    ax.set_title("Per-Class Precision / Recall / F1",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Breed Class"); ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_short, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.15); ax.legend(fontsize=10)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    for bar in [*bars1, *bars2, *bars3]:
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom",
                    fontsize=6, rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_class_metrics.png"), dpi=150)
    plt.close()
    print("✅ per_class_metrics.png")

    # ── PLOT 6: OVERALL METRICS SUMMARY ──────────────────────────────────
    metrics_names  = ["Accuracy",
                      "Precision\n(Macro)", "Precision\n(Weighted)",
                      "Recall\n(Macro)",    "Recall\n(Weighted)",
                      "F1\n(Macro)",        "F1\n(Weighted)"]
    metrics_values = [accuracy,
                      prec_macro, prec_weighted,
                      rec_macro,  rec_weighted,
                      f1_macro,   f1_weighted]
    colors = ["#b8720a","#3a6b30","#4a8a40",
              "#4a7cb4","#6a9cc4","#9b59b6","#b07ec0"]
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(metrics_names, metrics_values, color=colors,
                  width=0.55, alpha=0.88, edgecolor="white", linewidth=1.2)
    ax.set_title("Overall Model Performance Summary",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.15)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, metrics_values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "metrics_summary.png"), dpi=150)
    plt.close()
    print("✅ metrics_summary.png")

    # ── PLOT 7: CONFUSION MATRIX ──────────────────────────────────────────
    cm       = confusion_matrix(all_labels, all_preds)
    fig_size = max(12, num_classes)
    plt.figure(figsize=(fig_size, fig_size - 2))
    sns.heatmap(cm, annot=(num_classes <= 25), fmt="d", cmap="YlOrBr",
                xticklabels=train_ds.classes,
                yticklabels=train_ds.classes, linewidths=0.3)
    plt.title("Confusion Matrix (7-view TTA)",
              fontsize=14, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label",      fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,          fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("✅ confusion_matrix.png")

    # ── PLOT 8: NORMALIZED CONFUSION MATRIX ──────────────────────────────
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-6)
    plt.figure(figsize=(fig_size, fig_size - 2))
    sns.heatmap(cm_norm, annot=(num_classes <= 25), fmt=".2f",
                cmap="YlOrBr", vmin=0, vmax=1,
                xticklabels=train_ds.classes,
                yticklabels=train_ds.classes, linewidths=0.3)
    plt.title("Normalized Confusion Matrix — Row = Recall per Class",
              fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label",      fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0,          fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_normalized.png"), dpi=150)
    plt.close()
    print("✅ confusion_matrix_normalized.png")

    # ── PLOT 9: ROC CURVE ─────────────────────────────────────────────────
    y_test_bin = label_binarize(all_labels, classes=list(range(num_classes)))
    plt.figure(figsize=(11, 9))
    cmap     = plt.cm.get_cmap("tab20", num_classes)
    mean_auc = 0.0
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc      = auc(fpr, tpr)
        mean_auc    += roc_auc
        plt.plot(fpr, tpr, linewidth=1.3, color=cmap(i),
                 label=f"{train_ds.classes[i]} (AUC={roc_auc:.2f})")
    mean_auc /= num_classes
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.50)")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate",  fontsize=12)
    plt.title(f"Multi-class ROC Curve (7-view TTA)  |  Mean AUC = {mean_auc:.3f}",
              fontsize=14, fontweight="bold")
    plt.legend(fontsize=6, ncol=2, loc="lower right")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close()
    print("✅ roc_curve.png")

    print(f"\n{'='*60}")
    print(f"  ALL FILES SAVED TO: {out_dir}")
    print(f"{'='*60}")
    print("  train_loss_curve.png")
    print("  val_accuracy_curve.png")
    print("  training_curves.png")
    print("  lr_curve.png")
    print("  per_class_metrics.png")
    print("  metrics_summary.png")
    print("  confusion_matrix.png")
    print("  confusion_matrix_normalized.png")
    print("  roc_curve.png")
    print("  training_log.csv")
    print("  training_history.json")
    print(f"{'='*60}\n")


# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="EfficientNet-B4 Cattle Classifier — RTX 2050 Optimized"
    )
    p.add_argument("--data",   default="dataset/split")
    p.add_argument("--out",    default="models/efficientnet_b4.pth")
    p.add_argument("--epochs", type=int,   default=50)
    p.add_argument("--batch",  type=int,   default=8)
    p.add_argument("--lr",     type=float, default=3e-4)
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing checkpoint")
    args = p.parse_args()

    train_model(
        args.data, args.out,
        args.epochs, args.batch,
        args.lr, args.resume
    )