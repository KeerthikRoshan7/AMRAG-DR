"""
Training script for the CNN+ViT Hybrid Lesion Detector -- run this on
Kaggle (free T4/P100 GPU), matching the zero-cost deployment stack already
used for EMRA-DR: Kaggle for training, Groq for LLM inference, HF Spaces
for the demo.

Usage on Kaggle:
    1. Upload/attach a DR dataset as a Kaggle Dataset (e.g. APTOS 2019
       Blindness Detection, or EyePACS). You need a CSV with at minimum
       `image_path,severity_label` columns (0-4). See dataset.py docstring
       for the optional per-lesion columns.
    2. Add this repo (or just these two files: dataset.py + this script +
       ../models/lesion_detector.py) as Kaggle Utility Scripts, or paste
       into a notebook cell.
    3. Set CSV_PATH / IMAGE_ROOT below to the Kaggle input paths.
    4. Run all cells. Checkpoints save to /kaggle/working/checkpoints/.
    5. Download best_model.pt and drop it into
       amrag-dr/checkpoints/lesion_detector.pt for local inference.

This is a genuine, runnable multi-task training loop -- it is NOT executed
in this sandbox (no GPU / no dataset access here), so run it on Kaggle
before wiring the real checkpoint into the local app.
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lesion_detector import CNNViTHybridLesionDetector, LesionDetectorConfig
from training.dataset import build_train_val_datasets

# ---------------------------------------------------------------------
# Config -- edit these for your Kaggle environment
# ---------------------------------------------------------------------
CSV_PATH = "/kaggle/input/datasets/mariaherrerot/aptos2019/train_1.csv"
IMAGE_ROOT = "/kaggle/input/datasets/mariaherrerot/aptos2019/train_images/train_images"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
BATCH_SIZE = 32
EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 6
LESION_LOSS_WEIGHT = 0.4  # multi-task loss balance: severity vs lesion burden
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    train_ds, val_ds = build_train_val_datasets(CSV_PATH, IMAGE_ROOT)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = CNNViTHybridLesionDetector(LesionDetectorConfig()).to(DEVICE)

    severity_criterion = nn.CrossEntropyLoss()
    lesion_criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_val_qwk = -1.0
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(DEVICE, non_blocking=True)
            severity = batch["severity"].to(DEVICE, non_blocking=True)
            lesion_burden = batch["lesion_burden"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                out = model(images)
                loss_sev = severity_criterion(out["severity_logits"], severity)
                loss_lesion = lesion_criterion(out["lesion_burden"], lesion_burden)
                loss = loss_sev + LESION_LOSS_WEIGHT * loss_lesion

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        scheduler.step()

        # --- validation ---
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE, non_blocking=True)
                severity = batch["severity"].to(DEVICE, non_blocking=True)
                out = model(images)
                preds = out["severity_logits"].argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(severity.cpu().numpy())

        val_acc = accuracy_score(all_targets, all_preds)
        val_qwk = cohen_kappa_score(all_targets, all_preds, weights="quadratic")

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={running_loss/len(train_loader):.4f} "
              f"| val_acc={val_acc:.4f} | val_QWK={val_qwk:.4f} | {elapsed:.1f}s")

        # Early stopping on validation Quadratic Weighted Kappa (standard
        # metric for ordinal DR severity grading -- more informative than
        # raw accuracy since severity classes are ordered, not categorical).
        if val_qwk > best_val_qwk:
            best_val_qwk = val_qwk
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_qwk": val_qwk,
                "val_acc": val_acc,
            }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print(f"  -> saved new best checkpoint (QWK={val_qwk:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (no improvement for "
                      f"{EARLY_STOP_PATIENCE} epochs)")
                break

    print(f"Training complete. Best val QWK: {best_val_qwk:.4f}")
    print(f"Checkpoint: {os.path.join(CHECKPOINT_DIR, 'best_model.pt')}")


if __name__ == "__main__":
    train()
