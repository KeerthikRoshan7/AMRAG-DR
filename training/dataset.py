"""
Dataset loader for DR severity + lesion-burden training on Kaggle.

Expected input: an EyePACS/APTOS/Kermany-style CSV with columns:
    image_path, severity_label, ma, he, ex, cws, nv, vt

`ma..vt` are optional. If your dataset only has severity labels (most
public DR datasets do, e.g. APTOS 2019, EyePACS), leave those columns
absent -- the training loop switches to severity-only supervision and the
lesion head trains as a weakly-supervised auxiliary task using pseudo-
labels derived from Grad-CAM++ activation statistics (see
`weak_lesion_pseudo_labels` below). This mirrors the "modality-complementary
fusion" scoping decision already used for EMRA-DR: real paired lesion-level
annotations are rarely available, so the framework is explicit about which
supervision is real vs. weak/proxy.
"""

import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMG_SIZE = 224

TRAIN_TRANSFORMS = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),  # per Sec 5.6
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=25, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.15),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

EVAL_TRANSFORMS = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

LESION_COLUMNS = ["ma", "he", "ex", "cws", "nv", "vt"]


class DRLesionDataset(Dataset):
    """
    IMPORTANT augmentation-bleed note (carried over from the EMRA-DR V2
    bugfix): apply `random_split` on the DataFrame/index level FIRST, then
    build separate Dataset instances (one with TRAIN_TRANSFORMS, one with
    EVAL_TRANSFORMS) from each split. Never wrap a single dataset object
    with one transform and call random_split on it afterward -- that leaks
    augmented train-time transforms into the validation split.
    """

    def __init__(self, dataframe: pd.DataFrame, image_root: str, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform or EVAL_TRANSFORMS
        self.has_lesion_labels = all(c in self.df.columns for c in LESION_COLUMNS)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_root, row["image_path"])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=image)
        image = augmented["image"]

        severity = int(row["severity_label"])

        if self.has_lesion_labels:
            lesion = row[LESION_COLUMNS].values.astype(np.float32)
        else:
            # weak proxy target: monotonically increasing with severity,
            # refined later by Grad-CAM-derived pseudo-labels in train loop
            lesion = weak_lesion_pseudo_labels(severity)

        return {
            "image": image,
            "severity": torch.tensor(severity, dtype=torch.long),
            "lesion_burden": torch.tensor(lesion, dtype=torch.float32),
            "has_real_lesion_labels": self.has_lesion_labels,
        }


def weak_lesion_pseudo_labels(severity: int) -> np.ndarray:
    """
    Coarse proxy lesion-burden vector when per-lesion ground truth isn't
    available. Values loosely follow the ICDR clinical progression pattern
    (e.g. NV only appears at PDR; MA/HE increase steadily then plateau at
    severe NPDR before PDR -- consistent with published lesion-frequency
    studies). This is a WEAK SUPERVISION SIGNAL, not ground truth -- flag it
    as such in any thesis writeup / ablation table.
    """
    table = {
        0: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        1: [0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        2: [0.6, 0.4, 0.3, 0.2, 0.0, 0.1],
        3: [0.8, 0.9, 0.6, 0.5, 0.0, 0.4],
        4: [0.5, 0.7, 0.5, 0.3, 1.0, 0.7],
    }
    return np.array(table.get(severity, [0.0] * 6), dtype=np.float32)


def build_train_val_datasets(csv_path: str, image_root: str, val_split: float = 0.15,
                              seed: int = 42):
    """Split at the DataFrame level (fixes the augmentation-bleed bug),
    then attach different transforms per split."""
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_size = int(len(df) * val_split)
    val_df = df.iloc[:val_size]
    train_df = df.iloc[val_size:]

    train_ds = DRLesionDataset(train_df, image_root, transform=TRAIN_TRANSFORMS)
    val_ds = DRLesionDataset(val_df, image_root, transform=EVAL_TRANSFORMS)
    return train_ds, val_ds
