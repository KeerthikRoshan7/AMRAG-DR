import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define constants
IMG_SIZE = 224
LESION_COLUMNS = ["ma", "he", "ex", "cws", "nv", "vt"]

# Transforms
TRAIN_TRANSFORMS = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=25, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

EVAL_TRANSFORMS = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class DRLesionDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_root: str, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform or EVAL_TRANSFORMS
        self.has_lesion_labels = all(c in self.df.columns for c in LESION_COLUMNS)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # APTOS 2019/EyePACS standard: 'id_code' + '.png'
        img_path = os.path.join(self.image_root, str(row["id_code"]) + ".png")
        image = cv2.imread(img_path)
        
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]

        # Use 'diagnosis' for the severity label
        severity = int(row["diagnosis"])

        if self.has_lesion_labels:
            lesion = row[LESION_COLUMNS].values.astype(np.float32)
        else:
            lesion = weak_lesion_pseudo_labels(severity)

        return {
            "image": image,
            "severity": torch.tensor(severity, dtype=torch.long),
            "lesion_burden": torch.tensor(lesion, dtype=torch.float32),
            "has_real_lesion_labels": self.has_lesion_labels,
        }

def weak_lesion_pseudo_labels(severity: int) -> np.ndarray:
    table = {
        0: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        1: [0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
        2: [0.6, 0.4, 0.3, 0.2, 0.0, 0.1],
        3: [0.8, 0.9, 0.6, 0.5, 0.0, 0.4],
        4: [0.5, 0.7, 0.5, 0.3, 1.0, 0.7],
    }
    return np.array(table.get(severity, [0.0] * 6), dtype=np.float32)

def build_train_val_datasets(csv_path: str, image_root: str, val_split: float = 0.15, seed: int = 42):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_size = int(len(df) * val_split)
    
    train_ds = DRLesionDataset(df.iloc[val_size:], image_root, transform=TRAIN_TRANSFORMS)
    val_ds = DRLesionDataset(df.iloc[:val_size], image_root, transform=EVAL_TRANSFORMS)
    return train_ds, val_ds
