"""
CNN + ViT Hybrid Lesion Detector for Diabetic Retinopathy.

Architecture (per AM-RAG Section 5.2, Visual Processing Pipeline):
    Fundus Image -> Preprocessing -> Lesion Detection (CNN + ViT Hybrid)
                  -> Visual Feature Encoder -> Multimodal Feature Fusion

This module implements the "Lesion Detection Module (CNN + ViT Hybrid)" block.
It is a multi-task network with two heads sharing one backbone:
  1. A DR severity grading head (5-class: No DR, Mild, Moderate, Severe, PDR)
  2. A lesion regression head estimating per-lesion burden:
        MA  - Microaneurysm count (normalized)
        HE  - Hemorrhage count (normalized)
        EX  - Hard exudate burden
        CWS - Cotton wool spot count
        NV  - Neovascularization score
        VT  - Vessel tortuosity score

Backbone: timm CNN (EfficientNet-B3 by default) for local texture/lesion
features, fused with a ViT-Small branch for global structural context
(vessel arcades, optic disc / macula layout), matching the "CNN + ViT
Hybrid" described in the paper. Grad-CAM is computed on the CNN branch's
last conv layer for the Explainability Agent.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import timm


@dataclass
class LesionDetectorConfig:
    cnn_backbone: str = "efficientnet_b3"
    vit_backbone: str = "vit_small_patch16_224"
    num_severity_classes: int = 5
    num_lesion_features: int = 6  # MA, HE, EX, CWS, NV, VT
    fusion_dim: int = 512
    dropout: float = 0.3
    pretrained: bool = True


class CNNViTHybridLesionDetector(nn.Module):
    """Multi-task DR severity + lesion-burden regression network."""

    def __init__(self, config: LesionDetectorConfig = LesionDetectorConfig()):
        super().__init__()
        self.config = config

        # --- CNN branch: local lesion texture (microaneurysms, exudates) ---
        self.cnn = timm.create_model(
            config.cnn_backbone,
            pretrained=config.pretrained,
            num_classes=0,  # remove classifier head, return pooled features
        )
        cnn_feat_dim = self.cnn.num_features

        # --- ViT branch: global retinal structure (vessels, disc, macula) ---
        self.vit = timm.create_model(
            config.vit_backbone,
            pretrained=config.pretrained,
            num_classes=0,
        )
        vit_feat_dim = self.vit.num_features

        # --- Fusion layer ---
        self.fusion = nn.Sequential(
            nn.Linear(cnn_feat_dim + vit_feat_dim, config.fusion_dim),
            nn.BatchNorm1d(config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

        # --- Task heads ---
        self.severity_head = nn.Linear(config.fusion_dim, config.num_severity_classes)
        self.lesion_head = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim // 2),
            nn.GELU(),
            nn.Linear(config.fusion_dim // 2, config.num_lesion_features),
            nn.Softplus(),  # lesion counts/burdens are non-negative
        )

        # Cache last CNN conv feature map for Grad-CAM (Explainability Agent)
        self._last_conv_features = None
        self._register_gradcam_hook(target_grid=14)

    def _register_gradcam_hook(self, target_grid: int = 14, dummy_input_size: int = 224):
        """Hook a CNN conv layer for Grad-CAM.

        The naive "literal last Conv2d" (previously used) resolves to
        EfficientNet-B3's conv_head, whose output grid is only ~7x7 for a
        224px input -- upsampled 32x to the display size, which is why
        small lesions (microaneurysms, small hemorrhages) show up as one
        diffuse blob rather than distinct regions.

        Instead, run a dummy forward pass, record every Conv2d layer's
        output spatial resolution, and pick the layer whose grid size is
        closest to `target_grid` (default 14x14 -- one stage earlier than
        the head, still deep enough to be semantically meaningful, but
        4x the spatial resolution of the final 7x7 grid). This is robust
        to backbone changes since it doesn't hardcode a module path.
        """
        candidates = []

        def make_probe(name, mod):
            def probe(module, inp, out):
                if out.dim() == 4:
                    candidates.append((name, mod, out.shape[-1]))
            return probe

        probe_handles = []
        for name, m in self.cnn.named_modules():
            if isinstance(m, nn.Conv2d):
                probe_handles.append(m.register_forward_hook(make_probe(name, m)))

        was_training = self.cnn.training
        self.cnn.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, dummy_input_size, dummy_input_size)
            self.cnn(dummy)
        self.cnn.train(was_training)

        for h in probe_handles:
            h.remove()

        target_layer = None
        if candidates:
            target_layer = min(candidates, key=lambda c: abs(c[2] - target_grid))[1]
        else:
            for m in self.cnn.modules():
                if isinstance(m, nn.Conv2d):
                    target_layer = m

        def hook(module, inp, out):
            self._last_conv_features = out

        if target_layer is not None:
            target_layer.register_forward_hook(hook)
            self._gradcam_layer = target_layer
        else:
            self._gradcam_layer = None

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) preprocessed fundus image batch.
        Returns:
            dict with:
              severity_logits: (B, num_severity_classes)
              lesion_burden:   (B, num_lesion_features) -> [MA, HE, EX, CWS, NV, VT]
              fused_features:  (B, fusion_dim) -> feeds Multimodal Feature Fusion Module
        """
        cnn_feats = self.cnn(x)
        vit_feats = self.vit(x)
        fused = self.fusion(torch.cat([cnn_feats, vit_feats], dim=1))

        return {
            "severity_logits": self.severity_head(fused),
            "lesion_burden": self.lesion_head(fused),
            "fused_features": fused,
        }


LESION_NAMES = ["microaneurysms", "hemorrhages", "hard_exudates",
                "cotton_wool_spots", "neovascularization", "vessel_tortuosity"]

SEVERITY_LABELS = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "PDR",
]


def load_detector(checkpoint_path: str | None, device: str = "cpu",
                   config: LesionDetectorConfig = LesionDetectorConfig()):
    """Load the detector. If checkpoint_path is None or missing, returns an
    ImageNet-pretrained-only model clearly flagged as UNTRAINED for DR
    grading -- useful to exercise the rest of the pipeline (RAG + agents)
    before Kaggle training finishes."""
    model = CNNViTHybridLesionDetector(config)
    is_trained = False
    if checkpoint_path:
        try:
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
            is_trained = True
        except FileNotFoundError:
            pass
    model.to(device)
    model.eval()
    return model, is_trained
