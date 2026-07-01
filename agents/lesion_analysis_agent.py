"""
Agent 1: Lesion Analysis Agent (AM-RAG Section 5.3.2)

Responsibilities: Detect retinal lesions, quantify lesion burden, identify
disease severity indicators.
Input: Retinal Image.
Output: Structured retinal pathology profile P = {MA, HE, EX, CWS, NV, VT}
        plus the DR severity grade and Grad-CAM attention map.

This agent wraps the trained CNNViTHybridLesionDetector (models/lesion_detector.py)
-- it is not itself an LLM call, it's the perception layer that feeds
everything downstream.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.lesion_detector import (
    CNNViTHybridLesionDetector, LesionDetectorConfig,
    LESION_NAMES, SEVERITY_LABELS, load_detector,
)
from training.dataset import EVAL_TRANSFORMS


class LesionAnalysisAgent:
        def __init__(self, checkpoint_path: str | None = None, model=None, device = "cpu"):
                self.device = device
                if model is not None:
                        if isinstance(model, dict):
                                if 'model_state_dict' in model:
                                        self.model = self.initialize_model_architecture() 
                                        self.model.load_state_dict(model['model_state_dict'])
                                else:
                                        self.model = self.initialize_model_architecture()
                                        self.model.load_state_dict(model)
                        else:
                                self.model = model
                self.model.to(self.device)
                self.model.eval()

    def analyze(self, image: Image.Image) -> dict:
        """Run lesion detection on a single PIL fundus image."""
        img_np = np.array(image.convert("RGB"))
        transformed = EVAL_TRANSFORMS(image=img_np)
        tensor = transformed["image"].unsqueeze(0).to(self.device)

        # keep gradients w.r.t. the cached conv activations for Grad-CAM
        tensor.requires_grad_(False)
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            out = self.model(tensor)
            severity_probs = F.softmax(out["severity_logits"], dim=1)[0]
            severity_idx = int(severity_probs.argmax().item())

            # Grad-CAM: backprop the predicted class logit to the cached
            # last-conv activations captured by the forward hook.
            target_logit = out["severity_logits"][0, severity_idx]
            gradcam_map = self._compute_gradcam(target_logit)

        lesion_burden = out["lesion_burden"][0].detach().cpu().numpy()

        return {
            "severity_grade": severity_idx,
            "severity_label": SEVERITY_LABELS[severity_idx],
            "severity_confidence": float(severity_probs[severity_idx].item()),
            "severity_probs": {SEVERITY_LABELS[i]: float(p) for i, p in enumerate(severity_probs.tolist())},
            "lesion_burden": {name: float(v) for name, v in zip(LESION_NAMES, lesion_burden)},
            "gradcam_map": gradcam_map,  # (H, W) numpy array, normalized 0-1, or None
            "fused_features": out["fused_features"][0].detach().cpu().numpy(),
            "is_trained_checkpoint": self.is_trained,
        }

    def _compute_gradcam(self, target_logit: torch.Tensor):
        """Grad-CAM over the CNN branch's last conv layer (Explainability
        Agent Sec 5.3.2 -- 'generate lesion attention maps')."""
        activations = self.model._last_conv_features
        if activations is None:
            return None

        grads = torch.autograd.grad(
            target_logit, activations, retain_graph=True, create_graph=False,
            allow_unused=True,
        )[0]
        if grads is None:
            return None

        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam
