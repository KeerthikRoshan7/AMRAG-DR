import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.lesion_detector import (
    CNNViTHybridLesionDetector, LesionDetectorConfig,
    LESION_NAMES, SEVERITY_LABELS, load_detector, load_state_dict_compat,
)
from training.dataset import EVAL_TRANSFORMS

class LesionAnalysisAgent:
    def __init__(self, checkpoint_path: str | None = None, model=None, device="cpu"):
        self.device = device
        self.is_trained = checkpoint_path is not None
        
        if model is not None:
            if isinstance(model, dict):
                # Handle state_dict unpacking
                self.model = self.initialize_model_architecture()
                state_dict = model.get('model_state_dict', model)
                load_state_dict_compat(self.model, state_dict)
            else:
                self.model = model
        else:
            # Fallback to demo mode if no model provided
            self.model = self.initialize_model_architecture()
            
        self.model.to(self.device)
        self.model.eval()

    def initialize_model_architecture(self):
        # Ensure this matches your actual model init logic
        config = LesionDetectorConfig()
        return CNNViTHybridLesionDetector(config)

    def analyze(self, image: Image.Image) -> dict:
        img_np = np.array(image.convert("RGB"))
        transformed = EVAL_TRANSFORMS(image=img_np)
        tensor = transformed["image"].unsqueeze(0).to(self.device)

        # Grad-CAM needs gradients enabled, even if the model is in eval mode
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            out = self.model(tensor)
            severity_probs = F.softmax(out["severity_logits"], dim=1)[0]
            severity_idx = int(severity_probs.argmax().item())

            # Grad-CAM: backprop target logit to the cached conv activations
            target_logit = out["severity_logits"][0, severity_idx]
            gradcam_map = self._compute_gradcam(target_logit)

            # Per-lesion-type CAMs: the severity-only CAM above smears all
            # lesion types into one heatmap. Backpropping from each of the
            # 6 lesion_burden outputs separately gives a distinct map per
            # lesion type (MA, HE, EX, CWS, NV, VT), so the UI can show
            # "where are the exudates" vs "where are the hemorrhages"
            # instead of one blended blob.
            lesion_gradcam_maps = self._compute_lesion_gradcams(out["lesion_burden"][0])

        lesion_burden = out["lesion_burden"][0].detach().cpu().numpy()

        return {
            "severity_grade": severity_idx,
            "severity_label": SEVERITY_LABELS[severity_idx],
            "severity_confidence": float(severity_probs[severity_idx].item()),
            "severity_probs": {SEVERITY_LABELS[i]: float(p) for i, p in enumerate(severity_probs.tolist())},
            "lesion_burden": {name: float(v) for name, v in zip(LESION_NAMES, lesion_burden)},
            "gradcam_map": gradcam_map,
            "lesion_gradcam_maps": lesion_gradcam_maps,
            "fused_features": out["fused_features"][0].detach().cpu().numpy(),
            "is_trained_checkpoint": self.is_trained,
        }

    def _compute_gradcam(self, target_logit: torch.Tensor):
        """Grad-CAM++ over the CNN branch's hooked conv layer (see
        models/lesion_detector.py::_register_gradcam_hook for layer choice).

        Uses the Grad-CAM++ alpha-weighting (Chattopadhay et al., 2018)
        instead of plain gradient-mean weighting: it down-weights pixels
        with large-but-inconsistent gradients, which produces tighter
        localization when multiple lesion instances are present in one
        image -- the common case here (several microaneurysms/hemorrhages
        at once), vs. vanilla Grad-CAM's tendency to average them into one
        diffuse region.
        """
        activations = getattr(self.model, "_last_conv_features", None)
        if activations is None:
            return None

        print(f"[GradCAM debug] activations min={activations.min().item():.4f}, "
              f"max={activations.max().item():.4f}")   # <-- temporary, remove after checking

        grads = torch.autograd.grad(
            target_logit, activations, retain_graph=True, create_graph=False,
            allow_unused=True,
        )[0]

        if grads is None:
            return None

        cam = self._gradcam_plusplus_from_grads(activations, grads)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()

        # Improved normalization: use percentile to handle outliers
        if cam.max() > 0:
            v_min, v_max = np.percentile(cam, [2, 98])
            cam = np.clip((cam - v_min) / (v_max - v_min + 1e-8), 0, 1)
        return cam

    def _compute_lesion_gradcams(self, lesion_burden_vec: torch.Tensor) -> dict:
        """One Grad-CAM++ map per lesion type, backpropped independently
        from each lesion_burden output."""
        activations = getattr(self.model, "_last_conv_features", None)
        if activations is None:
            return {}

        maps = {}
        for i, name in enumerate(LESION_NAMES):
            grads = torch.autograd.grad(
                lesion_burden_vec[i], activations, retain_graph=True,
                create_graph=False, allow_unused=True,
            )[0]
            if grads is None:
                continue
            cam = self._gradcam_plusplus_from_grads(activations, grads)
            cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()
            if cam.max() > 0:
                v_min, v_max = np.percentile(cam, [2, 98])
                cam = np.clip((cam - v_min) / (v_max - v_min + 1e-8), 0, 1)
            maps[name] = cam
        return maps

    @staticmethod
    def _gradcam_plusplus_from_grads(activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        grads_power_2 = gradients ** 2
        grads_power_3 = grads_power_2 * gradients

        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        alpha = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)
        alpha = torch.where(gradients != 0, alpha, torch.zeros_like(alpha))

        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        return F.relu(cam)
