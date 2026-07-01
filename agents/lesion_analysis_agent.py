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
    def __init__(self, checkpoint_path: str | None = None, model=None, device="cpu"):
        self.device = device
        self.is_trained = checkpoint_path is not None
        
        if model is not None:
            if isinstance(model, dict):
                # Handle state_dict unpacking
                self.model = self.initialize_model_architecture()
                state_dict = model.get('model_state_dict', model)
                self.model.load_state_dict(state_dict)
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

            # Grad-CAM: backprop target logit to the cached last-conv activations
            target_logit = out["severity_logits"][0, severity_idx]
            gradcam_map = self._compute_gradcam(target_logit)

        lesion_burden = out["lesion_burden"][0].detach().cpu().numpy()

        return {
            "severity_grade": severity_idx,
            "severity_label": SEVERITY_LABELS[severity_idx],
            "severity_confidence": float(severity_probs[severity_idx].item()),
            "severity_probs": {SEVERITY_LABELS[i]: float(p) for i, p in enumerate(severity_probs.tolist())},
            "lesion_burden": {name: float(v) for name, v in zip(LESION_NAMES, lesion_burden)},
            "gradcam_map": gradcam_map,
            "fused_features": out["fused_features"][0].detach().cpu().numpy(),
            "is_trained_checkpoint": self.is_trained,
        }

    def _compute_gradcam(self, target_logit: torch.Tensor):
        """Grad-CAM over the CNN branch's last conv layer."""
        # Ensure your model exposes _last_conv_features via a forward hook
        activations = getattr(self.model, "_last_conv_features", None)
        if activations is None:
            return None

        grads = torch.autograd.grad(
            target_logit, activations, retain_graph=True, create_graph=False,
            allow_unused=True,
        )[0]
        
        if grads is None:
            return None

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam
