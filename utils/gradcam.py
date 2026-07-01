"""Overlay a Grad-CAM heatmap on the original fundus image for display."""

import numpy as np
import cv2
from PIL import Image


def overlay_gradcam(original_image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """
    Args:
        original_image: original PIL fundus image (any size)
        cam: (H, W) normalized [0,1] Grad-CAM array (typically 224x224)
        alpha: heatmap blend strength
    Returns:
        PIL Image with heatmap overlay, resized to original_image's size.
    """
    orig_np = np.array(original_image.convert("RGB"))
    h, w = orig_np.shape[:2]

    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap + (1 - alpha) * orig_np).astype(np.uint8)
    return Image.fromarray(overlay)
