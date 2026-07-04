import os
import sys
from pathlib import Path
from typing import Dict, Any
from io import BytesIO

from .vit import models_vit as models
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch import Tensor

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

CURRENT_FILE = Path(__file__).resolve()
SRC_ROOT = CURRENT_FILE.parents[3]
CHECKPOINT_DIR = SRC_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "modelo.pth"

CLASS_NAMES = {
    0: "normal",
    1: "abnormal",
}

_model_instance = None


def _vit_reshape_transform(tensor: Tensor, patch_size: int = 14, input_size: int = 224):
    """
    Converte a saída de um bloco ViT (batch, tokens, dim) em um mapa
    espacial (batch, dim, h, w), descartando tokens que não são patches
    (ex: CLS token e, se existirem, register tokens do DINOv2).
    """
    grid_size = input_size // patch_size  # ex: 224 // 14 = 16
    num_patches = grid_size * grid_size

    batch, num_tokens, dim = tensor.shape
    num_extra_tokens = num_tokens - num_patches

    if num_extra_tokens < 0:
        raise ValueError(
            f"Número de tokens ({num_tokens}) menor que o esperado de patches "
            f"({num_patches}). Verifique patch_size/input_size."
        )

    # remove tokens extras do início (CLS + eventuais register tokens)
    result = tensor[:, num_extra_tokens:, :]
    result = result.reshape(batch, grid_size, grid_size, dim)
    result = result.permute(0, 3, 1, 2)
    return result


class RetinaScanModel:
    def __init__(
        self,
        checkpoint_path: Path,
        model_name: str = 'RETFound_mae',
        input_size: int = 224,
        num_classes: int = 2,
        threshold: float = 0.5,
        patch_size: int = 16,
    ):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.input_size = input_size
        self.num_classes = num_classes
        self.threshold = threshold
        self.patch_size = patch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.model = self._load_model()
        self._gradcam = self._build_gradcam()

    def _load_model(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        checkpoint_args = checkpoint["args"]
        model = self._build_model(checkpoint_args)

        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)

        model.to(self.device)
        model.eval()

        return model

    def _build_model(self, checkpoint_args):
        model = models.__dict__[self.model_name](
            num_classes=self.num_classes,
            drop_path_rate=0.2,
            args=checkpoint_args
        )
        return model

    def _build_gradcam(self) -> GradCAM:
        # último bloco do transformer -> normalização antes da MLP final
        # ajuste o índice/atributo se sua arquitetura usar outro nome (ex: self.model.encoder.layers)
        target_layer = self.model.blocks[-1].norm1

        return GradCAM(
            model=self.model,
            target_layers=[target_layer],
            reshape_transform=lambda t: _vit_reshape_transform(
                t, patch_size=self.patch_size, input_size=self.input_size
            ),
        )

    def _prepare_image(self, image_bytes: bytes) -> Tensor:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def _denormalize_for_display(self, tensor: Tensor) -> np.ndarray:
        """Converte o tensor normalizado de volta pra imagem RGB float [0,1] HxWx3."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
        img = tensor * std + mean
        img = img.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        return img.astype(np.float32)

    def _generate_gradcam_png(self, x: Tensor, target_class: int) -> bytes:
        targets = [ClassifierOutputTarget(target_class)]

        # GradCAM precisa de gradiente habilitado
        grayscale_cam = self._gradcam(input_tensor=x, targets=targets)
        grayscale_cam = grayscale_cam[0]  # (H, W)

        rgb_img = self._denormalize_for_display(x)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        success, buffer = cv2.imencode(".png", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("Falha ao codificar Grad-CAM em PNG")

        return buffer.tobytes()

    def predict_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        x = self._prepare_image(image_bytes)

        with torch.no_grad():
            logits = self.model(x)

            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            if logits.ndim == 3:
                logits = logits[:, 0, :]
            elif logits.ndim == 1:
                logits = logits.unsqueeze(0)

            probs = F.softmax(logits, dim=-1).detach().cpu()
            abnormal_prob = probs[0, 1].item()
            pred_idx = 1 if abnormal_prob >= self.threshold else 0
            confidence = float(probs[0, pred_idx].item())

            probabilities = {
                CLASS_NAMES[i]: float(probs[0, i].item()) for i in range(self.num_classes)
            }

        # gradcam precisa rodar FORA do no_grad (faz backward internamente)
        gradcam_png = self._generate_gradcam_png(x, target_class=pred_idx)

        return {
            "predicted_class": pred_idx,
            "predicted_label": CLASS_NAMES[pred_idx],
            "confidence": confidence,
            "probabilities": probabilities,
            "gradcam_png": gradcam_png,
        }


def get_retina_scan_model() -> RetinaScanModel:
    """Singleton otimizado por worker"""
    global _model_instance
    if _model_instance is None:
        checkpoint_path = Path(SRC_ROOT) / "checkpoint" / "dinov2_ODIR_v2.0.0-best.pth"
        _model_instance = RetinaScanModel(
            checkpoint_path=checkpoint_path,
            model_name="RETFound_dinov2",
            input_size=224,
            num_classes=2,
            threshold=0.5,
            patch_size=14,
        )
    return _model_instance