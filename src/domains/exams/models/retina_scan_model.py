import os
import sys
from pathlib import Path
from typing import Dict, Any
from io import BytesIO

from .vit import models_vit as models
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch import Tensor

CURRENT_FILE = Path(__file__).resolve()
SRC_ROOT = CURRENT_FILE.parents[3]
CHECKPOINT_DIR = SRC_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "modelo.pth"



CLASS_NAMES = {
    0: "normal",
    1: "abnormal",
}


_model_instance = None


class RetinaScanModel:
    def __init__(
        self,
        checkpoint_path: Path,
        model_name: str = 'RETFound_mae',
        input_size: int = 224,
        num_classes: int = 2,
        threshold: float = 0.5,
    ):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.input_size = input_size
        self.num_classes = num_classes
        self.threshold = threshold
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

    def _prepare_image(self, image_bytes: bytes) -> Tensor:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        x = self._prepare_image(image_bytes)
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

        return {
            "predicted_class": pred_idx,
            "predicted_label": CLASS_NAMES[pred_idx],
            "confidence": confidence,
            "probabilities": probabilities,
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
        )
    return _model_instance
