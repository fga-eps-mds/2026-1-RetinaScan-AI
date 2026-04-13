import os
from io import BytesIO
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import models_vit as models
from timm import create_model


CLASS_NAMES = {
    0: 'normal',
    1: 'abnormal',
}

class RetinaScanModel:
    def __init__(
        self,
        checkpoint_path,
        model_name='RETFound_mae',
        input_size: int = 224,
        num_classes: int = 2,
        device: str | None = None,
        threshold: float = 0.5,
    ):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.input_size = input_size
        self.num_classes = num_classes
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.model = self._load_model()

        pass

    def _build_model(self, checkpoint_args): 
        model = models.__dict__[self.model_name](
            # img_size=self.input_size,
            num_classes=self.num_classes,
            drop_path_rate=0.2,
            #global_pool=True,
            args=checkpoint_args
        )

        return model   

    def _load_model(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        checkpoint_args = checkpoint["args"]  # 👈 pega args do treino

        model = self._build_model(checkpoint_args)

        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=True)

        model.to(self.device)
        model.eval()

        return model
    
    def _prepare_image(self, image_bytes: bytes):
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        tensor = self.transform(image).unsqueeze(0)

        return tensor.to(self.device)
    
    @torch.no_grad()
    def predict_bytes(self, image_bytes: bytes):
        x = self._prepare_image(image_bytes)
        logits = self.model(x)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        print("logits shape:", logits.shape)

        if logits.ndim == 3:
            logits = logits[:, 0, :]
        elif logits.ndim == 1:
            logits = logits.unsqueeze(0)

        probs = F.softmax(logits, dim=-1).detach().cpu()
        print("probs shape:", probs.shape)

        abnormal_prob = probs[0, 1].item()
        pred_idx = 1 if abnormal_prob >= self.threshold else 0

        confidence = float(probs[0, pred_idx].item())

        print("probs:", probs)

        probabilities = {
            CLASS_NAMES[i]: float(probs[0, i].item())
            for i in range(self.num_classes)
        }

        return {
            "predicted_class": pred_idx,
            "predicted_label": CLASS_NAMES[pred_idx],
            "confidence": confidence,
            "probabilities": probabilities,
        }
