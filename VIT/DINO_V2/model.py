import torch
import torch.nn as nn
from transformers import Dinov2Model
from peft import LoraConfig, get_peft_model

class DINOv2Classifier(nn.Module):
    def __init__(self, backbone, hidden_dim=1024, num_classes=40, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        patch_mean = outputs.last_hidden_state[:, 1:].mean(dim=1)
        features = cls_token + patch_mean
        logits = self.head(features)
        return logits

def build_model(num_classes: int, device: str):
    backbone = Dinov2Model.from_pretrained("facebook/dinov2-large")

    for p in backbone.parameters():
        p.requires_grad = False

    backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()

    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=0.1,
        bias="lora_only",
    )
    backbone = get_peft_model(backbone, lora_config)

    model = DINOv2Classifier(
        backbone=backbone,
        hidden_dim=1024,
        num_classes=num_classes,
        dropout=0.3,
    )

    for p in model.head.parameters():
        p.requires_grad = True

    return model.to(device)
