import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import CLIPModel

class StrongerCLIPLoRA(nn.Module):
    def __init__(
        self,
        num_classes: int = 40,
        model_name: str = "openai/clip-vit-base-patch32",
        lora_r: int = 2,
        lora_alpha: int = 8,
        lora_dropout: float = 0.10,
        target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj"),
    ):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)

        for p in self.clip.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=list(target_modules),
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.clip.vision_model = get_peft_model(self.clip.vision_model, lora_cfg)

        dim = self.clip.config.projection_dim

        self.classifier = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.35),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        vision_outputs = self.clip.vision_model(pixel_values=x, return_dict=True)
        pooled_output = vision_outputs.pooler_output
        features = self.clip.visual_projection(pooled_output)
        features = features / features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return self.classifier(features)

def build_model(num_classes: int, device: str, **kwargs) -> nn.Module:
    model = StrongerCLIPLoRA(num_classes=num_classes, **kwargs)
    return model.to(device)

def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

count_parameters(StrongerCLIPLoRA())