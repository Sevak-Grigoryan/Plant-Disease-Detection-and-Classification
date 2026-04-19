import torch
import torch.nn as nn
from torchvision.models import googlenet, GoogLeNet_Weights

def build_model(num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:

    weights = GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = googlenet(weights=weights, aux_logits=False, dropout=dropout)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def calculate_parameters(model: nn.Module) -> dict:

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": frozen,
    }

def print_parameter_summary(model: nn.Module) -> None:
    stats = calculate_parameters(model)
    print(f"Total parameters      : {stats['total']:,}")
    print(f"Trainable parameters  : {stats['trainable']:,}")
    print(f"Non-trainable params  : {stats['non_trainable']:,}")

if __name__ == "__main__":
    demo_model = build_model(num_classes=40)
    print_parameter_summary(demo_model)
