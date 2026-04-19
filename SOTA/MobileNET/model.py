import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

def build_model(num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:

    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_large(weights=weights)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    if hasattr(model.classifier[2], "p"):
        model.classifier[2].p = dropout

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
