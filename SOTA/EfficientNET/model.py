import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    efficientnet_b0,
    efficientnet_b3,
)

def build_model(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
    variant: str = "b0",
) -> nn.Module:

    variant = variant.lower()
    if variant == "b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
    elif variant == "b3":
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b3(weights=weights)
    else:
        raise ValueError(f"Unsupported EfficientNet variant: {variant}")

    in_features = model.classifier[1].in_features
    model.classifier[0] = nn.Dropout(p=dropout, inplace=True)
    model.classifier[1] = nn.Linear(in_features, num_classes)
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
