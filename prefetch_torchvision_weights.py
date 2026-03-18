#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models import create_model


MODELS_TO_CACHE = [
    ("efficientnet", "medium", 14),
    ("efficientnet", "small", 1),
    ("resnet", "medium", 14),
    ("resnet", "small", 1),
    ("convnext", "small", 14),
    ("convnext", "tiny", 1),
    ("swin", "small", 14),
    ("swin", "tiny", 1),
    ("vit", "base", 14),
    ("vit", "small", 1),
]


def main():
    for model_name, model_size, num_classes in MODELS_TO_CACHE:
        print(f"Caching weights for model_name={model_name} model_size={model_size} num_classes={num_classes}", flush=True)
        model = create_model(
            model_name=model_name,
            model_size=model_size,
            pretrained=True,
            num_classes=num_classes,
            use_lora=False,
            use_qlora=False,
            lora_r=16,
        )
        del model

    print("All requested torchvision weights were instantiated and cached.", flush=True)


if __name__ == "__main__":
    main()
