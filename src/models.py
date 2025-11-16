# -----------------------------------------------------------------
# File: models.py
# -----------------------------------------------------------------
# Description:
# Defines the ModelWrapper. This is the core model creation factory,
# now with support for num_classes, LoRA, and Q-LoRA.
# -----------------------------------------------------------------

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

# --- LoRA / Q-LoRA Imports ---
# Make sure to install: pip install peft bitsandbytes accelerate
try:
    from peft import get_peft_model, LoraConfig, TaskType
    from bitsandbytes.optim import AdamW8bit
    import bitsandbytes as bnb
    from accelerate import Accelerator
    PEFT_INSTALLED = True
except ImportError:
    PEFT_INSTALLED = False
    print("Warning: 'peft', 'bitsandbytes', or 'accelerate' not found. LoRA/Q-LoRA will be disabled.")


class ModelWrapper(nn.Module):
    def __init__(self, 
                 model_name: str, 
                 model_size: str = 'base',
                 pretrained: bool = True,
                 num_classes: int = 1,
                 use_lora: bool = False,
                 use_qlora: bool = False):
        """
        Wrapper to create a model with optional LoRA/Q-LoRA.

        Args:
            model_name (str): 'convnext', 'efficientnet', 'vit', 'swin', 'resnet'
            model_size (str): 'tiny', 'small', 'base', 'large', etc.
            pretrained (bool): Whether to use pretrained weights.
            num_classes (int): Number of output classes (1 for binary, 14 for gate).
            use_lora (bool): Whether to apply LoRA.
            use_qlora (bool): Whether to apply Q-LoRA (4-bit quantization).
        """
        super().__init__()
        
        if not PEFT_INSTALLED and (use_lora or use_qlora):
            raise ImportError("Please install 'peft', 'bitsandbytes', and 'accelerate' to use LoRA/Q-LoRA.")
            
        if use_qlora and not use_lora:
            raise ValueError("Q-LoRA (use_qlora=True) requires LoRA (use_lora=True) to be enabled.")

        self.model_name = model_name.lower()
        self.model_size = model_size.lower()
        self.num_classes = num_classes
        self.use_lora = use_lora
        self.use_qlora = use_qlora

        # Create base model with new weights API
        self.model = self._create_model(pretrained)

    def _get_model_fn(self):
        """Helper to get the model constructor from torchvision"""
        model_config = {
            'convnext': {
                'tiny': models.convnext_tiny,
                'small': models.convnext_small,
                'base': models.convnext_base,
                'large': models.convnext_large
            },
            'efficientnet': {
                'small': models.efficientnet_b0,
                'medium': models.efficientnet_b4,
                'large': models.efficientnet_b7
            },
            'vit': {
                'small': models.vit_b_16,
                'base': models.vit_b_16,
                'large': models.vit_l_16
            },
            'swin': {
                'tiny': models.swin_t,
                'small': models.swin_s,
                'base': models.swin_b,
            },
            'resnet': {
                'small': models.resnet18,
                'medium': models.resnet50,
                'large': models.resnet101
            }
        }
        if self.model_name not in model_config:
            raise ValueError(f"Unknown model_name: {self.model_name}")
        if self.model_size not in model_config[self.model_name]:
            raise ValueError(f"Unknown model_size '{self.model_size}' for '{self.model_name}'")
            
        return model_config[self.model_name][self.model_size]

    def _modify_head(self, model):
        """Modify the model's final layer for num_classes"""
        in_features = None
        
        if hasattr(model, 'fc'): # ResNet
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.num_classes)
            
        elif self.model_name == 'convnext': # ConvNeXt has a different structure
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, self.num_classes)
            
        elif hasattr(model, 'classifier'): # EfficientNet
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, self.num_classes)
            
        elif hasattr(model, 'heads'): # ViT
            in_features = model.heads.head.in_features
            model.heads.head = nn.Linear(in_features, self.num_classes)
            
        elif hasattr(model, 'head'): # Swin
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, self.num_classes)
            
        else:
            raise TypeError(f"Could not find a classifier head for model {self.model_name}")

    def _create_model(self, pretrained):
        """Create model with optional LoRA/Q-LoRA"""
        
        weights = 'DEFAULT' if pretrained else None
        model_fn = self._get_model_fn()
        
        quantization_config = None
        if self.use_qlora:
            quantization_config = bnb.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        # Load the base model with or without quantization
        model = model_fn(
            weights=weights,
            quantization_config=quantization_config
        )

        # Modify the head BEFORE applying LoRA
        self._modify_head(model)
        
        if self.use_lora:
            # Find all linear layers to target for LoRA
            # This is a robust way to catch them in CNNs and Transformers
            target_modules = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, bnb.nn.Linear4bit)):
                    # Get the final part of the name (e.g., 'fc', 'query', 'classifier')
                    module_name = name.split('.')[-1]
                    if module_name not in target_modules:
                         target_modules.append(module_name)
            
            # Ensure we don't target the final classification head if it's not quantized
            # Or, just target common module names
            target_modules = ['query', 'key', 'value', 'fc', 'fc1', 'fc2', 'head', 'classifier']

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION # Use this for general vision models
            )
            
            model = get_peft_model(model, lora_config)
            print("Successfully applied LoRA/Q-LoRA to model.")
            model.print_trainable_parameters()

        return model

    def forward(self, x):
        return self.model(x)


def create_model(model_name, model_size='base', pretrained=True, num_classes=1, use_lora=False, use_qlora=False):
    """
    Factory function to create the ModelWrapper.
    """
    return ModelWrapper(
        model_name=model_name,
        model_size=model_size,
        pretrained=pretrained,
        num_classes=num_classes,
        use_lora=use_lora,
        use_qlora=use_qlora
    )

def get_optimizer(model, lr, use_qlora=False):
    """
    Returns the appropriate optimizer.
    AdamW8bit is required for Q-LoRA.
    """
    if use_qlora and PEFT_INSTALLED:
        print("Using 8-bit AdamW optimizer for Q-LoRA")
        return AdamW8bit(model.parameters(), lr=lr)
    else:
        print("Using standard AdamW optimizer")
        return torch.optim.AdamW(model.parameters(), lr=lr)