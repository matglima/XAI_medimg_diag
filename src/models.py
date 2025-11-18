# -----------------------------------------------------------------
# File: models.py
# -----------------------------------------------------------------
# Description:
# Defines the ModelWrapper. This is the core model creation factory,
# now with support for num_classes, LoRA, Q-LoRA, and custom lora_r.
# compatible with ResNet, EfficientNet, ConvNeXt, ViT, and Swin.
# -----------------------------------------------------------------

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# --- LoRA / Q-LoRA Imports ---
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
                 use_qlora: bool = False,
                 lora_r: int = 16): 
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
        self.lora_r = lora_r 

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
        """
        Modify the model's final layer for num_classes.
        Handles unwrapping Sequentials for PEFT compatibility.
        """
        # --- 1. ResNet ---
        if hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.num_classes)
            
        # --- 2. ConvNeXt ---
        elif self.model_name == 'convnext': 
            # ConvNeXt classifier is Sequential(LayerNorm, Flatten, Linear).
            # We MUST keep the LayerNorm, so we only modify the internal Linear layer.
            # The internal linear layer is usually at index 2.
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, self.num_classes)
            
        # --- 3. EfficientNet ---
        elif hasattr(model, 'classifier'): 
            # EfficientNet classifier is Sequential(Dropout, Linear).
            # To avoid "Target module Sequential not supported", we unwrap it.
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
            elif isinstance(model.classifier, nn.Linear):
                in_features = model.classifier.in_features
            else:
                in_features = 1280 # Fallback
            
            # Replace the Sequential block with a single Linear layer
            model.classifier = nn.Linear(in_features, self.num_classes)
            
        # --- 4. ViT ---
        elif hasattr(model, 'heads'): 
            # ViT uses 'heads' which is Sequential(OrderedDict([('head', Linear)])).
            # We unwrap this to be direct for PEFT simplicity.
            if isinstance(model.heads, nn.Sequential):
                # Access the linear layer inside
                for module in model.heads.modules():
                    if isinstance(module, nn.Linear):
                        in_features = module.in_features
                        break
            else:
                in_features = model.heads.head.in_features
            
            # Replace 'heads' with a single Linear layer if possible, 
            # or replace the internal 'head' if the structure forces it.
            # Torchvision ViT forward pass calls self.heads(x). 
            # We can replace self.heads with a Linear layer safely.
            model.heads = nn.Linear(in_features, self.num_classes)
            
        # --- 5. Swin ---
        elif hasattr(model, 'head'): 
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, self.num_classes)
            
        else:
            raise TypeError(f"Could not find a classifier head for model {self.model_name}")

    def _create_model(self, pretrained):
        """Create model with optional LoRA/Q-LoRA"""
        
        weights = 'DEFAULT' if pretrained else None
        model_fn = self._get_model_fn()
        
        model_kwargs = {'weights': weights}

        if self.use_qlora:
            quantization_config = bnb.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model_kwargs['quantization_config'] = quantization_config

        # Load base model
        model = model_fn(**model_kwargs)
        
        # Modify head BEFORE applying LoRA
        self._modify_head(model)
        
        if self.use_lora:
            # --- Dynamic LoRA Targeting ---
            # Instead of a hardcoded list that might hit a Sequential block,
            # we scan the model and find the actual Linear modules.
            
            # Keywords to look for in layer names
            target_keywords = ['query', 'key', 'value', 'fc', 'head', 'classifier', 'downsample', 'project', 'expand']
            actual_targets = []

            for name, module in model.named_modules():
                # Only target Linear layers (and 4bit Linear)
                if isinstance(module, (nn.Linear, bnb.nn.Linear4bit)):
                    # Check if the name looks like a target
                    if any(k in name for k in target_keywords):
                        # We found a target.
                        # If it's inside a Sequential (like ConvNeXt 'classifier.2'),
                        # name will be 'classifier.2'. PEFT handles this correctly.
                        # We extract the suffix to be safe/clean.
                        
                        suffix = name.split('.')[-1]
                        # If the suffix is just a number (like in Sequential), we might need the full path or parent
                        if suffix.isdigit():
                            # Use the full name (e.g. classifier.2)
                            actual_targets.append(name)
                        else:
                            # Use the suffix (e.g. fc, head)
                            actual_targets.append(suffix)

            # Deduplicate
            actual_targets = list(set(actual_targets))
            
            print(f"Auto-detected LoRA Target Modules: {actual_targets}")

            lora_config = LoraConfig(
                r=self.lora_r, 
                lora_alpha=self.lora_r * 2, 
                target_modules=actual_targets,
                lora_dropout=0.1,
                bias="none",
            )
            
            model = get_peft_model(model, lora_config)
            print("Successfully applied LoRA/Q-LoRA to model.")
            model.print_trainable_parameters()

        return model

    def forward(self, x):
        return self.model(x)


def create_model(model_name, model_size='base', pretrained=True, num_classes=1, 
                 use_lora=False, use_qlora=False, lora_r=16): 
    return ModelWrapper(
        model_name=model_name,
        model_size=model_size,
        pretrained=pretrained,
        num_classes=num_classes,
        use_lora=use_lora,
        use_qlora=use_qlora,
        lora_r=lora_r 
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