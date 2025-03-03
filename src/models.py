# models.py
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from typing import Optional
from torchcam.methods import GradCAM
from PIL import Image

class ModelWrapper(nn.Module):
    def __init__(self, 
                 model_name: str, 
                 model_size: str = 'base',
                 pretrained: bool = True):
        super().__init__()
        self.model_name = model_name.lower()
        self.model_size = model_size.lower()
        self.transform = self._get_transforms()
        # Create base model with new weights API
        self.model = self._create_model(pretrained)
        self._modify_head()
        
        # Standard normalization for visualization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def _create_model(self, pretrained):
        """Create model with proper weights handling"""
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
        
        weights = 'DEFAULT' if pretrained else None
        return model_config[self.model_name][self.model_size](weights=weights)

    def _modify_head(self):
        """Modify for binary classification"""
        in_features = None
        if hasattr(self.model, 'fc'):  # ConvNeXt, ResNet
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, 1)
        elif hasattr(self.model, 'classifier'):  # EfficientNet
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, 1)
        elif hasattr(self.model, 'heads'):  # Swin
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, 1)
        else:  # ViT
            in_features = self.model.heads.head.in_features
            self.model.heads.head = nn.Linear(in_features, 1)
    
    def _get_transforms(self):
        """Get model-specific preprocessing transforms"""
        return {
            'train': transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.ToTensor(),
                self._get_normalization()
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self._get_normalization()
            ])
        }
    
    def _get_normalization(self):
        """Get model-appropriate normalization"""
        if 'convnext' in self.model_name or 'swin' in self.model_name:
            return transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
        # Add other model-specific normalizations if needed
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])

    def forward(self, x):
        return self.model(x)

    def preprocess_image(self, img_path, img_size=224):
        """Preprocess image for model input"""
        img = Image.open(img_path).convert('RGB')
        img = transforms.Resize((img_size, img_size))(img)
        img_tensor = transforms.ToTensor()(img)
        return self.normalize(img_tensor)

def create_model(model_name, model_size='base', pretrained=True):
    return ModelWrapper(model_name, model_size, pretrained)