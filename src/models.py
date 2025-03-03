# models.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import transforms
from torchvision.transforms import functional as TF
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from typing import Optional, Dict, Any

class ModelWrapper(nn.Module):
    def __init__(self, 
                 model_name: str, 
                 model_size: str = 'base', 
                 pretrained: bool = True,
                 gradcam_layer: Optional[str] = None):
        super().__init__()
        self.model_name = model_name.lower()
        self.model_size = model_size.lower()
        self.pretrained = pretrained
        self.transform = self._get_transforms()
        
        # Create base model
        self.model = self._create_model()
        self._modify_head()
        
        # GradCAM setup
        self.gradcam = None
        self._setup_gradcam(gradcam_layer)
        
    def _get_model_config(self) -> Dict[str, Any]:
        """Get architecture configuration based on model name and size"""
        configs = {
            'convnext': {
                'base': models.convnext_base,
                'small': models.convnext_small,
                'large': models.convnext_large,
                'tiny': models.convnext_tiny
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
        return configs[self.model_name][self.model_size]

    def _create_model(self):
        """Instantiate the selected model"""
        try:
            model_fn = self._get_model_config()
            return model_fn(pretrained=self.pretrained)
        except KeyError:
            raise ValueError(f"Invalid model combination: {self.model_name} {self.model_size}")

    def _modify_head(self):
        """Modify model head for binary classification"""
        in_features = None
        
        # Get input features for the last layer
        if hasattr(self.model, 'fc'):  # ResNet, ConvNeXt
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, 1)
        elif hasattr(self.model, 'classifier'):  # EfficientNet
            if isinstance(self.model.classifier, nn.Sequential):
                in_features = self.model.classifier[-1].in_features
                self.model.classifier[-1] = nn.Linear(in_features, 1)
            else:
                in_features = self.model.classifier.in_features
                self.model.classifier = nn.Linear(in_features, 1)
        elif hasattr(self.model, 'heads'):  # Swin Transformer
            in_features = self.model.head.in_features
            self.model.head = nn.Linear(in_features, 1)
        else:
            raise NotImplementedError(f"Unsupported model architecture: {self.model_name}")

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

    def _setup_gradcam(self, layer_name: Optional[str]):
        """Initialize GradCAM hook"""
        # Don't set up GradCAM by default to avoid validation errors
        self.gradcam = None
        self.gradcam_layer = layer_name
        
        # Store the layer name but don't initialize GradCAM yet
        if layer_name is None:
            # Set default target layers for different architectures
            if 'resnet' in self.model_name:
                self.gradcam_layer = 'layer4'
            elif 'efficientnet' in self.model_name:
                self.gradcam_layer = 'features'
            elif 'convnext' in self.model_name:
                self.gradcam_layer = 'features'
            elif 'vit' in self.model_name:
                self.gradcam_layer = 'encoder.layers.encoder_layer_11'
            elif 'swin' in self.model_name:
                self.gradcam_layer = 'layers.3'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_gradcam(self, 
                   input_tensor: torch.Tensor, 
                   target_class: int = None) -> torch.Tensor:
        """
        Generate GradCAM heatmap for input tensor
        Returns:
            torch.Tensor: Heatmap tensor (H, W)
        """
        if self.gradcam_layer is None:
            raise RuntimeError("GradCAM layer not specified")
        
        # Initialize GradCAM only when needed
        self.gradcam = GradCAM(self.model, self.gradcam_layer)
        
        # Store original model mode
        original_mode = self.training
        
        # GradCAM requires eval mode
        self.eval()
        
        try:
            # Make sure gradients are enabled for this operation
            with torch.enable_grad():
                # Forward pass
                output = self(input_tensor)
                
                # Backward pass for specified class
                if target_class is None:
                    target_class = (output.sigmoid() > 0.5).long()
                
                self.zero_grad()
                output[:, target_class].sum().backward(retain_graph=True)
                
                # Generate heatmap
                activation_map = self.gradcam(target_class)
                
                # Post-process heatmap
                heatmap = overlay_mask(TF.to_pil_image(input_tensor.squeeze(0).cpu()),
                                     TF.to_pil_image(activation_map[0].squeeze(0).cpu()),
                                     alpha=0.5)
                
                return TF.to_tensor(heatmap)
        
        except Exception as e:
            print(f"Error generating GradCAM: {e}")
            raise
        finally:
            # Restore original model mode
            self.train(original_mode)
            # Clear hooks to prevent memory leaks
            if self.gradcam is not None:
                self.gradcam.clear_hooks()
                self.gradcam = None
            self.train(original_mode)
            self.gradcam.clear_hooks()

    @staticmethod
    def available_models() -> Dict[str, list]:
        return {
            'convnext': ['tiny', 'small', 'base', 'large'],
            'efficientnet': ['small', 'medium', 'large'],
            'vit': ['small', 'base', 'large'],
            'swin': ['tiny', 'small', 'base'],
            'resnet': ['small', 'medium', 'large']
        }

def create_model(model_name: str, 
                model_size: str = 'base',
                pretrained: bool = True,
                gradcam_layer: Optional[str] = None) -> ModelWrapper:
    """
    Factory function for creating models
    Example:
        model = create_model('resnet', 'medium')
        gradcam = model.get_gradcam(input_tensor)
    """
    return ModelWrapper(model_name, model_size, pretrained, gradcam_layer)