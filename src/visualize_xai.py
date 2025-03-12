# visualize_gradcam.py
import argparse
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from models import create_model
from PIL import Image
import numpy as np
import os

# Set all seeds
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

def get_last_conv_layer(model):
    """Automatically find the last convolutional layer in a model"""
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = name
    if last_conv is None:
        raise RuntimeError("No convolutional layers found in the model")
    return last_conv

def get_true_label(labels_path, image_path):
    """Retrieve true label from CSV based on image filename"""
    labels_df = pd.read_csv(labels_path)
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    labels_df['image_id'] = labels_df['image_id'].astype(str)
    return labels_df[labels_df['image_id'] == image_id][args.diagnosis].values[0]

def visualize(model, img_path, target_layer, labels_path):
    # Get true label from CSV
    true_label = get_true_label(labels_path, img_path)
    
    # Load and prepare image
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    
    # --- Grad-CAM Processing ---
    # Preprocess image
    img_tensor = model.preprocess_image(img_path)
    
    # Forward pass and CAM extraction
    with SmoothGradCAMpp(model.model, target_layer=target_layer) as cam_extractor:
        output = model(img_tensor.unsqueeze(0))
        prob = torch.sigmoid(output).item()
        class_idx = 0 if prob < 0.5 else 1
        
        # Handle single-class output dimension
        if output.shape[1] == 1:  # Binary classification with single output
            activation_map = cam_extractor(0, output)  # Always use class 0 for CAM
        else:
            activation_map = cam_extractor(class_idx, output)
    
    # Create Grad-CAM overlay
    gradcam_result = overlay_mask(img, to_pil_image(activation_map[0].squeeze().numpy(), mode='F'), alpha=0.5)
    
    # --- Saliency Map Processing ---
    # Preprocess image with gradient tracking
    saliency_tensor = model.preprocess_image(img_path).requires_grad_()
    # Forward pass
    saliency_output = model(saliency_tensor.unsqueeze(0))
    # Backward pass to get gradients
    saliency_output.backward()
    # Process gradients
    gradients = saliency_tensor.grad.data.abs()
    saliency_map = gradients.max(dim=0)[0].cpu().numpy()
    # Normalize to [0,1]
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

    # --- Visualization ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # Original image
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title(f'Original (True: {true_label})')
    # Grad-CAM
    ax[1].imshow(gradcam_result)
    ax[1].axis('off')
    ax[1].set_title(f'GradCAM (Pred: {class_idx}, Prob: {prob:.2f})')
    # Saliency map
    ax[2].imshow(saliency_map, cmap='hot')
    ax[2].axis('off')
    ax[2].set_title('Saliency Map')
    
    return fig

def main(args):
    # Load model safely
    model = create_model(args.model_name, args.model_size, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint_path, weights_only=True))
    model.eval()
    
    # Auto-detect target layer if not specified
    if args.target_layer is None:
        args.target_layer = get_last_conv_layer(model.model)
        print(f"Automatically detected target layer: {args.target_layer}")

    # Generate visualization
    fig = visualize(model, args.image_path, args.target_layer, args.labels_path)
    fig.savefig(args.output_path, bbox_inches='tight')
    print(f"Saved visualization to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_size', type=str, default='base')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--labels_path', type=str, default='data/labels_brset.csv',
                       help='Path to labels CSV file')
    parser.add_argument('--target_layer', type=str, default=None,
                       help="Layer name for GradCAM (autodetected if not specified)")
    parser.add_argument('--output_path', type=str, default='visualization_output.png')
    parser.add_argument('--diagnosis', type=str, default='diabetic_retinopathy')
    
    args = parser.parse_args()
    main(args)
