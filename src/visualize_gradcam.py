# visualize_gradcam.py
import argparse
import matplotlib.pyplot as plt
import torch
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from models import create_model
from PIL import Image
import numpy as np

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

def visualize(model, img_path, target_layer, true_label):
    # Preprocess image
    img_tensor = model.preprocess_image(img_path)
    
    # Create CAM extractor
    with SmoothGradCAMpp(model.model, target_layer=target_layer) as cam_extractor:
        output = model(img_tensor.unsqueeze(0))
        prob = torch.sigmoid(output).item()
        class_idx = 0 if prob < 0.5 else 1
        activation_map = cam_extractor(class_idx, output)
    
    # Visualization
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    result = overlay_mask(img, to_pil_image(activation_map[0].squeeze().numpy(), mode='F'), alpha=0.5)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title(f'Original (True: {true_label})')
    
    ax[1].imshow(result)
    ax[1].axis('off')
    ax[1].set_title(f'GradCAM (Pred: {class_idx})')
    
    return fig

def main(args):
    # Load model
    model = create_model(args.model_name, args.model_size, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()
    
    # Auto-detect target layer if not specified
    if args.target_layer is None:
        args.target_layer = get_last_conv_layer(model.model)
        print(f"Automatically detected target layer: {args.target_layer}")

    # Generate visualization
    fig = visualize(model, args.image_path, args.target_layer, args.true_label)
    fig.savefig(args.output_path, bbox_inches='tight')
    print(f"Saved visualization to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_size', type=str, default='base')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--target_layer', type=str, default=None,
                       help="Layer name for GradCAM (autodetected if not specified)")
    parser.add_argument('--true_label', type=int, default=1, required=True)
    parser.add_argument('--output_path', type=str, default='gradcam_output.png')
    
    args = parser.parse_args()
    main(args)
