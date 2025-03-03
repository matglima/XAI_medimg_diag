# visualize_gradcam.py
import argparse
import matplotlib.pyplot as plt
import torch
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from models import create_model
from PIL import Image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.io.image import read_image
import torch
import numpy as np

# Set all seeds
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Force deterministic algorithms (for CUDA)
torch.use_deterministic_algorithms(True)




def visualize(model, img_path, target_layer):
    # Preprocess image
    img_tensor = model.preprocess_image(img_path)
    
    # Create CAM extractor
    with SmoothGradCAMpp(model.model, target_layer=target_layer) as cam_extractor:
        # Forward pass
        output = model(img_tensor.unsqueeze(0))
        
        # Get class index (binary classification)
        class_idx = 0 if torch.sigmoid(output).item() < 0.5 else 0
        
        # Extract CAM
        activation_map = cam_extractor(class_idx, output)
    
    # Overlay CAM
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    result = overlay_mask(img, to_pil_image(activation_map[0].squeeze().numpy(), mode='F'), alpha=0.5)
    
    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title('Original')
    
    ax[1].imshow(result)
    ax[1].axis('off')
    ax[1].set_title('GradCAM')
    
    return fig

def main(args):
    # Load model
    model = create_model(args.model_name, args.model_size, pretrained=False)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()
    
    # Visualize
    fig = visualize(model, args.image_path, args.target_layer)
    fig.savefig(args.output_path, bbox_inches='tight')
    print(f"Saved visualization to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--model_size', type=str, default='base')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--target_layer', type=str, required=True,
                       help="Layer name for GradCAM (e.g., 'layer4' for ResNet)")
    parser.add_argument('--output_path', type=str, default='gradcam_output.png')
    
    args = parser.parse_args()
    main(args)