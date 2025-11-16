# -----------------------------------------------------------------
# File: 3_calibrate_moe.py
# -----------------------------------------------------------------
# Description:
# Phase 2B: Assembles the MoE and runs a brief, low-LR
# fine-tuning (calibration) on the final layers.
# -----------------------------------------------------------------

import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
import os

from dataloader import MultiLabelRetinaDataset, get_random_splits, train_transform, val_transform
from moe_model import HybridMoE
from models import get_optimizer # Re-use optimizer factory

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This MUST match the list in 1_train_gate.py and run_all_experts.sh
PATHOLOGIES = [
    'diabetic_retinopathy', 'age_related_macular_degeneration', 'glaucoma', 
    'hypertensive_retinopathy', 'pathology_5', 'pathology_6', 'pathology_7', 
    'pathology_8', 'pathology_9', 'pathology_10', 'pathology_11', 
    'pathology_12', 'pathology_13', 'pathology_14'
]

# --- Main Calibration Function ---
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # --- Define Model Configurations ---
    # These MUST match the configs used for training
    gate_config = {
        'model_name': args.gate_model_name,
        'model_size': args.gate_model_size,
        'use_lora': args.gate_use_lora,
        'use_qlora': args.gate_use_qlora,
    }
    expert_config = {
        'model_name': args.expert_model_name,
        'model_size': args.expert_model_size,
        'use_lora': args.expert_use_lora,
        'use_qlora': args.expert_use_qlora,
    }

    # --- 1. Initialize MoE Structure ---
    logger.info("Initializing HybridMoE structure...")
    moe = HybridMoE(gate_config, expert_config, PATHOLOGIES, device)
    
    # --- 2. Load All Checkpoints ---
    logger.info("Loading pre-trained checkpoints...")
    # Adjust path for full models
    gate_ckpt_path = args.gate_ckpt_path
    if not args.gate_use_lora:
        gate_ckpt_path = os.path.join(args.gate_ckpt_path, "gate_best_model.pth")
        
    moe.load_checkpoints(
        gate_ckpt_path=gate_ckpt_path,
        expert_ckpt_dir=args.expert_ckpt_dir,
        gate_is_lora=args.gate_use_lora,
        expert_is_lora=args.expert_use_lora
    )
    
    # --- 3. Freeze Parameters for Calibration ---
    logger.info("Freezing model parameters for calibration...")
    total_params = 0
    trainable_params = 0
    
    for name, param in moe.named_parameters():
        total_params += param.numel()
        # Freeze everything by default
        param.requires_grad = False
        
        # Unfreeze ONLY LoRA adapters and final classifier layers
        # This is the key to calibration: don't destroy pre-trained features.
        if 'lora_' in name or 'fc' in name or 'classifier' in name or 'head' in name:
            param.requires_grad = True
            trainable_params += param.numel()
            # logger.info(f"Unfreezing: {name}")

    logger.info(f"Calibration complete. Trainable params: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")

    # --- 4. Load Data for Calibration ---
    logger.info("Loading calibration data (using training split)...")
    labels_df = pd.read_csv(args.labels_path)
    labels_df['image_id'] = labels_df['image_id'].astype(str)
    splits = get_random_splits(labels_df) # Use same splits as gate
    
    train_dataset = MultiLabelRetinaDataset(
        labels_df, args.image_dir, PATHOLOGIES, train_transform, splits['train']
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True
    )
    logger.info(f"Calibration samples: {len(train_dataset)}")

    # --- 5. Run Calibration Loop ---
    criterion = nn.BCEWithLogitsLoss()
    # Use the correct optimizer (8-bit if any component used Q-LoRA)
    use_qlora = args.gate_use_qlora or args.expert_use_qlora
    optimizer = get_optimizer(moe, args.lr, use_qlora)
    
    moe.train()
    for epoch in range(args.epochs):
        logger.info(f"\n--- Calibration Epoch {epoch+1}/{args.epochs} ---")
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Calibrating (LR={args.lr:.0e})")
        
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = moe(images)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss / (len(progress_bar)))
            
    # --- 6. Save Final Calibrated Model ---
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "moe_calibrated_final.pth")
    
    # IMPORTANT: Save the *entire* model state dict,
    # as it now contains all loaded+calibrated components.
    
    # If LoRA was used, merge weights before saving for easy inference
    if args.gate_use_lora or args.expert_use_lora:
        logger.info("Merging LoRA adapters into base model for saving...")
        if hasattr(moe.gate, 'model') and hasattr(moe.gate.model, 'merge_and_unload'):
            moe.gate.model = moe.gate.model.merge_and_unload()
        for expert in moe.experts:
            if hasattr(expert, 'model') and hasattr(expert.model, 'merge_and_unload'):
                expert.model = expert.model.merge_and_unload()
        logger.info("Adapters merged.")
    
    torch.save(moe.state_dict(), save_path)
    logger.info(f"Final calibrated model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Hybrid MoE Model")
    
    # Data params
    parser.add_argument('--labels-path', type=str, required=True, help="Path to labels.csv")
    parser.add_argument('--image-dir', type=str, required=True, help="Directory with fundus images")
    parser.add_argument('--output-dir', type=str, default="checkpoints/final_moe", help="Directory to save final model")
    
    # Checkpoint paths
    parser.add_argument('--gate-ckpt-path', type=str, default="checkpoints/gate", help="Path to gate checkpoint (dir if LoRA, full file path if full model)")
    parser.add_argument('--expert-ckpt-dir', type=str, default="checkpoints/experts", help="Directory containing all expert checkpoints")

    # Gate Config (must match 1_train_gate.py)
    parser.add_argument('--gate-model-name', type=str, default='resnet')
    parser.add_argument('--gate-model-size', type=str, default='small')
    parser.add_argument('--gate-use-lora', action='store_true')
    parser.add_argument('--gate-use-qlora', action='store_true')

    # Expert Config (must match 2_train_experts.py)
    parser.add_argument('--expert-model-name', type=str, default='resnet')
    parser.add_argument('--expert-model-size', type=str, default='small')
    parser.add_argument('--expert-use-lora', action='store_true')
    parser.add_argument('--expert-use-qlora', action='store_true')
    
    # Calibration params
    parser.add_argument('--epochs', type=int, default=3, help="Number of calibration epochs (SHORT)")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size (use smaller for calibration)")
    parser.add_argument('--lr', type=float, default=1e-6, help="Learning rate (VERY LOW)")
    parser.add_argument('--num-workers', type=int, default=4, help="Dataloader workers")
    
    args = parser.parse_args()
    main(args)