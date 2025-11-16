# -----------------------------------------------------------------
# File: evaluate.py
# -----------------------------------------------------------------
# Description:
# Phase 3: Evaluates the final, calibrated MoE model on
# the test set and prints a full report.
# -----------------------------------------------------------------

import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from tqdm import tqdm
import os

from dataloader import MultiLabelRetinaDataset, get_random_splits, val_transform
from moe_model import HybridMoE

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This MUST match all other scripts
    PATHOLOGIES = [
        'diabetes',
        'diabetic_retinopathy',
        'macular_edema',
        'scar',
        'nevus',
        'amd',
        'vascular_occlusion',
        'hypertensive_retinopathy',
        'drusens',
        'hemorrhage',
        'retinal_detachment',
        'myopic_fundus',
        'increased_cup_disc',
        'other'
    ]

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # --- 1. Define Model Configs (must match calibration) ---
    gate_config = {
        'model_name': args.gate_model_name,
        'model_size': args.gate_model_size,
        'use_lora': False, 'use_qlora': False # weights are merged, so LoRA=False
    }
    expert_config = {
        'model_name': args.expert_model_name,
        'model_size': args.expert_model_size,
        'use_lora': False, 'use_qlora': False
    }

    # --- 2. Initialize MoE Structure ---
    logger.info("Initializing HybridMoE structure for evaluation...")
    model = HybridMoE(gate_config, expert_config, PATHOLOGIES, device)
    
    # --- 3. Load Final Calibrated Weights ---
    logger.info(f"Loading final calibrated model from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- 4. Load Test Data ---
    logger.info("Loading test data...")
    labels_df = pd.read_csv(args.labels_path)
    labels_df['image_id'] = labels_df['image_id'].astype(str)
    # Use the same split logic as the gate to get the 'test' set
    splits = get_random_splits(labels_df) 
    
    test_dataset = MultiLabelRetinaDataset(
        labels_df, args.image_dir, PATHOLOGIES, val_transform, splits['test'] # Use val_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )
    logger.info(f"Test samples: {len(test_dataset)}")

    # --- 5. Run Evaluation ---
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating on Test Set"):
            images = images.to(device)
            
            outputs = model(images)
            
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds_rounded = np.round(all_preds)
    
    # --- 6. Print Full Report ---
    logger.info("\n--- FINAL MODEL EVALUATION REPORT ---")
    
    # Overall Metrics
    f1_macro = f1_score(all_targets, all_preds_rounded, average='macro', zero_division=0)
    auc_macro = roc_auc_score(all_targets, all_preds, average='macro')
    
    logger.info(f"\nOverall Metrics (Macro Average):")
    logger.info(f"  Macro F1-Score: {f1_macro:.4f}")
    logger.info(f"  Macro AUC:      {auc_macro:.4f}")
    
    # Per-Class Metrics
    logger.info("\nPer-Class Metrics:")
    print("=" * 60)
    print(f"{'Pathology':<35} | {'AUC':<7} | {'F1-Score':<10}")
    print("-" * 60)
    
    for i, pathology in enumerate(PATHOLOGIES):
        class_auc = roc_auc_score(all_targets[:, i], all_preds[:, i])
        class_f1 = f1_score(all_targets[:, i], all_preds_rounded[:, i], zero_division=0)
        print(f"{pathology:<35} | {class_auc:<7.4f} | {class_f1:<10.4f}")
    
    print("=" * 60)
    
    logger.info("\nClassification Report (Micro/Macro/Weighted):")
    report = classification_report(all_targets, all_preds_rounded, target_names=PATHOLOGIES, zero_division=0)
    print(report)
    logger.info("--- END OF REPORT ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Final Hybrid MoE Model")
    
    # Data params
    parser.add_argument('--labels-path', type=str, required=True, help="Path to labels.csv")
    parser.add_argument('--image-dir', type=str, required=True, help="Directory with fundus images")
    
    # Model path
    parser.add_argument('--model-path', type=str, default="checkpoints/final_moe/moe_calibrated_final.pth", help="Path to final calibrated .pth model")

    # Model Config (must match calibration)
    parser.add_argument('--gate-model-name', type=str, default='resnet')
    parser.add_argument('--gate-model-size', type=str, default='small')
    parser.add_argument('--expert-model-name', type=str, default='resnet')
    parser.add_argument('--expert-model-size', type=str, default='small')
    
    # Eval params
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for evaluation")
    parser.add_argument('--num-workers', type=int, default=4, help="Dataloader workers")
    
    args = parser.parse_args()
    main(args)