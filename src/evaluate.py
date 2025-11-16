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
import warnings

from dataloader import MultiLabelRetinaDataset, get_random_splits, val_transform
from moe_model import HybridMoE
from config import BRSET_LABELS # <-- NEW IMPORT

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_metrics_report(targets, preds, pathology_list): # <-- Pass list as arg
    """Generates a detailed, multi-line string report."""
    preds_rounded = np.round(preds)
    report_lines = []
    
    # Overall Metrics
    f1_macro = f1_score(targets, preds_rounded, average='macro', zero_division=0)
    
    # Robust AUC
    auc_scores = []
    num_classes = targets.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(num_classes):
            try:
                if len(np.unique(targets[:, i])) > 1:
                    auc_scores.append(roc_auc_score(targets[:, i], preds[:, i]))
                else:
                    auc_scores.append(np.nan) # Use nan to ignore in mean
            except ValueError:
                auc_scores.append(np.nan)
                
    auc_macro = np.nanmean(auc_scores)
    
    report_lines.append("\n--- FINAL MODEL EVALUATION REPORT ---")
    report_lines.append(f"\nOverall Metrics (Macro Average):")
    report_lines.append(f"  Macro F1-Score: {f1_macro:.4f}")
    report_lines.append(f"  Macro AUC:      {auc_macro:.4f}")
    
    # Per-Class Metrics
    report_lines.append("\nPer-Class Metrics:")
    report_lines.append("=" * 60)
    report_lines.append(f"{'Pathology':<35} | {'AUC':<7} | {'F1-Score':<10}")
    report_lines.append("-" * 60)
    
    for i, pathology in enumerate(pathology_list): # <-- Use dynamic list
        class_auc = auc_scores[i] if not np.isnan(auc_scores[i]) else 0.0
        class_f1 = f1_score(targets[:, i], preds_rounded[:, i], zero_division=0)
        auc_str = f"{class_auc:<7.4f}" if not np.isnan(auc_scores[i]) else "N/A    "
        report_lines.append(f"{pathology:<35} | {auc_str} | {class_f1:<10.4f}")
    
    report_lines.append("=" * 60)
    
    report_lines.append("\nClassification Report (Micro/Macro/Weighted):")
    class_report = classification_report(targets, preds_rounded, target_names=pathology_list, zero_division=0)
    report_lines.append(class_report)
    report_lines.append("--- END OF REPORT ---")
    
    return "\n".join(report_lines)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # --- DYNAMICALLY LOAD PATHOLOGIES ---
    PATHOLOGIES = BRSET_LABELS
    
    # --- 1. Define Model Configs (must match training) ---
    gate_config = {
        'model_name': args.gate_model_name,
        'model_size': args.gate_model_size,
        'use_lora': args.gate_use_lora,
        'use_qlora': args.gate_use_qlora,
        'lora_r': args.lora_r
    }
    expert_config = {
        'model_name': args.expert_model_name,
        'model_size': args.expert_model_size,
        'use_lora': args.expert_use_lora,
        'use_qlora': args.expert_use_qlora,
        'lora_r': args.lora_r
    }

    # --- 2. Initialize MoE Structure (with adapters) ---
    logger.info("Initializing HybridMoE structure for evaluation (with adapters)...")
    model = HybridMoE(gate_config, expert_config, PATHOLOGIES, device)
    
    # --- 3. Load Final Calibrated Weights (loads adapter weights) ---
    logger.info(f"Loading final calibrated model from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)

    # --- 4. Merge adapters for efficient inference ---
    if args.gate_use_lora or args.expert_use_lora:
        logger.info("Merging LoRA adapters into base model for evaluation...")
        if hasattr(model.gate, 'model') and hasattr(model.gate.model, 'merge_and_unload'):
            model.gate.model = model.gate.model.merge_and_unload()
        for expert in model.experts:
            if hasattr(expert, 'model') and hasattr(expert.model, 'merge_and_unload'):
                expert.model = expert.model.merge_and_unload()
        logger.info("Adapters merged.")

    model.eval()

    # --- 5. Load Test Data ---
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

    # --- 6. Run Evaluation ---
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating on Test Set"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # --- 7. Print Full Report ---
    report = get_metrics_report(all_targets, all_preds, PATHOLOGIES)
    logger.info(report)
    
    report_path = os.path.join(os.path.dirname(args.model_path), "final_evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Final Hybrid MoE Model")
    
    # Data params
    parser.add_argument('--labels-path', type=str, required=True, help="Path to labels.csv")
    parser.add_argument('--image-dir', type=str, required=True, help="Directory with fundus images")
    
    # Model path
    parser.add_argument('--model-path', type=str, default="checkpoints/final_moe/moe_calibrated_final.pth", help="Path to final calibrated .pth model")
    
    # --- Model Config Flags ---
    parser.add_argument('--gate-model-name', type=str, default='resnet')
    parser.add_argument('--gate-model-size', type=str, default='small')
    parser.add_argument('--gate-use-lora', action='store_true')
    parser.add_argument('--gate-use-qlora', action='store_true')
    parser.add_argument('--expert-model-name', type=str, default='resnet')
    parser.add_argument('--expert-model-size', type=str, default='small')
    parser.add_argument('--expert-use-lora', action='store_true')
    parser.add_argument('--expert-use-qlora', action='store_true')
    parser.add_argument('--lora-r', type=int, default=16, help="Rank for LoRA (must match training)")
    
    # Eval params
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for evaluation")
    parser.add_argument('--num-workers', type=int, default=4, help="Dataloader workers")
    
    args = parser.parse_args()
    main(args)