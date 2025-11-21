# -----------------------------------------------------------------
# File: evaluate.py
# -----------------------------------------------------------------
# Description:
# Phase 4: Evaluates the final, calibrated MoE model and
# logs a final report to MLflow.
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

# --- MLflow Import ---
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: 'mlflow' package not found. MLflow logging will be disabled.")

from dataloader import MultiLabelRetinaDataset, get_random_splits, val_transform
from moe_model import HybridMoE
from config import BRSET_LABELS

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_metrics_report(targets, preds, pathology_list):
    """Generates a detailed, multi-line string report AND a metrics dictionary."""
    preds_rounded = np.round(preds)
    report_lines = []
    metrics_dict = {} # For MLflow
    
    # Overall Metrics
    f1_macro = f1_score(targets, preds_rounded, average='macro', zero_division=0)
    
    auc_scores = []
    num_classes = targets.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(num_classes):
            try:
                if len(np.unique(targets[:, i])) > 1:
                    auc_scores.append(roc_auc_score(targets[:, i], preds[:, i]))
                else:
                    auc_scores.append(np.nan)
            except ValueError:
                auc_scores.append(np.nan)
                
    auc_macro = np.nanmean(auc_scores)
    
    metrics_dict['final_auc_macro'] = auc_macro
    metrics_dict['final_f1_macro'] = f1_macro
    
    report_lines.append("\n--- FINAL MODEL EVALUATION REPORT ---")
    report_lines.append(f"\nOverall Metrics (Macro Average):")
    report_lines.append(f"  Macro F1-Score: {f1_macro:.4f}")
    report_lines.append(f"  Macro AUC:      {auc_macro:.4f}")
    
    report_lines.append("\nPer-Class Metrics:")
    report_lines.append("=" * 60)
    report_lines.append(f"{'Pathology':<35} | {'AUC':<7} | {'F1-Score':<10}")
    report_lines.append("-" * 60)
    
    for i, pathology in enumerate(pathology_list):
        class_auc = auc_scores[i] if not np.isnan(auc_scores[i]) else 0.0
        class_f1 = f1_score(targets[:, i], preds_rounded[:, i], zero_division=0)
        auc_str = f"{class_auc:<7.4f}" if not np.isnan(auc_scores[i]) else "N/A    "
        report_lines.append(f"{pathology:<35} | {auc_str} | {class_f1:<10.4f}")
        
        # Add per-class metrics for MLflow
        metrics_dict[f'test_auc_{pathology}'] = class_auc
        metrics_dict[f'test_f1_{pathology}'] = class_f1
    
    report_lines.append("=" * 60)
    class_report = classification_report(targets, preds_rounded, target_names=pathology_list, zero_division=0)
    report_lines.append(class_report)
    report_lines.append("--- END OF REPORT ---")
    
    return "\n".join(report_lines), metrics_dict

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    PATHOLOGIES = BRSET_LABELS
    
    # --- 1. Define Model Configs (must match training) ---
    # We set use_lora=False because the model saved by 3_calibrate_moe.py
    # is *already merged* and saved as a standard state_dict.
    gate_config = {
        'model_name': args.gate_model_name,
        'model_size': args.gate_model_size,
        'use_lora': False, 'use_qlora': False, 'lora_r': args.lora_r
    }
    expert_config = {
        'model_name': args.expert_model_name,
        'model_size': args.expert_model_size,
        'use_lora': False, 'use_qlora': False, 'lora_r': args.lora_r
    }

    # --- 2. Initialize MoE Structure ---
    logger.info("Initializing HybridMoE structure for evaluation...")
    # 1. Load the state dictionary
    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
    # The saved state_dict already has merged weights, so the model must be initialized
    # as a standard (non-PEFT) model to accept all feature layer keys.
    model = HybridMoE(
        gate_config=dict(
            model_name=args.gate_model_name,
            model_size=args.gate_model_size,
            # REMOVED: use_lora=args.gate_use_lora, use_qlora=args.gate_use_qlora
            use_lora=False, # <-- NEW
            use_qlora=False, # <-- NEW
            lora_r=args.lora_r
        ),
        expert_config=dict(
            model_name=args.expert_model_name,
            model_size=args.expert_model_size,
            # REMOVED: use_lora=args.expert_use_lora, use_qlora=args.expert_use_qlora
            use_lora=False, # <-- NEW
            use_qlora=False, # <-- NEW
            lora_r=args.lora_r
        ),
        pathology_list=PATHOLOGIES,
        device=device
    ).to(device)
    # --- 3. Load Final Calibrated Weights ---
    logger.info(f"Loading final calibrated model from: {args.model_path}")
    # --- START FIX: Strip redundant 'base_model.model' prefix ---
    new_state_dict = {}
    prefix_to_strip = "base_model.model."
    
    for k, v in state_dict.items():
        # Check for the primary PEFT-related prefix that exists in the saved file
        if prefix_to_strip in k:
            # We strip the full prefix, keeping the rest of the key (e.g., 'gate.model.features...')
            new_key = k.replace(prefix_to_strip, "")
            new_state_dict[new_key] = v
        else:
            # Keep the key as-is (e.g., final classifier head keys might not be prefixed)
            new_state_dict[k] = v

    # 2. Load the state dict using the cleaned keys
    # Use strict=False as a safeguard against any minor key discrepancies
    load_result = model.load_state_dict(new_state_dict, strict=False)
    
    # Optional: Log the result for confirmation (check if 'missing' is empty)
    if load_result.missing_keys:
        logger.warning(f"Still missing keys after cleanup: {load_result.missing_keys[:5]}...")
    else:
        logger.info("All model weights successfully loaded after prefix cleanup.")
    # --- END FIX ---
    model.to(device)
    model.eval()
    
    # --- 4. Load Test Data ---
    logger.info("Loading test data...")
    labels_df = pd.read_csv(args.labels_path)
    labels_df['image_id'] = labels_df['image_id'].astype(str)
    splits = get_random_splits(labels_df) 
    
    test_dataset = MultiLabelRetinaDataset(
        labels_df, args.image_dir, PATHOLOGIES, val_transform, splits['test']
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )
    logger.info(f"Test samples: {len(test_dataset)}")

    # --- 5. Run Evaluation ---
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Evaluating on Test Set"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # --- 6. Generate Report ---
    report_str, metrics_dict = get_metrics_report(all_targets, all_preds, PATHOLOGIES)
    print(report_str) # Always print to console
    
    # Save report file
    report_path = os.path.join(os.path.dirname(args.model_path), "final_evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report_str)
    logger.info(f"Report saved to {report_path}")
    
    # --- 7. Log to MLflow ---
    if args.use_mlflow:
        if not MLFLOW_AVAILABLE:
            logger.error("MLflow is not installed. Skipping MLflow logging.")
        else:
            try:
                mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))
                mlflow.set_experiment(os.environ.get('MLFLOW_EXPERIMENT_NAME'))
                
                with mlflow.start_run(run_name=args.run_name):
                    logger.info("Logging final metrics to MLflow...")
                    
                    # Log all CLI parameters
                    mlflow.log_params(vars(args))
                    
                    # Log all scalar metrics
                    mlflow.log_metrics(metrics_dict)
                    
                    # Log the report file as an artifact
                    mlflow.log_artifact(report_path)
                    
                    logger.info("MLflow logging complete.")
                    
            except Exception as e:
                logger.error(f"Could not log to MLflow: {e}")
                logger.warning("Metrics were printed/saved locally but not logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Final Hybrid MoE Model")
    
    # Add all arguments
    parser.add_argument('--labels-path', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--model-path', type=str, default="checkpoints/final_moe/moe_calibrated_final.pth")
    parser.add_argument('--run-name', type=str, default="Final_MoE_Evaluation")
    parser.add_argument('--gate-model-name', type=str, default='resnet')
    parser.add_argument('--gate-model-size', type=str, default='small')
    parser.add_argument('--expert-model-name', type=str, default='resnet')
    parser.add_argument('--expert-model-size', type=str, default='small')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--gate-use-lora', action='store_true')
    parser.add_argument('--gate-use-qlora', action='store_true')
    parser.add_argument('--expert-use-lora', action='store_true')
    parser.add_argument('--expert-use-qlora', action='store_true')
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--use-mlflow', action='store_true', help="Enable MLflow logging")
    
    args = parser.parse_args()
    main(args)