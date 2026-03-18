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
from experiment_utils import save_run_manifest, set_global_seed
from moe_model import HybridMoE
from config import BRSET_LABELS

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_auc_scores(targets, preds):
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
    return auc_scores


def apply_thresholds(preds, thresholds):
    threshold_array = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    return (preds >= threshold_array).astype(np.int32)


def summarize_metrics(targets, preds, thresholds, pathology_list):
    preds_rounded = apply_thresholds(preds, thresholds)
    auc_scores = compute_auc_scores(targets, preds)
    f1_macro = f1_score(targets, preds_rounded, average='macro', zero_division=0)
    auc_macro = np.nanmean(auc_scores)

    per_class_rows = []
    for i, pathology in enumerate(pathology_list):
        class_auc = auc_scores[i] if not np.isnan(auc_scores[i]) else 0.0
        class_f1 = f1_score(targets[:, i], preds_rounded[:, i], zero_division=0)
        per_class_rows.append({
            'pathology': pathology,
            'auc': float(class_auc),
            'f1': float(class_f1),
            'threshold': float(thresholds[i]),
        })

    metrics_dict = {
        'final_auc_macro': float(auc_macro),
        'final_f1_macro': float(f1_macro),
        'threshold_mean': float(np.mean(thresholds)),
    }
    for row in per_class_rows:
        pathology = row['pathology']
        metrics_dict[f'test_auc_{pathology}'] = row['auc']
        metrics_dict[f'test_f1_{pathology}'] = row['f1']
        metrics_dict[f'threshold_{pathology}'] = row['threshold']

    return metrics_dict, pd.DataFrame(per_class_rows), preds_rounded, auc_scores


def tune_thresholds(targets, preds, pathology_list, threshold_values):
    thresholds = []
    rows = []

    for i, pathology in enumerate(pathology_list):
        best_threshold = 0.5
        best_score = -1.0
        best_positive_rate = 0.0

        for threshold in threshold_values:
            binary_preds = (preds[:, i] >= threshold).astype(np.int32)
            score = f1_score(targets[:, i], binary_preds, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
                best_positive_rate = float(binary_preds.mean())

        thresholds.append(best_threshold)
        rows.append({
            'pathology': pathology,
            'threshold': best_threshold,
            'validation_f1': float(best_score),
            'predicted_positive_rate': best_positive_rate,
        })

    return np.asarray(thresholds, dtype=np.float32), pd.DataFrame(rows)


def get_metrics_report(targets, preds, pathology_list, thresholds, title="FINAL MODEL EVALUATION REPORT"):
    """Generates a detailed, multi-line string report AND a metrics dictionary."""
    report_lines = []
    metrics_dict, per_class_df, preds_rounded, auc_scores = summarize_metrics(
        targets, preds, thresholds, pathology_list
    )
    f1_macro = metrics_dict['final_f1_macro']
    auc_macro = metrics_dict['final_auc_macro']

    report_lines.append(f"\n--- {title} ---")
    report_lines.append(f"\nOverall Metrics (Macro Average):")
    report_lines.append(f"  Macro F1-Score: {f1_macro:.4f}")
    report_lines.append(f"  Macro AUC:      {auc_macro:.4f}")
    report_lines.append(f"  Mean Threshold: {metrics_dict['threshold_mean']:.4f}")
    
    report_lines.append("\nPer-Class Metrics:")
    report_lines.append("=" * 78)
    report_lines.append(f"{'Pathology':<35} | {'Threshold':<10} | {'AUC':<7} | {'F1-Score':<10}")
    report_lines.append("-" * 78)

    for i, row in enumerate(per_class_df.to_dict(orient='records')):
        class_auc = auc_scores[i] if not np.isnan(auc_scores[i]) else 0.0
        auc_str = f"{class_auc:<7.4f}" if not np.isnan(auc_scores[i]) else "N/A    "
        report_lines.append(
            f"{row['pathology']:<35} | {row['threshold']:<10.4f} | {auc_str} | {row['f1']:<10.4f}"
        )
    
    report_lines.append("=" * 78)
    class_report = classification_report(targets, preds_rounded, target_names=pathology_list, zero_division=0)
    report_lines.append(class_report)
    report_lines.append("--- END OF REPORT ---")
    
    return "\n".join(report_lines), metrics_dict, per_class_df


def detect_subgroup_columns(dataframe, explicit_columns=None):
    if explicit_columns:
        selected = [column.strip() for column in explicit_columns.split(',') if column.strip()]
        return [column for column in selected if column in dataframe.columns]

    candidate_groups = [
        ['sex', 'patient_sex', 'gender'],
        ['camera', 'camera_model', 'camera_type', 'camera_device', 'camera_id', 'device'],
        ['diabetes', 'diabetes_history', 'diabetes_diagnosis', 'has_diabetes_history', 'patient_diabetes'],
    ]
    detected = []
    for candidates in candidate_groups:
        for column in candidates:
            if column in dataframe.columns:
                detected.append(column)
                break
    return detected


def detect_age_column(dataframe, explicit_age_column=None):
    if explicit_age_column and explicit_age_column in dataframe.columns:
        return explicit_age_column

    for column in ['patient_age', 'age', 'age_years']:
        if column in dataframe.columns:
            return column
    return None


def compute_subgroup_metrics(predictions_df, pathology_list, thresholds, overall_metrics, subgroup_columns, age_column,
                             age_bins, min_subgroup_size):
    rows = []
    target_columns = [f'target_{pathology}' for pathology in pathology_list]
    pred_columns = [f'pred_{pathology}' for pathology in pathology_list]

    def add_group(group_name, group_value, group_df):
        if len(group_df) < min_subgroup_size:
            return

        targets = group_df[target_columns].to_numpy(dtype=np.float32)
        preds = group_df[pred_columns].to_numpy(dtype=np.float32)
        metrics_dict, _, _, _ = summarize_metrics(targets, preds, thresholds, pathology_list)
        rows.append({
            'group_name': group_name,
            'group_value': str(group_value),
            'n': int(len(group_df)),
            'macro_auc': float(metrics_dict['final_auc_macro']),
            'macro_f1': float(metrics_dict['final_f1_macro']),
            'auc_gap_vs_overall': float(metrics_dict['final_auc_macro'] - overall_metrics['final_auc_macro']),
            'f1_gap_vs_overall': float(metrics_dict['final_f1_macro'] - overall_metrics['final_f1_macro']),
        })

    for column in subgroup_columns:
        normalized = predictions_df[column].fillna('missing').astype(str)
        for value in sorted(normalized.unique()):
            group_df = predictions_df[normalized == value]
            add_group(column, value, group_df)

    if age_column and age_column in predictions_df.columns:
        age_values = pd.to_numeric(predictions_df[age_column], errors='coerce')
        labels = []
        for start, end in zip(age_bins[:-1], age_bins[1:]):
            if end >= age_bins[-1]:
                labels.append(f'{int(start)}+')
            else:
                labels.append(f'{int(start)}-{int(end) - 1}')
        age_groups = pd.cut(age_values, bins=age_bins, labels=labels, right=False, include_lowest=True)
        for value in age_groups.dropna().unique():
            group_df = predictions_df[age_groups == value]
            add_group('age_bin', value, group_df)

    return pd.DataFrame(rows)


def build_threshold_grid(threshold_min, threshold_max, threshold_step):
    threshold_values = np.arange(threshold_min, threshold_max + (threshold_step / 2.0), threshold_step)
    threshold_values = np.clip(threshold_values, 0.0, 1.0)
    threshold_values = np.unique(np.round(threshold_values, 4))
    return threshold_values[(threshold_values > 0.0) & (threshold_values < 1.0)]


def collect_predictions(model, loader, device):
    all_preds, all_targets = [], []
    all_gate_probs, all_expert_probs, all_image_ids = [], [], []

    with torch.no_grad():
        for images, targets, image_ids in tqdm(loader, desc="Evaluating on Split"):
            images = images.to(device, non_blocking=True)
            outputs = model.forward_with_components(images)
            final_logits = outputs['final_logits']
            all_preds.append(torch.sigmoid(final_logits).cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_gate_probs.append(torch.sigmoid(outputs['gate_logits']).cpu().numpy())
            all_expert_probs.append(torch.sigmoid(outputs['expert_logits']).cpu().numpy())
            all_image_ids.extend(list(image_ids))

    return {
        'preds': np.concatenate(all_preds, axis=0),
        'targets': np.concatenate(all_targets, axis=0),
        'gate_probs': np.concatenate(all_gate_probs, axis=0),
        'expert_probs': np.concatenate(all_expert_probs, axis=0),
        'image_ids': all_image_ids,
    }

def main(args):
    set_global_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    output_dir = os.path.dirname(args.model_path) or "."
    save_run_manifest(output_dir, "evaluation_run_manifest.json", args, extra={'phase': 'evaluation'})
    
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
        device=device,
        fusion_strategy=args.fusion_strategy,
        top_k=args.top_k,
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
    
    # --- 4. Load Validation/Test Data ---
    logger.info("Loading validation and test data...")
    labels_df = pd.read_csv(args.labels_path)
    labels_df['image_id'] = labels_df['image_id'].astype(str)
    splits = get_random_splits(
        labels_df,
        random_state=args.seed,
        split_manifest_path=args.split_manifest,
    ) 

    val_dataset = MultiLabelRetinaDataset(
        labels_df, args.image_dir, PATHOLOGIES, val_transform, splits['val'], return_image_id=True
    )
    test_dataset = MultiLabelRetinaDataset(
        labels_df, args.image_dir, PATHOLOGIES, val_transform, splits['test'], return_image_id=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # --- 5. Run Validation/Test Inference ---
    logger.info("Collecting validation predictions for threshold tuning...")
    val_outputs = collect_predictions(model, val_loader, device)

    logger.info("Collecting test predictions for final reporting...")
    test_outputs = collect_predictions(model, test_loader, device)

    if args.disable_threshold_tuning:
        thresholds = np.full(len(PATHOLOGIES), 0.5, dtype=np.float32)
        threshold_df = pd.DataFrame({
            'pathology': PATHOLOGIES,
            'threshold': thresholds,
            'validation_f1': np.nan,
            'predicted_positive_rate': np.nan,
        })
        logger.info("Threshold tuning disabled. Using fixed threshold 0.5 for all classes.")
    else:
        threshold_grid = build_threshold_grid(args.threshold_min, args.threshold_max, args.threshold_step)
        thresholds, threshold_df = tune_thresholds(
            val_outputs['targets'],
            val_outputs['preds'],
            PATHOLOGIES,
            threshold_grid,
        )
        logger.info(
            "Tuned thresholds on validation split. Mean threshold: %.4f",
            float(np.mean(thresholds))
        )
    
    # --- 6. Generate Report ---
    report_str, metrics_dict, per_class_df = get_metrics_report(
        test_outputs['targets'],
        test_outputs['preds'],
        PATHOLOGIES,
        thresholds,
    )
    print(report_str) # Always print to console
    
    # Save report file
    report_path = os.path.join(output_dir, "final_evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report_str)
    logger.info(f"Report saved to {report_path}")

    per_class_path = os.path.join(output_dir, "final_per_class_metrics.csv")
    per_class_df.to_csv(per_class_path, index=False)
    logger.info(f"Per-class metrics saved to {per_class_path}")

    threshold_path = os.path.join(output_dir, "final_thresholds.csv")
    threshold_df.to_csv(threshold_path, index=False)
    logger.info(f"Thresholds saved to {threshold_path}")

    predictions_output = args.predictions_output or os.path.join(output_dir, "final_predictions.csv")
    predictions_df = pd.DataFrame({'image_id': test_outputs['image_ids']})
    for i, pathology in enumerate(PATHOLOGIES):
        predictions_df[f'target_{pathology}'] = test_outputs['targets'][:, i]
        predictions_df[f'pred_{pathology}'] = test_outputs['preds'][:, i]
        predictions_df[f'gate_{pathology}'] = test_outputs['gate_probs'][:, i]
        predictions_df[f'expert_{pathology}'] = test_outputs['expert_probs'][:, i]
        predictions_df[f'threshold_{pathology}'] = thresholds[i]
        predictions_df[f'pred_label_{pathology}'] = (test_outputs['preds'][:, i] >= thresholds[i]).astype(np.int32)

    metadata_columns = [column for column in detect_subgroup_columns(labels_df, args.subgroup_columns)]
    age_column = detect_age_column(labels_df, args.age_column)
    merge_columns = ['image_id'] + metadata_columns + ([age_column] if age_column else [])
    predictions_df = predictions_df.merge(
        labels_df[merge_columns].drop_duplicates(subset=['image_id']),
        on='image_id',
        how='left'
    )
    predictions_df.to_csv(predictions_output, index=False)
    logger.info(f"Predictions saved to {predictions_output}")

    subgroup_path = os.path.join(output_dir, "final_subgroup_metrics.csv")
    age_bins = [float(value.strip()) for value in args.age_bins.split(',') if value.strip()]
    subgroup_df = compute_subgroup_metrics(
        predictions_df,
        PATHOLOGIES,
        thresholds,
        metrics_dict,
        metadata_columns,
        age_column,
        age_bins,
        args.min_subgroup_size,
    )
    if subgroup_df.empty:
        logger.warning("No subgroup metrics were generated. Check metadata column availability and subgroup sizes.")
    else:
        subgroup_df.to_csv(subgroup_path, index=False)
        logger.info(f"Subgroup metrics saved to {subgroup_path}")
    
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
                    mlflow.log_artifact(per_class_path)
                    mlflow.log_artifact(threshold_path)
                    mlflow.log_artifact(predictions_output)
                    if os.path.exists(subgroup_path):
                        mlflow.log_artifact(subgroup_path)
                    
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
    parser.add_argument('--fusion-strategy', type=str, default='additive')
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-manifest', type=str, default=None)
    parser.add_argument('--predictions-output', type=str, default=None)
    parser.add_argument('--disable-threshold-tuning', action='store_true')
    parser.add_argument('--threshold-min', type=float, default=0.05)
    parser.add_argument('--threshold-max', type=float, default=0.95)
    parser.add_argument('--threshold-step', type=float, default=0.05)
    parser.add_argument('--subgroup-columns', type=str, default=None,
                        help="Comma-separated metadata columns for subgroup analysis; defaults to auto-detect.")
    parser.add_argument('--age-column', type=str, default=None,
                        help="Metadata column to use for age-bin subgroup analysis.")
    parser.add_argument('--age-bins', type=str, default='0,45,65,200',
                        help="Comma-separated age bin edges for subgroup analysis.")
    parser.add_argument('--min-subgroup-size', type=int, default=25,
                        help="Minimum subgroup size required to report subgroup metrics.")
    parser.add_argument('--use-mlflow', action='store_true', help="Enable MLflow logging")
    
    args = parser.parse_args()
    main(args)
