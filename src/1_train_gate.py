# -----------------------------------------------------------------
# File: 1_train_gate.py
# -----------------------------------------------------------------
# Description:
# Phase 1A: Trains the Multi-Label "Gate" model.
# NOW with robust AUC and async data transfer.
# -----------------------------------------------------------------

import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from tqdm import tqdm
import os
import warnings # <-- IMPORT WARNINGS

from dataloader import MultiLabelRetinaDataset, get_random_splits, train_transform, val_transform
from models import create_model, get_optimizer

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define all pathologies (must match CSV headers)
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

# --- Helper Functions ---

def run_epoch(model, loader, criterion, optimizer, device, is_training):
    model.train() if is_training else model.eval()
    
    total_loss = 0
    all_preds = []
    all_targets = []

    progress_bar = tqdm(loader, desc="Training" if is_training else "Validation")
    for images, targets in progress_bar:
        # --- START FIX: Use non_blocking=True ---
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # --- END FIX ---

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        
        progress_bar.set_postfix(loss=total_loss / (len(all_preds)))

    avg_loss = total_loss / len(loader)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return avg_loss, all_preds, all_targets

def get_metrics(targets, preds):
    # Calculate metrics for multi-label classification
    preds_rounded = np.round(preds)
    
    # --- START FIX: Robust AUC Calculation ---
    auc_scores = []
    num_classes = targets.shape[1]
    
    # Suppress warnings for classes with only one label
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(num_classes):
            try:
                # Check if both classes are present
                if len(np.unique(targets[:, i])) > 1:
                    class_auc = roc_auc_score(targets[:, i], preds[:, i])
                    auc_scores.append(class_auc)
                else:
                    # Only one class present, append 0.5 (neutral) or np.nan
                    auc_scores.append(0.5) 
            except ValueError:
                auc_scores.append(0.5) # Fallback
                
    # Calculate macro average from the list of scores
    auc_macro = np.nanmean(auc_scores) # Use nanmean in case we appended np.nan
    # --- END FIX ---

    f1_macro = f1_score(targets, preds_rounded, average='macro', zero_division=0)
    acc = accuracy_score(targets, preds_rounded) # This is subset accuracy (very strict)
    
    return {'f1_macro': f1_macro, 'auc_macro': auc_macro, 'accuracy': acc}

# --- Main Training Function ---

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # --- Load Data ---
    logger.info("Loading labels...")
    labels_df = pd.read_csv(args.labels_path)
    labels_df['image_id'] = labels_df['image_id'].astype(str)
    
    # Use random splits for multi-label
    logger.info("Creating random splits...")
    splits = get_random_splits(labels_df, test_size=0.2, val_size=0.1)
    
    # Note: The transforms are now passed from the global scope
    # The Dataset will pre-cache using pre_transform
    # and apply train_transform/val_transform on-the-fly
    train_dataset = MultiLabelRetinaDataset(
        labels_df, args.image_dir, PATHOLOGIES, train_transform, splits['train']
    )
    val_dataset = MultiLabelRetinaDataset(
        labels_df, args.image_dir, PATHOLOGIES, val_transform, splits['val']
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True # pin_memory=True is important
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # --- Initialize Model ---
    logger.info(f"Creating model: {args.model_name} {args.model_size}")
    model = create_model(
        model_name=args.model_name,
        model_size=args.model_size,
        pretrained=not args.no_pretrained,
        num_classes=len(PATHOLOGIES),
        use_lora=args.use_lora,
        use_qlora=args.use_qlora
    ).to(device)

    # --- Loss and Optimizer ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = get_optimizer(model, args.lr, args.use_qlora)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1)

    # --- Training Loop ---
    best_val_auc = 0
    epochs_no_improve = 0
    
    os.makedirs(args.output_dir, exist_ok=True)
    model_save_path = os.path.join(args.output_dir, "gate_best_model.pth")

    for epoch in range(args.epochs):
        logger.info(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        train_loss, train_preds, train_targets = run_epoch(
            model, train_loader, criterion, optimizer, device, is_training=True
        )
        train_metrics = get_metrics(train_targets, train_preds)
        logger.info(f"Train Loss: {train_loss:.4f} | Train AUC: {train_metrics['auc_macro']:.4f} | Train F1: {train_metrics['f1_macro']:.4f}")

        val_loss, val_preds, val_targets = run_epoch(
            model, val_loader, criterion, None, device, is_training=False
        )
        val_metrics = get_metrics(val_targets, val_preds)
        logger.info(f"Val Loss: {val_loss:.4f} | Val AUC: {val_metrics['auc_macro']:.4f} | Val F1: {val_metrics['f1_macro']:.4f}")
        
        scheduler.step(val_metrics['auc_macro'])
        
        # --- Save Best Model (based on Val AUC) ---
        if val_metrics['auc_macro'] > best_val_auc:
            logger.info(f"Val AUC improved ({best_val_auc:.4f} --> {val_metrics['auc_macro']:.4f}). Saving model...")
            best_val_auc = val_metrics['auc_macro']
            if args.use_lora:
                # Save only the LoRA adapters
                model.model.save_pretrained(args.output_dir)
            else:
                # Save the full model
                torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            logger.info(f"Val AUC did not improve. Counter: {epochs_no_improve}/{args.patience}")

        if epochs_no_improve >= args.patience:
            logger.info("Early stopping triggered.")
            break
            
    logger.info(f"Training complete. Best Val AUC: {best_val_auc:.4f}")
    if args.use_lora:
        logger.info(f"Best LoRA adapters saved in: {args.output_dir}")
    else:
        logger.info(f"Best model saved to: {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Label Gate Model")
    
    # Data params
    parser.add_argument('--labels-path', type=str, required=True, help="Path to labels.csv")
    parser.add_argument('--image-dir', type=str, required=True, help="Directory with fundus images")
    parser.add_argument('--output-dir', type=str, default="checkpoints/gate", help="Directory to save checkpoints")
    
    # Model params
    parser.add_argument('--model-name', type=str, default='resnet', choices=['convnext', 'efficientnet', 'vit', 'swin', 'resnet'])
    parser.add_argument('--model-size', type=str, default='small', help="e.g., 'small', 'base', 'large'")
    parser.add_argument('--no-pretrained', action='store_true', help="Do not use pretrained weights")

    # Training params
    parser.add_argument('--epochs', type=int, default=50, help="Max number of epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    
    # --- START FIX: Corrected .add.argument to .add_argument ---
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--num-workers', type=int, default=4, help="Dataloader workers")
    # --- END FIX ---
    
    # LoRA / Q-LoRA params
    parser.add_argument('--use-lora', action='store_true', help="Enable LoRA fine-tuning")
    parser.add_argument('--use-qlora', action='store_true', help="Enable Q-LoRA (4-bit) fine-tuning")
    
    args = parser.parse_args()
    
    main(args)