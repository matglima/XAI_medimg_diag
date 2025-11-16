# -----------------------------------------------------------------
# File: 2_train_experts.py
# -----------------------------------------------------------------
# Description:
# Phase 1B: Trains the 14 Binary "Expert" models.
# This is a refactor of your original train.py.
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

from dataloader import RetinaDataset, get_stratified_splits, train_transform, val_transform
from models import create_model, get_optimizer

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets.float(), reduction='none'
        )
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# --- Main Trainer Class ---

class ExpertTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_label = args.target_label
        
        os.makedirs(args.output_dir, exist_ok=True)
        self.model_save_path = os.path.join(args.output_dir, f"{self.target_label}_best_model.pth")
        self.lora_save_path = os.path.join(args.output_dir, f"{self.target_label}_lora_adapters")
        
        logger.info(f"--- Training Expert for: {self.target_label} ---")
        logger.info(f"Using device: {self.device}")

        # --- Load Data ---
        logger.info("Loading labels...")
        labels_df = pd.read_csv(args.labels_path)
        labels_df['image_id'] = labels_df['image_id'].astype(str)
        
        logger.info("Creating stratified splits...")
        splits = get_stratified_splits(labels_df, self.target_label)
        
        train_dataset = RetinaDataset(
            labels_df, args.image_dir, self.target_label, train_transform, splits['train']
        )
        val_dataset = RetinaDataset(
            labels_df, args.image_dir, self.target_label, val_transform, splits['val']
        )
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, 
            num_workers=args.num_workers, pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, 
            num_workers=args.num_workers, pin_memory=True
        )
        
        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        # --- Initialize Model ---
        logger.info(f"Creating model: {args.model_name} {args.model_size}")
        self.model = create_model(
            model_name=args.model_name,
            model_size=args.model_size,
            pretrained=not args.no_pretrained,
            num_classes=1, # Binary expert
            use_lora=args.use_lora,
            use_qlora=args.use_qlora
        ).to(self.device)

        # --- Loss and Optimizer ---
        self.criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma)
        self.optimizer = get_optimizer(self.model, args.lr, args.use_qlora)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=3, factor=0.1)

    def _run_epoch(self, loader, is_training):
        self.model.train() if is_training else self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_targets = []

        progress_bar = tqdm(loader, desc=f"Training {self.target_label}" if is_training else f"Validation {self.target_label}")
        for images, targets in progress_bar:
            images = images.to(self.device)
            targets = targets.to(self.device).squeeze() # Ensure 1D

            with torch.set_grad_enabled(is_training):
                outputs = self.model(images).squeeze() # Output shape (batch_size)
                loss = self.criterion(outputs, targets)
                
                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            progress_bar.set_postfix(loss=total_loss / (len(all_preds)))

        avg_loss = total_loss / len(loader)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        return avg_loss, all_preds, all_targets

    def train(self):
        best_val_auc = 0
        epochs_no_improve = 0
        
        for epoch in range(self.args.epochs):
            logger.info(f"\n--- Epoch {epoch+1}/{self.args.epochs} ---")
            
            train_loss, train_preds, train_targets = self._run_epoch(self.train_loader, is_training=True)
            train_auc = roc_auc_score(train_targets, train_preds)
            train_f1 = f1_score(train_targets, np.round(train_preds), zero_division=0)
            logger.info(f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} | Train F1: {train_f1:.4f}")

            val_loss, val_preds, val_targets = self._run_epoch(self.val_loader, is_training=False)
            val_auc = roc_auc_score(val_targets, val_preds)
            val_f1 = f1_score(val_targets, np.round(val_preds), zero_division=0)
            logger.info(f"Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")
            
            self.scheduler.step(val_auc)
            
            # --- Save Best Model (based on Val AUC) ---
            if val_auc > best_val_auc:
                logger.info(f"Val AUC improved ({best_val_auc:.4f} --> {val_auc:.4f}). Saving model...")
                best_val_auc = val_auc
                if self.args.use_lora:
                    self.model.model.save_pretrained(self.lora_save_path)
                else:
                    torch.save(self.model.state_dict(), self.model_save_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                logger.info(f"Val AUC did not improve. Counter: {epochs_no_improve}/{self.args.patience}")

            if epochs_no_improve >= self.args.patience:
                logger.info("Early stopping triggered.")
                break
                
        logger.info(f"Training complete for {self.target_label}. Best Val AUC: {best_val_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Binary Expert Model")
    
    # Data params
    parser.add_argument('--labels-path', type=str, required=True, help="Path to labels.csv")
    parser.add_argument('--image-dir', type=str, required=True, help="Directory with fundus images")
    parser.add_argument('--target-label', type=str, required=True, help="Specific pathology to train expert for")
    parser.add_argument('--output-dir', type=str, default="checkpoints/experts", help="Directory to save checkpoints")
    
    # Model params
    parser.add_argument('--model-name', type=str, default='resnet', choices=['convnext', 'efficientnet', 'vit', 'swin', 'resnet'])
    parser.add_argument('--model-size', type=str, default='small', help="e.g., 'small', 'base', 'large'")
    parser.add_argument('--no-pretrained', action='store_true', help="Do not use pretrained weights")

    # Training params
    parser.add_argument('--epochs', type=int, default=50, help="Max number of epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--num-workers', type=int, default=4, help="Dataloader workers")
    
    # Loss params
    parser.add_argument('--alpha', type=float, default=0.25, help='Alpha parameter for focal loss')
    parser.add_argument('--gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
    
    # LoRA / Q-LoRA params
    parser.add_argument('--use-lora', action='store_true', help="Enable LoRA fine-tuning")
    parser.add_argument('--use-qlora', action='store_true', help="Enable Q-LoRA (4-bit) fine-tuning")
    
    args = parser.parse_args()
    trainer = ExpertTrainer(args)
    trainer.train()