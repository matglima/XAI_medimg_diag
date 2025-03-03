# train.py
import os
import argparse
import logging
import torch
import optuna
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from dataloader import RetinaDataset, get_stratified_splits
from models import create_model
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_f1 = 0
        self.early_stop_counter = 0
        
        # Load data
        self.labels_df = self._load_labels()
        self.splits = get_stratified_splits(
            self.labels_df, args.target_label, 
            test_size=args.test_size, val_size=args.val_size
        )
        
        # Initialize model
        self.model = create_model(
            args.model_name,
            model_size=args.model_size,
            pretrained=args.pretrained
        ).to(self.device)
        
        # Add dropout if specified
        if args.dropout > 0:
            self._add_dropout(args.dropout)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max', patience=2
        )

    def _load_labels(self):
        labels_df = pd.read_csv(self.args.labels_path)
        labels_df['image_id'] = labels_df['image_id'].astype(str)
        return labels_df

    def _add_dropout(self, p):
        """Add dropout layers to the model architecture"""
        if 'resnet' in self.args.model_name:
            self.model.model.fc = nn.Sequential(
                nn.Dropout(p),
                self.model.model.fc
            )
        elif 'efficientnet' in self.args.model_name:
            self.model.model.classifier = nn.Sequential(
                nn.Dropout(p),
                self.model.model.classifier
            )
        # Add similar modifications for other architectures

    def _create_dataloaders(self, batch_size):
        transform = self.model.transform
        train_dataset = RetinaDataset(
            self.labels_df, self.args.image_dir,
            self.args.target_label, transform['train'],
            self.splits['train']
        )
        val_dataset = RetinaDataset(
            self.labels_df, self.args.image_dir,
            self.args.target_label, transform['val'],
            self.splits['val']
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        return train_loader, val_loader

    def _calculate_class_weights(self, targets, preds):
        """Dynamic class weighting based on F1 score"""
        targets_np = targets.detach().cpu().numpy()
        preds_np = torch.sigmoid(preds).detach().cpu().numpy()
        
        # Add zero_division parameter to handle the warning
        pos_weight = (1 - f1_score(targets_np, np.round(preds_np), zero_division=0))
        neg_weight = 1 - pos_weight
        return torch.tensor([neg_weight, pos_weight], device=self.device)

    def _train_epoch(self, train_loader, focal_loss):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []

        for images, targets in train_loader:
            images = images.to(self.device)
            targets = targets.float().to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images).squeeze(1)
            
            # Calculate dynamic class weights
            class_weights = self._calculate_class_weights(targets, outputs)
            loss = focal_loss(outputs, targets) * class_weights[1]
            
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs).detach().cpu())
            all_targets.append(targets.cpu())

        avg_loss = total_loss / len(train_loader)
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        return avg_loss, all_preds, all_targets

    def _validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.float().to(self.device)
                
                outputs = self.model(images).squeeze(1)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    outputs, targets
                )
                
                total_loss += loss.item()
                all_preds.append(torch.sigmoid(outputs).cpu())
                all_targets.append(targets.cpu())

        avg_loss = total_loss / len(val_loader)
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        
        return avg_loss, all_preds, all_targets

    def _early_stopping(self, val_f1):
        if val_f1 > self.best_f1:
            self.best_f1 = val_f1
            self.early_stop_counter = 0
            torch.save(self.model.state_dict(), self.args.model_name + '_best_model.pth')
        else:
            self.early_stop_counter += 1
            
        if self.early_stop_counter >= self.args.early_stop_patience:
            return True
        return False

    def train(self):
        logger.info(f"Starting training with device: {self.device}")
        logger.info(f"Model: {self.args.model_name}, Size: {self.args.model_size}")
        logger.info(f"Learning rate: {self.args.lr}, Batch size: {self.args.batch_size}")
        
        focal_loss = FocalLoss(alpha=self.args.alpha, gamma=self.args.gamma)
        train_loader, val_loader = self._create_dataloaders(self.args.batch_size)
        
        logger.info(f"Train dataset size: {len(train_loader.dataset)}")
        logger.info(f"Validation dataset size: {len(val_loader.dataset)}")

        for epoch in range(self.args.epochs):
            logger.info(f"\nEpoch {epoch+1}/{self.args.epochs}")
            logger.info("-" * 50)
            
            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Current learning rate: {current_lr:.2e}")
            
            # Training phase
            logger.info("Training phase:")
            train_loss, train_preds, train_targets = self._train_epoch(train_loader, focal_loss)
            
            # Validation phase
            logger.info("Validation phase:")
            val_loss, val_preds, val_targets = self._validate(val_loader)

            # Calculate metrics with zero_division parameter
            train_f1 = f1_score(train_targets, np.round(train_preds), zero_division=1.0)
            val_f1 = f1_score(val_targets, np.round(val_preds), zero_division=1.0)
            
            # Only calculate AUC if there are both classes present
            if len(np.unique(val_targets)) > 1:
                val_auc = roc_auc_score(val_targets, val_preds)
                logger.info(
                    f"Results:\n"
                    f"  Train Loss: {train_loss:.4f}\n"
                    f"  Val Loss: {val_loss:.4f}\n"
                    f"  Train F1: {train_f1:.4f}\n"
                    f"  Val F1: {val_f1:.4f}\n"
                    f"  Val AUC: {val_auc:.4f}"
                )
            else:
                logger.info(
                    f"Results:\n"
                    f"  Train Loss: {train_loss:.4f}\n"
                    f"  Val Loss: {val_loss:.4f}\n"
                    f"  Train F1: {train_f1:.4f}\n"
                    f"  Val F1: {val_f1:.4f}\n"
                    f"  Val AUC: N/A (requires both classes to be present)"
                )

            self.scheduler.step(val_f1)

            if self._early_stopping(val_f1):
                logger.info("\nEarly stopping triggered! Best validation F1: {:.4f}".format(self.best_f1))
                break

        # Final summary
        logger.info("\nTraining completed!")
        logger.info(f"Best validation F1: {self.best_f1:.4f}")
        logger.info("Loading best model weights...")
        self.model.load_state_dict(torch.load('best_model.pth'))
        return self.best_f1

def objective(trial, args):
    # Hyperparameter suggestions
    args.lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    args.dropout = trial.suggest_float('dropout', 0.0, 0.5)
    args.batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    args.alpha = trial.suggest_float('alpha', 0.1, 0.9)  # Focal loss alpha
    args.gamma = trial.suggest_float('gamma', 0.5, 3.0)  # Focal loss gamma
    
    trainer = Trainer(args)
    val_f1 = trainer.train()
    return val_f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retinal Disease Classification Training')
    # Data parameters
    parser.add_argument('--labels_path', type=str, required=True, default='data/labels_brset.csv')
    parser.add_argument('--image_dir', type=str, required=True, default='data/fundus_photos')
    parser.add_argument('--target_label', type=str, required=True, default='diabetic_retinopathy')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, required=True, default='resnet',
                       choices=['convnext', 'efficientnet', 'vit', 'swin', 'resnet'])
    parser.add_argument('--model_size', type=str, default='small')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=0.25,
                       help='Alpha parameter for focal loss')
    parser.add_argument('--gamma', type=float, default=2.0,
                       help='Gamma parameter for focal loss')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--early_stop_patience', type=int, default=5)
    
    # Optuna parameters
    parser.add_argument('--optuna_trials', type=int, default=0,
                       help='Number of Optuna optimization trials (0 to disable)')
    
    args = parser.parse_args()

    if args.optuna_trials > 0:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, args), n_trials=args.optuna_trials)
        
        logger.info("Best trial:")
        trial = study.best_trial
        logger.info(f"Value: {trial.value}")
        logger.info("Params:")
        for key, value in trial.params.items():
            logger.info(f"    {key}: {value}")
    else:
        trainer = Trainer(args)
        trainer.train()