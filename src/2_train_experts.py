# -----------------------------------------------------------------
# File: 2_train_experts.py
# -----------------------------------------------------------------
# Description:
# Phase 1B: Trains the 14 Binary "Expert" models.
# Refactored to PyTorch Lightning for clean logging and MLflow.
# -----------------------------------------------------------------

import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
import warnings
import os

# --- Lightning Imports ---
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
# --- End Lightning Imports ---

# --- MLflow Autolog ---
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: 'mlflow' package not found. MLflow logging will be disabled.")

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
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss

# --- PyTorch Lightning Module ---

class ExpertModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.model = create_model(
            model_name=self.hparams.model_name,
            model_size=self.hparams.model_size,
            pretrained=not self.hparams.no_pretrained,
            num_classes=1, # Binary
            use_lora=self.hparams.use_lora,
            use_qlora=self.hparams.use_qlora,
            lora_r=self.hparams.lora_r
        )
        
        self.criterion = FocalLoss(alpha=self.hparams.alpha, gamma=self.hparams.gamma)
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images).squeeze()
        loss = self.criterion(outputs, targets.squeeze())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images).squeeze()
        loss = self.criterion(outputs, targets.squeeze())
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        preds = torch.sigmoid(outputs)
        self.validation_step_outputs.append({'preds': preds.detach().cpu(), 'targets': targets.detach().cpu()})
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).numpy()
        targets = torch.cat([x['targets'] for x in self.validation_step_outputs]).numpy()
        
        metrics = self._calculate_metrics(targets, preds)
        
        self.log('val_auc', metrics['auc'], prog_bar=True)
        self.log('val_f1', metrics['f1'])
        
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images).squeeze()
        loss = self.criterion(outputs, targets.squeeze())
        
        self.log('test_loss', loss, on_epoch=True)
        preds = torch.sigmoid(outputs)
        self.test_step_outputs.append({'preds': preds.detach().cpu(), 'targets': targets.detach().cpu()})

    def on_test_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.test_step_outputs]).numpy()
        targets = torch.cat([x['targets'] for x in self.test_step_outputs]).numpy()
        
        metrics = self._calculate_metrics(targets, preds)
        
        self.log('test_auc', metrics['auc'])
        self.log('test_f1', metrics['f1'])

        print(f"\n--- Expert '{self.hparams.target_label}' Test Metrics ---")
        print(f"Test AUC: {metrics['auc']:.4f}")
        print(f"Test F1:  {metrics['f1']:.4f}")

        if self.hparams.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                'test_auc': metrics['auc'],
                'test_f1': metrics['f1']
            })

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = get_optimizer(self.model, self.hparams.lr, self.hparams.use_qlora)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_auc",
                "mode": "max"
            },
        }

    def _calculate_metrics(self, targets, preds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                auc = roc_auc_score(targets, preds)
            except ValueError:
                auc = 0.5
            f1 = f1_score(targets, np.round(preds), zero_division=0)
        return {'f1': f1, 'auc': auc}

# --- DataModule for Lightning ---

class ExpertDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.labels_df = pd.read_csv(self.hparams.labels_path)
        self.labels_df['image_id'] = self.labels_df['image_id'].astype(str)
        self.splits = get_stratified_splits(self.labels_df, self.hparams.target_label)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = RetinaDataset(
                self.labels_df, self.hparams.image_dir, self.hparams.target_label, train_transform, self.splits['train']
            )
            self.val_dataset = RetinaDataset(
                self.labels_df, self.hparams.image_dir, self.hparams.target_label, val_transform, self.splits['val']
            )
        if stage == 'test' or stage is None:
            self.test_dataset = RetinaDataset(
                self.labels_df, self.hparams.image_dir, self.hparams.target_label, val_transform, self.splits['test']
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, 
            num_workers=self.hparams.num_workers, pin_memory=True, persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, 
            num_workers=self.hparams.num_workers, pin_memory=True, persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False, 
            num_workers=self.hparams.num_workers, pin_memory=True, persistent_workers=True
        )

# --- Main execution ---

def main(args):
    # --- 1. Init Data ---
    dm = ExpertDataModule(args)
    
    # --- 2. Init Model ---
    model = ExpertModule(args)
    
    # --- 3. Init Loggers and Callbacks ---
    callbacks = [
        EarlyStopping(monitor='val_auc', patience=args.patience, mode='max', verbose=True),
        TQDMProgressBar(refresh_rate=10)
    ]
    
    mlflow_logger = None
    
    if args.use_mlflow:
        if not MLFLOW_AVAILABLE:
            logger.error("MLflow is not installed. Disabling MLflow logging.")
            args.use_mlflow = False
        else:
            logger.info("Enabling MLflow autologging...")
            # MLflow Autolog will handle logging, params, and checkpoints
            mlflow.pytorch.autolog(
                log_models=True,
                checkpoint=True,
                checkpoint_monitor='val_auc', # Monitor val_auc for experts
                checkpoint_mode='max',
                checkpoint_save_best_only=True,
                checkpoint_save_freq='epoch',
                # Save to a *sub-directory* named after the expert
                checkpoint_dirpath=os.path.join(args.output_dir, f"{args.target_label}_checkpoints"),
                checkpoint_filename=f'{args.target_label}_best_model'
            )
            # We use a nested run for better organization
            mlf_logger = MLflowLogger(
                experiment_name=os.environ.get('MLFLOW_EXPERIMENT_NAME'),
                run_name=args.run_name,
                tracking_uri=os.environ.get('MLFLOW_TRACKING_URI')
            )

    if not args.use_mlflow:
        logger.info("MLflow is disabled. Using local ModelCheckpoint.")
        # Fallback to local checkpointing if MLflow is off
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.output_dir, # Save directly to the expert dir
            filename=f'{args.target_label}_best_model',
            monitor='val_auc',
            mode='max',
            save_top_k=1,
        )
        callbacks.append(checkpoint_callback)

    # --- 4. Init Trainer ---
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=mlflow_logger if args.use_mlflow else False,
        callbacks=callbacks,
        log_every_n_steps=10
    )
    
    # --- 5. Run Training ---
    logger.info(f"--- Starting Expert Training for: {args.target_label} ---")
    trainer.fit(model, datamodule=dm)
    
    # --- 6. Run Testing ---
    logger.info(f"--- Starting Expert Testing for: {args.target_label} ---")
    trainer.test(datamodule=dm, ckpt_path='best')
    
    logger.info(f"--- Expert Training & Testing Complete for: {args.target_label} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Binary Expert Model (Lightning)")
    
    # Add all arguments
    parser.add_argument('--labels-path', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--target-label', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default="checkpoints/experts")
    parser.add_argument('--run-name', type=str, default="Expert_Run")
    parser.add_argument('--model-name', type=str, default='resnet')
    parser.add_argument('--model-size', type=str, default='small')
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--use-lora', action='store_true')
    parser.add_argument('--use-qlora', action='store_true')
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--use-mlflow', action='store_true', help="Enable MLflow logging")
    
    args = parser.parse_args()
    main(args)