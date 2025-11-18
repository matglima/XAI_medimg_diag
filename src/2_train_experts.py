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
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from lightning.pytorch.loggers import MLflowLogger
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
        if not self.validation_step_outputs:
            return
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
        if not self.test_step_outputs:
            return
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
    
    # --- START FIX: Always use ModelCheckpoint to find the best .ckpt ---
    # We save the .ckpt file flat in the expert dir
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir, # Save directly to the expert dir
        filename=f'{args.target_label}_best_model',
        monitor='val_auc',
        mode='max',
        save_top_k=1,
    )
    callbacks = [
        EarlyStopping(monitor='val_auc', patience=args.patience, mode='max', verbose=True),
        TQDMProgressBar(refresh_rate=10),
        checkpoint_callback # Always add the callback
    ]
    # --- END FIX ---
    
    mlflow_logger = None
    
    if args.use_mlflow:
        if not MLFLOW_AVAILABLE:
            logger.error("MLflow is not installed. Disabling MLflow logging.")
            args.use_mlflow = False
        else:
            logger.info("Enabling MLflow autologging...")
            # --- START FIX: Disable MLflow's checkpointing ---
            mlflow.pytorch.autolog(
                log_models=False, # Disable auto-logging models
                checkpoint=False, # Disable auto-checkpointing
                disable=True      # Disable complex autologging
            )
            # Re-enable simple metric logging
            mlflow.pytorch.autolog(
                log_models=False,
                log_datasets=False,
                log_input_examples=False,
                log_loss_metrics=True,
                log_opt_hyperparams=True
            )
            # We use a nested run for better organization
            mlf_logger = MLflowLogger(
                experiment_name=os.environ.get('MLFLOW_EXPERIMENT_NAME'),
                run_name=args.run_name,
                tracking_uri=os.environ.get('MLFLOW_TRACKING_URI')
            )

    if not args.use_mlflow:
        logger.info("MLflow is disabled. Using local ModelCheckpoint.")
        # This is now handled above
        pass

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

    # --- START FIX: Save final model in the correct format ---
    logger.info(f"Saving final expert model for {args.target_label}...")
    
    best_ckpt_path = checkpoint_callback.best_model_path
    if not best_ckpt_path or not os.path.exists(best_ckpt_path):
        logger.error("Could not find best checkpoint path. Saving from last model state.")
        best_model_to_save = model.model
    else:
        logger.info(f"Loading best model from: {best_ckpt_path}")
        best_model_to_save = ExpertModule.load_from_checkpoint(best_ckpt_path).model

    if args.use_lora or args.use_qlora:
        # Save as PEFT adapters in a subdirectory named after the label
        final_save_dir = os.path.join(args.output_dir, args.target_label)
        os.makedirs(final_save_dir, exist_ok=True)
        logger.info(f"Saving LoRA adapters to: {final_save_dir}")
        best_model_to_save.save_pretrained(final_save_dir)
        if args.use_mlflow:
            mlflow.log_artifact(final_save_dir, artifact_path=f"{args.target_label}_adapters")
    else:
        # Save as raw .pth state_dict named after the label
        final_save_path = os.path.join(args.output_dir, f"{args.target_label}.pth")
        logger.info(f"Saving full model state_dict to: {final_save_path}")
        torch.save(best_model_to_save.state_dict(), final_save_path)
        if args.use_mlflow:
            mlflow.log_artifact(final_save_path, artifact_path=f"{args.target_label}_model")
            
    # Clean up the .ckpt file to save space
    if os.path.exists(best_ckpt_path):
        logger.info(f"Cleaning up temporary checkpoint: {best_ckpt_path}")
        os.remove(best_ckpt_path)
            
    logger.info(f"Final expert model for {args.target_label} saved.")
    # --- END FIX ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Binary Expert Model (Lightning)")
    
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