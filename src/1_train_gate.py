# -----------------------------------------------------------------
# File: 1_train_gate.py
# -----------------------------------------------------------------
# Description:
# Phase 1A: Trains the Multi-Label "Gate" model.
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

from dataloader import MultiLabelRetinaDataset, get_random_splits, train_transform, val_transform
from models import create_model, get_optimizer
from config import BRSET_LABELS

# --- MLflow Autolog ---
# We try to import mlflow. If it fails, we set a flag.
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: 'mlflow' package not found. MLflow logging will be disabled.")

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- PyTorch Lightning Module ---

class GateModule(pl.LightningModule):
    def __init__(self, hparams):
        """
        hparams is a Namespace object from argparse
        """
        super().__init__()
        self.save_hyperparameters(hparams)
        self.pathologies = BRSET_LABELS
        
        self.model = create_model(
            model_name=self.hparams.model_name,
            model_size=self.hparams.model_size,
            pretrained=not self.hparams.no_pretrained,
            num_classes=len(self.pathologies),
            use_lora=self.hparams.use_lora,
            use_qlora=self.hparams.use_qlora,
            lora_r=self.hparams.lora_r
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        
        # self.log() is automatically captured by autolog()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        preds = torch.sigmoid(outputs)
        self.validation_step_outputs.append({'preds': preds.detach().cpu(), 'targets': targets.detach().cpu()})
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).numpy()
        targets = torch.cat([x['targets'] for x in self.validation_step_outputs]).numpy()
        
        metrics = self._calculate_metrics(targets, preds)
        
        # Log val_auc_macro, which will be used for checkpointing
        self.log('val_auc_macro', metrics['auc_macro'], prog_bar=True)
        self.log('val_f1_macro', metrics['f1_macro'])
        
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        
        self.log('test_loss', loss, on_epoch=True)
        preds = torch.sigmoid(outputs)
        self.test_step_outputs.append({'preds': preds.detach().cpu(), 'targets': targets.detach().cpu()})

    def on_test_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.test_step_outputs]).numpy()
        targets = torch.cat([x['targets'] for x in self.test_step_outputs]).numpy()
        
        metrics = self._calculate_metrics(targets, preds)
        
        # Log final test metrics
        self.log('test_auc_macro', metrics['auc_macro'])
        self.log('test_f1_macro', metrics['f1_macro'])
        
        print(f"\n--- Gate Test Metrics ---")
        print(f"Test AUC (Macro): {metrics['auc_macro']:.4f}")
        print(f"Test F1 (Macro):  {metrics['f1_macro']:.4f}")
        
        # This is optional, but good for MLflow: log test metrics manually
        # as autolog might not capture this print statement.
        if self.hparams.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                'test_auc_macro': metrics['auc_macro'],
                'test_f1_macro': metrics['f1_macro']
            })

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = get_optimizer(self.model, self.hparams.lr, self.hparams.use_qlora)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_auc_macro",
                "mode": "max"
            },
        }

    def _calculate_metrics(self, targets, preds):
        # ... (same robust metrics function as before) ...
        preds_rounded = np.round(preds)
        auc_scores = []
        num_classes = targets.shape[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(num_classes):
                try:
                    if len(np.unique(targets[:, i])) > 1:
                        auc_scores.append(roc_auc_score(targets[:, i], preds[:, i]))
                    else:
                        auc_scores.append(0.5) 
                except ValueError:
                    auc_scores.append(0.5)
        auc_macro = np.nanmean(auc_scores)
        f1_macro = f1_score(targets, preds_rounded, average='macro', zero_division=0)
        return {'f1_macro': f1_macro, 'auc_macro': auc_macro}

# --- DataModule for Lightning ---

class GateDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.PATHOLOGIES = BRSET_LABELS
        self.labels_df = pd.read_csv(self.hparams.labels_path)
        self.labels_df['image_id'] = self.labels_df['image_id'].astype(str)
        self.splits = get_random_splits(self.labels_df, test_size=0.2, val_size=0.1)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MultiLabelRetinaDataset(
                self.labels_df, self.hparams.image_dir, self.PATHOLOGIES, train_transform, self.splits['train']
            )
            self.val_dataset = MultiLabelRetinaDataset(
                self.labels_df, self.hparams.image_dir, self.PATHOLOGIES, val_transform, self.splits['val']
            )
        if stage == 'test' or stage is None:
            self.test_dataset = MultiLabelRetinaDataset(
                self.labels_df, self.hparams.image_dir, self.PATHOLOGIES, val_transform, self.splits['test']
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
    dm = GateDataModule(args)
    
    # --- 2. Init Model ---
    model = GateModule(args)
    
    # --- 3. Init Loggers and Callbacks ---
    callbacks = [
        EarlyStopping(monitor='val_auc_macro', patience=args.patience, mode='max', verbose=True),
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
                checkpoint_monitor='val_auc_macro',
                checkpoint_mode='max',
                checkpoint_save_best_only=True,
                checkpoint_save_freq='epoch',
                # This ensures the checkpoint is saved in the *correct* dir
                # so the rest of the pipeline can find it.
                checkpoint_dirpath=args.output_dir, 
                checkpoint_filename='gate_best_model' # This will be the name
            )
            # We still create a logger to pass to the Trainer
            mlflow_logger = MLflowLogger(
                experiment_name=os.environ.get('MLFLOW_EXPERIMENT_NAME'),
                run_name=args.run_name,
                tracking_uri=os.environ.get('MLFLOW_TRACKING_URI')
            )
    
    if not args.use_mlflow:
        logger.info("MLflow is disabled. Using local ModelCheckpoint.")
        # Fallback to local checkpointing if MLflow is off
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            filename='gate_best_model', # Fixed name for the pipeline
            monitor='val_auc_macro',
            mode='max',
            save_top_k=1,
        )
        callbacks.append(checkpoint_callback)

    # --- 4. Init Trainer ---
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=mlflow_logger if args.use_mlflow else False, # Disable logger if not using MLflow
        callbacks=callbacks,
        log_every_n_steps=10
    )
    
    # --- 5. Run Training ---
    logger.info("--- Starting Gate Model Training ---")
    trainer.fit(model, datamodule=dm)
    
    # --- 6. Run Testing ---
    logger.info("--- Starting Gate Model Testing ---")
    # 'ckpt_path="best"' automatically loads the best model
    trainer.test(datamodule=dm, ckpt_path='best')
    
    logger.info("--- Gate Model Training & Testing Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Label Gate Model (Lightning)")
    
    # Add all arguments
    parser.add_argument('--labels-path', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default="checkpoints/gate")
    parser.add_argument('--run-name', type=str, default="Gate_Model_Run")
    parser.add_argument('--model-name', type=str, default='resnet')
    parser.add_argument('--model-size', type=str, default='small')
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--use-lora', action='store_true')
    parser.add_argument('--use-qlora', action='store_true')
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--use-mlflow', action='store_true', help="Enable MLflow logging")
    
    args = parser.parse_args()
    main(args)