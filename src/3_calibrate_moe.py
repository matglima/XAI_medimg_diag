# -----------------------------------------------------------------
# File: src/3_calibrate_moe.py
# -----------------------------------------------------------------
# Description:
# Phase 2B: Assembles the MoE and runs a brief, low-LR
# fine-tuning (calibration) on the final layers.
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
import warnings

# --- Lightning Imports ---
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
# --- End Lightning Imports ---

# --- MLflow Autolog ---
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: 'mlflow' package not found. MLflow logging will be disabled.")

from dataloader import MultiLabelRetinaDataset, get_random_splits, train_transform, val_transform
from moe_model import HybridMoE
from models import get_optimizer
from config import BRSET_LABELS

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- PyTorch Lightning Module ---

class CalibrationModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.PATHOLOGIES = BRSET_LABELS
        
        # --- Define Model Configurations ---
        gate_config = {
            'model_name': self.hparams.gate_model_name,
            'model_size': self.hparams.gate_model_size,
            'use_lora': self.hparams.gate_use_lora,
            'use_qlora': self.hparams.gate_use_qlora,
            'lora_r': self.hparams.lora_r
        }
        expert_config = {
            'model_name': self.hparams.expert_model_name,
            'model_size': self.hparams.expert_model_size,
            'use_lora': self.hparams.expert_use_lora,
            'use_qlora': self.hparams.expert_use_qlora,
            'lora_r': self.hparams.lora_r
        }

        # --- 1. Initialize MoE Structure ---
        logger.info("Initializing HybridMoE structure...")
        # We must use self.device *after* it's assigned by Lightning
        self.model = HybridMoE(gate_config, expert_config, self.PATHOLOGIES, 'cpu')
        
        # --- 2. Load All Checkpoints ---
        logger.info("Loading pre-trained checkpoints...")
        
        # --- START FIX: Correctly find gate checkpoint ---
        gate_ckpt_path = self.hparams.gate_ckpt_path
        if not self.hparams.gate_use_lora:
            # If not using LoRA, find the .pth file
            try:
                # Find the *first* file ending in .pth
                gate_ckpt_file = [f for f in os.listdir(gate_ckpt_path) if f.endswith('.pth')][0]
                gate_ckpt_path = os.path.join(gate_ckpt_path, gate_ckpt_file)
                logger.info(f"Found full gate checkpoint: {gate_ckpt_path}")
            except IndexError:
                logger.error(f"CRITICAL: --gate-use-lora=False, but no .pth file found in {gate_ckpt_path}")
                # Re-raise the error so the script stops
                raise FileNotFoundError(f"No .pth checkpoint found in {self.hparams.gate_ckpt_path}. Did training for Phase 1 fail or was it LoRA?")
        else:
            # If using LoRA, the path is just the directory
            logger.info(f"Using LoRA adapters for gate from: {gate_ckpt_path}")
        # --- END FIX ---
            
        self.model.load_checkpoints(
            gate_ckpt_path=gate_ckpt_path,
            expert_ckpt_dir=self.hparams.expert_ckpt_dir,
            gate_is_lora=self.hparams.gate_use_lora,
            expert_is_lora=self.hparams.expert_use_lora
        )
        
        # --- 3. Freeze Parameters for Calibration ---
        logger.info("Freezing model parameters for calibration...")
        total_params, trainable_params = 0, 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            param.requires_grad = False
            if 'lora_' in name or 'fc' in name or 'classifier' in name or 'head' in name:
                param.requires_grad = True
                trainable_params += param.numel()
        logger.info(f"Calibration params: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")

        self.criterion = nn.BCEWithLogitsLoss()
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        self.log('cal_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        loss = self.criterion(outputs, targets)
        self.log('cal_val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        preds = torch.sigmoid(outputs)
        self.validation_step_outputs.append({'preds': preds.detach().cpu(), 'targets': targets.detach().cpu()})
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).numpy()
        targets = torch.cat([x['targets'] for x in self.validation_step_outputs]).numpy()
        
        metrics = self._calculate_metrics(targets, preds)
        
        self.log('cal_val_auc_macro', metrics['auc_macro'], prog_bar=True)
        self.log('cal_val_f1_macro', metrics['f1_macro'])
        
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        use_qlora = self.hparams.gate_use_qlora or self.hparams.expert_use_qlora
        optimizer = get_optimizer(trainable_params, self.hparams.lr, use_qlora)
        return optimizer

    def _calculate_metrics(self, targets, preds):
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

class CalibrationDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.PATHOLOGIES = BRSET_LABELS
        self.labels_df = pd.read_csv(self.hparams.labels_path)
        self.labels_df['image_id'] = self.labels_df['image_id'].astype(str)
        self.splits = get_random_splits(self.labels_df, test_size=0.2, val_size=0.1)

    def setup(self, stage=None):
        self.train_dataset = MultiLabelRetinaDataset(
            self.labels_df, self.hparams.image_dir, self.PATHOLOGIES, train_transform, self.splits['train']
        )
        self.val_dataset = MultiLabelRetinaDataset(
            self.labels_df, self.hparams.image_dir, self.PATHOLOGIES, val_transform, self.splits['val']
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

# --- Main execution ---

def main(args):
    # --- 1. Init Data ---
    dm = CalibrationDataModule(args)
    
    # --- 2. Init Model ---
    model = CalibrationModule(args)
    
    # --- 3. Init Loggers and Callbacks ---
    callbacks = [
        TQDMProgressBar(refresh_rate=10)
    ]
    
    mlflow_logger = None
    checkpoint_callback = None # Initialize
    
    if args.use_mlflow:
        if not MLFLOW_AVAILABLE:
            logger.error("MLflow is not installed. Disabling MLflow logging.")
            args.use_mlflow = False
        else:
            logger.info("Enabling MLflow autologging...")
            mlflow.pytorch.autolog(
                log_models=True,
                checkpoint=True,
                checkpoint_monitor='cal_val_auc_macro',
                checkpoint_mode='max',
                checkpoint_save_best_only=True,
                checkpoint_save_freq='epoch',
                checkpoint_dirpath=args.output_dir,
                checkpoint_filename='moe_calibrated_best'
            )
            mlflow_logger = MLflowLogger(
                experiment_name=os.environ.get('MLFLOW_EXPERIMENT_NAME'),
                run_name=args.run_name,
                tracking_uri=os.environ.get('MLFLOW_TRACKING_URI')
            )
            
    if not args.use_mlflow:
        logger.info("MLflow is disabled. Using local ModelCheckpoint.")
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.output_dir,
            filename='moe_calibrated_best',
            monitor='cal_val_auc_macro',
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
        callbacks=callbacks
    )
    
    # --- 5. Run Training ---
    logger.info("--- Starting MoE Calibration ---")
    trainer.fit(model, datamodule=dm)
    
    logger.info("--- MoE Calibration Complete ---")
    
    # --- 6. Save final merged model ---
    logger.info("Loading best model from checkpoint...")
    
    best_ckpt_path = ""
    if args.use_mlflow:
        # MLflow autolog saves checkpoints in a subdirectory
        ckpt_dir = os.path.join(args.output_dir, "checkpoints")
        best_ckpt_path = os.path.join(ckpt_dir, "moe_calibrated_best.ckpt")
    else:
        # Ensure checkpoint_callback is not None
        if checkpoint_callback:
            best_ckpt_path = checkpoint_callback.best_model_path
        else:
            logger.error("Checkpoint callback was not initialized. Cannot find best model.")
            best_ckpt_path = "" # Will cause fallback

    if not os.path.exists(best_ckpt_path):
        logger.warning(f"Could not find best checkpoint at {best_ckpt_path}. Saving last model state.")
        best_model = model # Fallback to saving the last model state
    else:
        logger.info(f"Loading best model from: {best_ckpt_path}")
        best_model = CalibrationModule.load_from_checkpoint(best_ckpt_path)
    
    if args.gate_use_lora or args.expert_use_lora:
        logger.info("Merging LoRA adapters into base model for final saving...")
        if hasattr(best_model.model.gate, 'model') and hasattr(best_model.model.gate.model, 'merge_and_unload'):
            best_model.model.gate.model = best_model.model.gate.model.merge_and_unload()
        for expert in best_model.model.experts:
            if hasattr(expert, 'model') and hasattr(expert.model, 'merge_and_unload'):
                expert.model = expert.model.merge_and_unload()
        logger.info("Adapters merged.")
    
    # Save the final, merged state_dict
    final_save_path = os.path.join(args.output_dir, "moe_calibrated_final.pth")
    torch.save(best_model.model.state_dict(), final_save_path)
    logger.info(f"Final merged model saved to {final_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Hybrid MoE Model (Lightning)")
    
    parser.add_argument('--labels-path', type=str, required=True)
    parser.add_argument('--image-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default="checkpoints/final_moe")
    parser.add_argument('--run-name', type=str, default="Calibration_Run")
    parser.add_argument('--gate-ckpt-path', type=str, required=True)
    parser.add_argument('--expert-ckpt-dir', type=str, required=True)
    parser.add_argument('--gate-model-name', type=str, default='resnet')
    parser.add_argument('--gate-model-size', type=str, default='small')
    parser.add_argument('--gate-use-lora', action='store_true')
    parser.add_argument('--gate-use-qlora', action='store_true')
    parser.add_argument('--expert-model-name', type=str, default='resnet')
    parser.add_argument('--expert-model-size', type=str, default='small')
    parser.add_argument('--expert-use-lora', action='store_true')
    parser.add_argument('--expert-use-qlora', action='store_true')
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add-Adding --use-mlflow to notebook_args...
    parser.add_argument('--use-mlflow', action='store_true', help="Enable MLflow logging")
    
    args = parser.parse_args()
    main(args)