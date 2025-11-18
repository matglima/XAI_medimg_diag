# -----------------------------------------------------------------
# File: 3_calibrate_moe.py
# -----------------------------------------------------------------
# Description:
# Phase 2B: Assembles and calibrates the MoE.
# Refactored to PyTorch Lightning.
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
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
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
        
        # --- START FIX: Correctly find gate checkpoint and resolve path ---
        gate_ckpt_path = self.hparams.gate_ckpt_path
        if not self.hparams.gate_use_lora:
            # If not using LoRA, find the .pth file
            try:
                # Find the *first* file ending in .pth
                gate_ckpt_file = [f for f in os.listdir(gate_ckpt_path) if f.endswith('.pth')][0]
                gate_ckpt_path = os.path.join(gate_ckpt_path, gate_ckpt_file)
                logger.info(f"Found full gate checkpoint: {gate_ckpt_path}")
                # --- FIX: Resolve to absolute path to prevent Hub lookup ---
                gate_ckpt_path = os.path.abspath(gate_ckpt_path)
            except IndexError:
                logger.error(f"CRITICAL: --gate-use-lora=False, but no .pth file found in {gate_ckpt_path}")
                raise FileNotFoundError(f"No .pth checkpoint found in {self.hparams.gate_ckpt_path}. Did training for Phase 1 fail?")
        else:
            # If using LoRA, the path is just the directory
            logger.info(f"Using LoRA adapters for gate from: {gate_ckpt_path}")
            # --- FIX: Resolve to absolute path to prevent Hub lookup ---
            gate_ckpt_path = os.path.abspath(gate_ckpt_path)

            
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
            # Only unfreeze LoRA params or classifier heads
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
        if not self.validation_step_outputs:
            return
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
                    auc_scores.append(0