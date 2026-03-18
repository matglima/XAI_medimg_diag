# -----------------------------------------------------------------
# File: 3_calibrate_moe.py
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
from lightning.pytorch.loggers import MLFlowLogger
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
from experiment_utils import save_run_manifest, set_global_seed
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
        self.model = HybridMoE(
            gate_config,
            expert_config,
            self.PATHOLOGIES,
            'cpu',
            fusion_strategy=self.hparams.fusion_strategy,
            top_k=self.hparams.top_k,
        )
        
        # --- CRITICAL CHANGE: NO LOADING HERE ---
        # We do not call load_checkpoints here to avoid the "Double Loading" crash 
        # and the "Optimizer Empty List" crash.
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.validation_step_outputs = []

    def setup_trainable_parameters(self):
        """
        Configures requires_grad for all parameters.
        Must be called AFTER load_checkpoints() injects LoRA adapters in main().
        """
        logger.info("Configuring trainable parameters (Freezing base, Unfreezing LoRA/Heads)...")
        logger.info(
            "Calibration mode: fusion_strategy=%s, fusion_only=%s",
            self.hparams.fusion_strategy,
            self.hparams.fusion_only,
        )
        total_params, trainable_params = 0, 0

        if self.hparams.fusion_only and not self.model.has_trainable_fusion_parameters():
            raise ValueError(
                f"fusion_only=True is incompatible with fusion strategy '{self.hparams.fusion_strategy}' because it has no trainable fusion parameters."
            )
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            # 1. Default: Freeze everything
            param.requires_grad = False

            if name.startswith('fusion_'):
                param.requires_grad = True
                trainable_params += param.numel()
                continue

            if self.hparams.fusion_only:
                continue
            
            # 2. Unfreeze logic: LoRA adapters, Classifiers/Heads
            # We match common names for heads/adapters
            if 'lora_' in name or 'fc' in name or 'classifier' in name or 'head' in name:
                param.requires_grad = True
                trainable_params += param.numel()
        
        if trainable_params == 0:
            logger.warning("WARNING: No trainable parameters found! Check layer names.")
            
        logger.info(f"Calibration params: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")

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
        # This list will be populated correctly because setup_trainable_parameters 
        # is called in main() before the trainer starts.
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
        self.splits = get_random_splits(
            self.labels_df,
            test_size=0.2,
            val_size=0.1,
            random_state=self.hparams.seed,
            split_manifest_path=self.hparams.split_manifest,
        )

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
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, 
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

# --- Main execution ---

def main(args):
    set_global_seed(args.seed)

    # --- 1. Init Data ---
    dm = CalibrationDataModule(args)
    
    # --- 2. Init Model ---
    model = CalibrationModule(args)
    os.makedirs(args.output_dir, exist_ok=True)
    save_run_manifest(
        args.output_dir,
        "calibration_run_manifest.json",
        args,
        extra={'phase': 'calibration'},
    )

    # --- 3. Manually Load Checkpoints HERE (Only for training start) ---
    logger.info("Loading pre-trained checkpoints into initialized model...")
    
    gate_ckpt_path = args.gate_ckpt_path
    if not args.gate_use_lora:
        try:
            gate_ckpt_file = [f for f in os.listdir(gate_ckpt_path) if f.endswith('.pth')][0]
            gate_ckpt_path = os.path.join(gate_ckpt_path, gate_ckpt_file)
            logger.info(f"Found full gate checkpoint: {gate_ckpt_path}")
            gate_ckpt_path = os.path.abspath(gate_ckpt_path)
        except IndexError:
            logger.error(f"CRITICAL: --gate-use-lora=False, but no .pth file found in {gate_ckpt_path}")
            raise FileNotFoundError(f"No .pth checkpoint found in {args.gate_ckpt_path}.")
    else:
        logger.info(f"Using LoRA adapters for gate from: {gate_ckpt_path}")
        gate_ckpt_path = os.path.abspath(gate_ckpt_path)

    expert_ckpt_dir = os.path.abspath(args.expert_ckpt_dir)
    logger.info(f"Resolved expert checkpoint directory to: {expert_ckpt_dir}")
    
    # Load weights (Injects LoRA if applicable)
    model.model.load_checkpoints(
        gate_ckpt_path=gate_ckpt_path,
        expert_ckpt_dir=expert_ckpt_dir,
        gate_is_lora=args.gate_use_lora,
        expert_is_lora=args.expert_use_lora
    )

    # --- 4. Configure Trainable Parameters ---
    # We do this AFTER load_checkpoints so the optimizer sees the correct parameters
    model.setup_trainable_parameters()

    # --- 5. Init Loggers and Callbacks ---
    callbacks = [ TQDMProgressBar(refresh_rate=10) ]
    mlflow_logger = None
    checkpoint_callback = None 
    
    if args.use_mlflow:
        if not MLFLOW_AVAILABLE:
            logger.error("MLflow is not installed. Disabling MLflow logging.")
            args.use_mlflow = False
        else:
            logger.info("Enabling MLflow autologging...")
            mlflow.pytorch.autolog(
                log_models=True, checkpoint=True, log_datasets=False,
                checkpoint_monitor='cal_val_auc_macro', checkpoint_mode='max',
                checkpoint_save_best_only=True, checkpoint_save_freq='epoch',
                checkpoint_dirpath=args.output_dir, checkpoint_filename='moe_calibrated_best'
            )
            mlflow_logger = MLFlowLogger(
                experiment_name=os.environ.get('MLFLOW_EXPERIMENT_NAME'),
                run_name=args.run_name, tracking_uri=os.environ.get('MLFLOW_TRACKING_URI')
            )
            
    if not args.use_mlflow:
        logger.info("MLflow is disabled. Using local ModelCheckpoint.")
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.output_dir, filename='moe_calibrated_best',
            monitor='cal_val_auc_macro', mode='max', save_top_k=1,
        )
        callbacks.append(checkpoint_callback)

    # --- 6. Init Trainer ---
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=mlflow_logger if args.use_mlflow else False,
        callbacks=callbacks,
        deterministic=True,
    )
    
    # --- 7. Run Training ---
    logger.info("--- Starting MoE Calibration ---")
    trainer.fit(model, datamodule=dm)
    
    logger.info("--- MoE Calibration Complete ---")
    
    # --- 8. Save final merged model ---
    
    # --- START FIX: Avoid reloading from checkpoint if we just finished training ---
    # We have 'model' in memory which is the trained state. 
    # If we try load_from_checkpoint() here with LoRA, it will crash because the 
    # structure won't match (base vs PEFT).
    
    # Check if we have a best model checkpoint from Lightning
    best_ckpt_path = ""
    if args.use_mlflow:
        ckpt_dir = os.path.join(args.output_dir, "checkpoints")
        best_ckpt_path = os.path.join(ckpt_dir, "moe_calibrated_best.ckpt")
    elif checkpoint_callback:
        best_ckpt_path = checkpoint_callback.best_model_path

    best_model_to_save = model # Default to current state (last epoch)
    
    if best_ckpt_path and os.path.exists(best_ckpt_path):
        logger.info(f"Loading best model weights from: {best_ckpt_path}")
        # We load the weights into the EXISTING model object (which has the correct LoRA structure)
        # instead of creating a new one via load_from_checkpoint
        checkpoint = torch.load(best_ckpt_path, map_location=model.device)
        model.load_state_dict(checkpoint['state_dict'])
        best_model_to_save = model
    else:
        logger.warning("Could not find best checkpoint. Saving the last model state.")

    # --- 9. Merge and Save ---
    if args.gate_use_lora or args.expert_use_lora:
        logger.info("Merging LoRA adapters into base model for final saving...")
        
        # Use best_model_to_save.model to access the HybridMoE
        moe_inner = best_model_to_save.model 
        
        if args.gate_use_lora and hasattr(moe_inner.gate, 'model') and hasattr(moe_inner.gate.model, 'merge_and_unload'):
            try:
                moe_inner.gate.model = moe_inner.gate.model.merge_and_unload()
                logger.info("Gate adapters merged.")
            except Exception as e: logger.warning(f"Could not merge gate adapters: {e}. Saving unmerged.")
        
        if args.expert_use_lora:
            for i, expert in enumerate(moe_inner.experts):
                if hasattr(expert, 'model') and hasattr(expert.model, 'merge_and_unload'):
                    try:
                        expert.model = expert.model.merge_and_unload()
                    except Exception as e: logger.warning(f"Could not merge expert {i} adapters: {e}. Saving unmerged.")
            logger.info("Expert adapters merged.")
    
    final_save_path = os.path.join(args.output_dir, "moe_calibrated_final.pth")
    torch.save(best_model_to_save.model.state_dict(), final_save_path)
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
    parser.add_argument('--fusion-strategy', type=str, default='additive')
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--fusion-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-manifest', type=str, default=None)
    parser.add_argument('--use-mlflow', action='store_true', help="Enable MLflow logging")
    
    args = parser.parse_args()
    main(args)
