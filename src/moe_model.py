# -----------------------------------------------------------------
# File: moe_model.py
# -----------------------------------------------------------------
# Description:
# Phase 2A: Defines the HybridMoE architecture.
# This model assembles the gate and all experts.
# -----------------------------------------------------------------

import torch
import torch.nn as nn
from models import create_model # This now accepts lora_r
import os

try:
    from peft import PeftModel
    PEFT_INSTALLED = True
except ImportError:
    PEFT_INSTALLED = False
    print("Warning: 'peft' not found. Loading LoRA adapters will fail.")

class HybridMoE(nn.Module):
    def __init__(self, gate_config, expert_config, pathology_list, device,
                 fusion_strategy='additive', top_k=1):
        """
        Initializes the MoE structure *without* weights.
        
        Args:
            gate_config (dict): Kwargs for create_model (name, size, lora, qlora, lora_r)
            expert_config (dict): Kwargs for *one* expert (name, size, lora, qlora, lora_r)
            pathology_list (list): List of pathology names in order.
            device: The torch device to load models onto.
        """
        super().__init__()
        self.pathology_list = pathology_list
        self.device = device
        fusion_aliases = {
            'additive': 'additive',
            'sum': 'additive',
            'scalar': 'scalar_calibration',
            'scalar_calibration': 'scalar_calibration',
            'soft': 'soft_gating',
            'soft_gating': 'soft_gating',
            'topk': 'topk',
            'top_k': 'topk',
        }
        normalized_strategy = fusion_aliases.get(fusion_strategy.lower())
        if normalized_strategy is None:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

        self.fusion_strategy = normalized_strategy
        self.top_k = int(top_k)
        
        # 1. Create the Gate model structure
        self.gate = create_model(
            **gate_config,
            num_classes=len(pathology_list),
            pretrained=False # We will load weights
        ).to(device)
        
        # 2. Create the Expert model structures
        self.experts = nn.ModuleList()
        for _ in self.pathology_list:
            expert = create_model(
                **expert_config,
                num_classes=1,
                pretrained=False # We will load weights
            ).to(device)
            self.experts.append(expert)

        if self.fusion_strategy == 'scalar_calibration':
            num_pathologies = len(self.pathology_list)
            self.fusion_alpha = nn.Parameter(torch.ones(1, num_pathologies))
            self.fusion_beta = nn.Parameter(torch.ones(1, num_pathologies))
            self.fusion_bias = nn.Parameter(torch.zeros(1, num_pathologies))

    def load_checkpoints(self, gate_ckpt_path, expert_ckpt_dir, gate_is_lora, expert_is_lora):
        """
        Loads the pre-trained weights into the model structures.
        """
        # 1. Load Gate Weights
        if gate_is_lora:
            if not PEFT_INSTALLED: raise ImportError("PEFT not installed.")
            print(f"Loading LoRA adapters for Gate from: {gate_ckpt_path}")
            # gate_ckpt_path should be the *directory* containing adapter_config.json
            self.gate.model = PeftModel.from_pretrained(self.gate.model, gate_ckpt_path)
        else:
            full_path = f"{gate_ckpt_path}.pth" if not gate_ckpt_path.endswith('.pth') else gate_ckpt_path
            print(f"Loading full weights for Gate from: {full_path}")
            
            # Load raw state dict
            state_dict = torch.load(full_path, map_location=self.device, weights_only=False)
            
            # --- FIX: Add 'model.' prefix for non-LoRA loading ---
            # The saved file (from 1_train_gate.py) has keys like 'features.0...'
            # But self.gate is a ModelWrapper, so it expects 'model.features.0...'
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[f'model.{k}'] = v
            
            # Load with the corrected keys
            self.gate.load_state_dict(new_state_dict)
            # --- END FIX ---
    
        # 2. Load Expert Weights
        for i, pathology in enumerate(self.pathology_list):
            if expert_is_lora:
                if not PEFT_INSTALLED: raise ImportError("PEFT not installed.")
                
                # --- FIX: Ensure path is correct (no suffixes) ---
                adapter_path = os.path.join(expert_ckpt_dir, pathology)
                # -----------------------------------------------
                
                print(f"Loading LoRA adapters for expert: {pathology}")
                self.experts[i].model = PeftModel.from_pretrained(self.experts[i].model, adapter_path)
            else:
                model_path = os.path.join(expert_ckpt_dir, f"{pathology}.pth")
                print(f"Loading full weights for expert: {pathology}")
                
                # Load raw state dict
                state_dict = torch.load(model_path, map_location=self.device, weights_only=False)

                # --- FIX: Add 'model.' prefix for expert non-LoRA loading ---
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict[f'model.{k}'] = v
                
                self.experts[i].load_state_dict(new_state_dict)
                # --- END FIX ---
        
        print("All checkpoints loaded successfully.")

    def has_trainable_fusion_parameters(self):
        return self.fusion_strategy == 'scalar_calibration'

    def _collect_logits(self, x):
        gate_logits = self.gate(x)

        expert_logits = []
        for expert in self.experts:
            expert_logits.append(expert(x))

        expert_logits_tensor = torch.cat(expert_logits, dim=1)
        return gate_logits, expert_logits_tensor

    def _combine_logits(self, gate_logits, expert_logits_tensor):
        if self.fusion_strategy == 'additive':
            expert_weights = torch.ones_like(expert_logits_tensor)
            final_logits = gate_logits + expert_logits_tensor
        elif self.fusion_strategy == 'scalar_calibration':
            expert_weights = self.fusion_beta.expand_as(expert_logits_tensor)
            final_logits = (
                self.fusion_alpha * gate_logits
                + self.fusion_beta * expert_logits_tensor
                + self.fusion_bias
            )
        elif self.fusion_strategy == 'soft_gating':
            expert_weights = torch.sigmoid(gate_logits)
            final_logits = gate_logits + (expert_weights * expert_logits_tensor)
        elif self.fusion_strategy == 'topk':
            gate_scores = torch.sigmoid(gate_logits)
            if self.top_k <= 0 or self.top_k >= gate_scores.shape[1]:
                expert_weights = torch.ones_like(gate_scores)
            else:
                top_k = min(self.top_k, gate_scores.shape[1])
                topk_indices = torch.topk(gate_scores, k=top_k, dim=1).indices
                expert_weights = torch.zeros_like(gate_scores)
                expert_weights.scatter_(1, topk_indices, 1.0)
            final_logits = gate_logits + (expert_weights * expert_logits_tensor)
        else:  # pragma: no cover
            raise RuntimeError(f"Unsupported fusion strategy: {self.fusion_strategy}")

        return final_logits, expert_weights

    def forward_with_components(self, x):
        gate_logits, expert_logits_tensor = self._collect_logits(x)
        final_logits, expert_weights = self._combine_logits(gate_logits, expert_logits_tensor)
        return {
            'gate_logits': gate_logits,
            'expert_logits': expert_logits_tensor,
            'expert_weights': expert_weights,
            'final_logits': final_logits,
        }

    def forward(self, x):
        """
        Runs the dense Hybrid MoE and returns the fused logits.
        """
        outputs = self.forward_with_components(x)
        final_logits = outputs['final_logits']
        return final_logits
