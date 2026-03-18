#!/usr/bin/env python3

import argparse
import csv
import gc
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import BRSET_LABELS
from models import create_model
from moe_model import HybridMoE


BACKBONE_SPECS = {
    "efficientnet": {"gate_size": "medium", "expert_size": "small"},
    "resnet": {"gate_size": "medium", "expert_size": "small"},
    "convnext": {"gate_size": "small", "expert_size": "tiny"},
    "swin": {"gate_size": "small", "expert_size": "tiny"},
    "vit": {"gate_size": "base", "expert_size": "small"},
}


def parse_csv_list(raw_value, cast_fn=str):
    return [cast_fn(item.strip()) for item in raw_value.split(",") if item.strip()]


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()


def is_oom_error(exc):
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def count_trainable_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def configure_calibration_trainable_params(model, fusion_only=False):
    for name, param in model.named_parameters():
        param.requires_grad = False

        if name.startswith("fusion_"):
            param.requires_grad = True
            continue

        if fusion_only:
            continue

        if "lora_" in name or "fc" in name or "classifier" in name or "head" in name:
            param.requires_grad = True


def build_gate_model(backbone_name, use_lora=False, lora_r=128):
    spec = BACKBONE_SPECS[backbone_name]
    return create_model(
        model_name=backbone_name,
        model_size=spec["gate_size"],
        pretrained=False,
        num_classes=len(BRSET_LABELS),
        use_lora=use_lora,
        use_qlora=False,
        lora_r=lora_r,
    )


def build_expert_model(backbone_name, use_lora=False, lora_r=128):
    spec = BACKBONE_SPECS[backbone_name]
    return create_model(
        model_name=backbone_name,
        model_size=spec["expert_size"],
        pretrained=False,
        num_classes=1,
        use_lora=use_lora,
        use_qlora=False,
        lora_r=lora_r,
    )


def build_moe_model(backbone_name, fusion_strategy, top_k, gate_use_lora=False, expert_use_lora=False, lora_r=128):
    spec = BACKBONE_SPECS[backbone_name]
    return HybridMoE(
        gate_config={
            "model_name": backbone_name,
            "model_size": spec["gate_size"],
            "use_lora": gate_use_lora,
            "use_qlora": False,
            "lora_r": lora_r,
        },
        expert_config={
            "model_name": backbone_name,
            "model_size": spec["expert_size"],
            "use_lora": expert_use_lora,
            "use_qlora": False,
            "lora_r": lora_r,
        },
        pathology_list=BRSET_LABELS,
        device="cpu",
        fusion_strategy=fusion_strategy,
        top_k=top_k,
    )


def get_scenarios():
    return [
        {
            "name": "gate_train_full",
            "description": "Gate-only training, full fine-tuning",
            "mode": "train",
            "builder": lambda backbone, lora_r: build_gate_model(backbone, use_lora=False, lora_r=lora_r),
        },
        {
            "name": "gate_train_lora",
            "description": "Gate-only training, LoRA",
            "mode": "train",
            "builder": lambda backbone, lora_r: build_gate_model(backbone, use_lora=True, lora_r=lora_r),
        },
        {
            "name": "expert_train_full",
            "description": "Single expert training, full fine-tuning",
            "mode": "train_binary",
            "builder": lambda backbone, lora_r: build_expert_model(backbone, use_lora=False, lora_r=lora_r),
        },
        {
            "name": "expert_train_lora",
            "description": "Single expert training, LoRA",
            "mode": "train_binary",
            "builder": lambda backbone, lora_r: build_expert_model(backbone, use_lora=True, lora_r=lora_r),
        },
        {
            "name": "moe_calibrate_additive_full_full",
            "description": "MoE calibration, additive fusion, full/full",
            "mode": "calibrate",
            "fusion_strategy": "additive",
            "top_k": 1,
            "gate_use_lora": False,
            "expert_use_lora": False,
            "fusion_only": False,
        },
        {
            "name": "moe_calibrate_scalar_full_full",
            "description": "MoE calibration, scalar fusion, full/full",
            "mode": "calibrate",
            "fusion_strategy": "scalar_calibration",
            "top_k": 1,
            "gate_use_lora": False,
            "expert_use_lora": False,
            "fusion_only": True,
        },
        {
            "name": "moe_calibrate_soft_full_full",
            "description": "MoE calibration, soft gating, full/full",
            "mode": "calibrate",
            "fusion_strategy": "soft_gating",
            "top_k": 1,
            "gate_use_lora": False,
            "expert_use_lora": False,
            "fusion_only": False,
        },
        {
            "name": "moe_calibrate_topk3_full_full",
            "description": "MoE calibration, top-k=3 fusion, full/full",
            "mode": "calibrate",
            "fusion_strategy": "topk",
            "top_k": 3,
            "gate_use_lora": False,
            "expert_use_lora": False,
            "fusion_only": False,
        },
        {
            "name": "moe_calibrate_scalar_full_lora",
            "description": "MoE calibration, scalar fusion, full gate + LoRA experts",
            "mode": "calibrate",
            "fusion_strategy": "scalar_calibration",
            "top_k": 1,
            "gate_use_lora": False,
            "expert_use_lora": True,
            "fusion_only": True,
        },
        {
            "name": "moe_calibrate_scalar_lora_full",
            "description": "MoE calibration, scalar fusion, LoRA gate + full experts",
            "mode": "calibrate",
            "fusion_strategy": "scalar_calibration",
            "top_k": 1,
            "gate_use_lora": True,
            "expert_use_lora": False,
            "fusion_only": True,
        },
        {
            "name": "moe_calibrate_scalar_lora_lora",
            "description": "MoE calibration, scalar fusion, LoRA gate + LoRA experts",
            "mode": "calibrate",
            "fusion_strategy": "scalar_calibration",
            "top_k": 1,
            "gate_use_lora": True,
            "expert_use_lora": True,
            "fusion_only": True,
        },
        {
            "name": "moe_eval_additive",
            "description": "MoE evaluation, additive fusion",
            "mode": "eval",
            "fusion_strategy": "additive",
            "top_k": 1,
        },
        {
            "name": "moe_eval_scalar",
            "description": "MoE evaluation, scalar fusion",
            "mode": "eval",
            "fusion_strategy": "scalar_calibration",
            "top_k": 1,
        },
        {
            "name": "moe_eval_soft",
            "description": "MoE evaluation, soft gating",
            "mode": "eval",
            "fusion_strategy": "soft_gating",
            "top_k": 1,
        },
        {
            "name": "moe_eval_topk3",
            "description": "MoE evaluation, top-k=3 fusion",
            "mode": "eval",
            "fusion_strategy": "topk",
            "top_k": 3,
        },
    ]


def instantiate_scenario(backbone_name, scenario, device, lora_r):
    mode = scenario["mode"]

    if mode in {"train", "train_binary"}:
        model = scenario["builder"](backbone_name, lora_r)
        return model.to(device)

    model = build_moe_model(
        backbone_name=backbone_name,
        fusion_strategy=scenario["fusion_strategy"],
        top_k=scenario["top_k"],
        gate_use_lora=scenario.get("gate_use_lora", False),
        expert_use_lora=scenario.get("expert_use_lora", False),
        lora_r=lora_r,
    )
    model = model.to(device)

    if mode == "calibrate":
        configure_calibration_trainable_params(model, fusion_only=scenario.get("fusion_only", False))
    else:
        model.eval()

    return model


def run_trial(model, scenario, batch_size, image_size, device):
    mode = scenario["mode"]
    criterion = nn.BCEWithLogitsLoss()
    optimizer = None

    if mode == "train":
        model.train()
        optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=1e-4)
        targets = torch.rand(batch_size, len(BRSET_LABELS), device=device)
    elif mode == "train_binary":
        model.train()
        optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=1e-4)
        targets = torch.rand(batch_size, device=device)
    elif mode == "calibrate":
        model.train()
        trainable_params = [param for param in model.parameters() if param.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found for calibration scenario.")
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)
        targets = torch.rand(batch_size, len(BRSET_LABELS), device=device)
    else:
        model.eval()
        targets = None

    inputs = torch.randn(batch_size, 3, image_size, image_size, device=device)

    if mode == "eval":
        with torch.no_grad():
            _ = model(inputs)
        return

    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)
    if mode == "train_binary":
        outputs = outputs.squeeze(-1)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()


def probe_scenario(backbone_name, scenario, batch_sizes, device, image_size, lora_r, warmup_runs):
    result = {
        "backbone": backbone_name,
        "scenario": scenario["name"],
        "description": scenario["description"],
        "max_batch_size": 0,
        "first_failed_batch_size": "",
        "peak_memory_mb": "",
        "trainable_params": "",
        "status": "ok",
        "error": "",
    }

    model = None
    try:
        cleanup_cuda()
        model = instantiate_scenario(backbone_name, scenario, device, lora_r)
        result["trainable_params"] = count_trainable_params(model)

        for batch_size in batch_sizes:
            try:
                cleanup_cuda()
                for _ in range(max(1, warmup_runs)):
                    run_trial(model, scenario, batch_size, image_size, device)
                if torch.cuda.is_available():
                    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    result["peak_memory_mb"] = round(peak_memory_mb, 2)
                result["max_batch_size"] = batch_size
            except RuntimeError as exc:
                if is_oom_error(exc):
                    result["first_failed_batch_size"] = batch_size
                    result["status"] = "oom"
                    cleanup_cuda()
                    break
                raise

        if result["max_batch_size"] == 0 and not result["first_failed_batch_size"]:
            result["status"] = "failed"
            result["error"] = "No successful batch size tested."
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
    finally:
        if model is not None:
            del model
        cleanup_cuda()

    return result


def write_results(output_path, rows):
    fieldnames = [
        "backbone",
        "scenario",
        "description",
        "max_batch_size",
        "first_failed_batch_size",
        "peak_memory_mb",
        "trainable_params",
        "status",
        "error",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Probe safe batch sizes for paper experiment scenarios.")
    parser.add_argument("--backbones", type=str, default="efficientnet,resnet,convnext,swin,vit")
    parser.add_argument("--scenarios", type=str, default="all")
    parser.add_argument("--batch-sizes", type=str, default="4,8,16,24,32,48,64,96,128,160,192,224,256,320,384,448,512")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--output", type=str, default="batch_probe_results.csv")
    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is not available. Run this probe on the target GPU server.")

    device = torch.device(args.device)
    backbones = parse_csv_list(args.backbones)
    batch_sizes = sorted(set(parse_csv_list(args.batch_sizes, int)))
    all_scenarios = get_scenarios()

    if args.scenarios == "all":
        scenarios = all_scenarios
    else:
        selected = set(parse_csv_list(args.scenarios))
        scenarios = [scenario for scenario in all_scenarios if scenario["name"] in selected]

    print(f"Using device: {device}")
    print(f"Backbones: {backbones}")
    print(f"Scenarios: {[scenario['name'] for scenario in scenarios]}")
    print(f"Batch sizes: {batch_sizes}")
    print("-" * 100)

    results = []
    start_time = time.time()

    for backbone_name in backbones:
        if backbone_name not in BACKBONE_SPECS:
            raise ValueError(f"Unsupported backbone '{backbone_name}'.")

        for scenario in scenarios:
            print(f"Probing backbone={backbone_name:<12} scenario={scenario['name']}")
            row = probe_scenario(
                backbone_name=backbone_name,
                scenario=scenario,
                batch_sizes=batch_sizes,
                device=device,
                image_size=args.image_size,
                lora_r=args.lora_r,
                warmup_runs=args.warmup_runs,
            )
            results.append(row)
            print(
                f"  -> status={row['status']}, max_batch_size={row['max_batch_size']}, "
                f"first_failed={row['first_failed_batch_size']}, peak_memory_mb={row['peak_memory_mb']}"
            )
            if row["error"]:
                print(f"  -> error={row['error']}")

    write_results(args.output, results)
    duration = time.time() - start_time
    print("-" * 100)
    print(f"Probe complete in {duration / 60.0:.2f} minutes.")
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
