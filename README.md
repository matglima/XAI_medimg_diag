# XAI_medimg_diag — Explainable Hybrid Mixture-of-Experts for Retinal Fundus Diagnosis

A reproducible deep-learning pipeline for **multi-label diagnosis of retinal pathologies** from
colour fundus photographs, built around a **Hybrid Mixture-of-Experts (MoE)** architecture and
paired with **explainable-AI (XAI)** visualisations and **fairness / subgroup** evaluation.

The code trains a multi-label *gate* network together with a bank of binary *expert* networks (one
per pathology), fuses their predictions with a configurable strategy, and reports calibrated,
threshold-tuned performance together with Grad-CAM / saliency explanations.

---

## Table of Contents

1. [Description](#description)
2. [Dataset Information](#dataset-information)
3. [Code Information](#code-information)
4. [Requirements](#requirements)
5. [Usage Instructions](#usage-instructions)
6. [Methodology](#methodology)
7. [Citations](#citations)
8. [License](#license)
9. [Contribution Guidelines](#contribution-guidelines)

---

## Description

`XAI_medimg_diag` implements and benchmarks a **Hybrid Mixture-of-Experts** classifier for
multi-label retinal disease detection. The central idea is to combine:

- a **gate** model — a single backbone trained on *all* pathologies simultaneously (multi-label),
  which provides a broad, shared view of the image; and
- a set of **expert** models — one binary classifier per pathology, each specialised on a single
  disease.

The gate and expert logits are merged by a **fusion module** whose behaviour is selectable at run
time (`additive`, `scalar_calibration`, `soft_gating`, or `topk`). A short **calibration** phase
optionally learns per-class scaling/bias parameters so the fused output is well-calibrated.

The repository is designed for **systematic experimentation**:

- Five interchangeable torchvision backbones — ResNet, EfficientNet, ConvNeXt, ViT, Swin.
- Optional **LoRA / Q-LoRA** parameter-efficient fine-tuning for both gate and experts.
- Shared, seed-controlled train/val/test split manifests for fair comparisons across runs.
- Optional **MLflow** experiment tracking.
- **Explainability** via Grad-CAM (SmoothGrad-CAM++) and gradient saliency maps.
- **Fairness analysis** through per-subgroup metrics (e.g. by sex, age bin, camera, comorbidity).

---

## Dataset Information

This project is built for the **BRSET — A Brazilian Multilabel Ophthalmological Dataset**, a public
collection of colour fundus photographs annotated with multiple ophthalmological labels and rich
patient metadata (e.g. age, sex, camera, comorbidities).

> **The dataset is *not* redistributed in this repository.** Image and label files are excluded via
> `.gitignore`. You must download BRSET from PhysioNet (see [Citations](#citations)) and place it
> under a local `data/` directory.

### Expected local layout

```
data/
├── labels_brset.csv        # master label/metadata file (one row per image)
└── fundus_photos/          # all fundus images (.png or .jpg), named <image_id>.png
```

### Label file format

`labels_brset.csv` must contain:

- an `image_id` column matching the image file names (without extension), and
- one binary column (0/1, or `yes`/`no` for `diabetes`) per pathology.

The 14 target pathologies (see `src/config.py`) are:

```
diabetes, diabetic_retinopathy, macular_edema, scar, nevus, amd,
vascular_occlusion, hypertensive_retinopathy, drusens, hemorrhage,
retinal_detachment, myopic_fundus, increased_cup_disc, other
```

Any additional columns (e.g. `patient_id`, `age`, `sex`, `camera`) are preserved and can be used for
subgroup / fairness analysis during evaluation.

> **Note:** `run_full_pipeline.py` automatically normalises the `diabetes` column from `yes`/`no` to
> `1`/`0` and writes a cleaned copy to `data/labels_brset_new.csv`, which the rest of the pipeline
> consumes.

### Using a different dataset

A helper script converts an *ImageFolder*-style dataset (one subfolder per class) into the
multi-label CSV expected here:

```bash
python src/preprocess_image_folder.py \
    --dataset-dir /path/to/imagefolder_dataset \
    --output-csv  data/labels_custom.csv
```

---

## Code Information

```
XAI_medimg_diag/
├── run_full_pipeline.py          # Orchestrates the end-to-end pipeline (phases 0–4)
├── run_all_paper_experiments.sh  # Full command catalog for the paper experiment suite
├── prefetch_torchvision_weights.py  # Pre-download/cache pretrained backbone weights
├── probe_batch_sizes.py          # Find the largest safe batch size per backbone/GPU
├── requirements.txt              # Python dependencies
├── notebook.ipynb                # Exploratory data analysis of the BRSET labels
└── src/
    ├── config.py                 # BRSET_LABELS — the 14 target pathologies
    ├── 0_build_cache.py          # Phase 0: resize all images once into image_cache.pth
    ├── 1_train_gate.py           # Phase 1: train the multi-label gate (Lightning)
    ├── 2_train_experts.py        # Phase 2: train the 14 binary experts (Lightning)
    ├── 3_calibrate_moe.py        # Phase 3: assemble + calibrate the Hybrid MoE (Lightning)
    ├── evaluate.py               # Phase 4: threshold tuning, metrics, subgroup/fairness report
    ├── models.py                 # ModelWrapper factory (backbones + LoRA/Q-LoRA head surgery)
    ├── moe_model.py              # HybridMoE: assembles gate + experts and fuses their logits
    ├── dataloader.py             # Datasets, cached image loading, split manifests
    ├── experiment_utils.py       # Seeding, run manifests, system-info capture
    ├── train.py                  # Standalone single-expert trainer with optional Optuna HPO
    ├── visualize_xai.py          # Grad-CAM (SmoothGrad-CAM++) + saliency map visualisation
    ├── preprocess_image_folder.py# Convert an ImageFolder dataset to the multi-label CSV format
    └── run_all_experts.sh        # Convenience loop to train every expert
```

### Key components

- **`models.ModelWrapper`** — creates a torchvision backbone, rewrites the classifier head for the
  requested number of classes, and (optionally) applies LoRA/Q-LoRA by auto-detecting target
  `Linear` modules. Supported backbones and sizes:

  | `--model-name` | `small` | `medium` | `base` | `large` / `tiny` |
  |----------------|---------|----------|--------|------------------|
  | `efficientnet` | B0 | B4 | – | B7 (`large`) |
  | `resnet`       | 18 | 50 | – | 101 (`large`) |
  | `convnext`     | – | – | base | `tiny`, `small`, `large` |
  | `swin`         | S | – | base | `tiny` |
  | `vit`          | B/16 | – | B/16 | L/16 (`large`) |

- **`moe_model.HybridMoE`** — holds the gate and a `ModuleList` of experts, loads their
  checkpoints (full weights or LoRA adapters), and fuses logits via one of four strategies:
  - `additive` — gate + expert logits (parameter-free);
  - `scalar_calibration` — learns per-class `alpha`, `beta`, `bias` (Platt-style calibration);
  - `soft_gating` — weights experts by the sigmoid of the gate logits;
  - `topk` — keeps only the top-*k* experts selected by the gate.

- **`evaluate.py`** — tunes a per-class decision threshold on the validation split, reports
  macro AUC / macro F1 and per-class scores on the test split, and computes subgroup metrics for
  fairness auditing.

---

## Requirements

- **Python** ≥ 3.9
- A **CUDA-capable GPU** is strongly recommended. LoRA/Q-LoRA additionally require `bitsandbytes`,
  which depends on a working CUDA toolchain.

Python dependencies (see `requirements.txt`):

```
torch          torchvision     lightning
optuna         mlflow          torchcam
peft           bitsandbytes    accelerate
pillow         pandas          scikit-learn   numpy
```

### Installation

```bash
# 1. Clone
git clone https://github.com/matglima/XAI_medimg_diag.git
cd XAI_medimg_diag

# 2. Create an environment (example with venv)
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Tip:** install the PyTorch build that matches your CUDA version first
> (see https://pytorch.org/get-started/locally/), then run `pip install -r requirements.txt`.
> `requirements.txt` is intentionally unpinned; pin versions in your own fork for exact
> reproducibility.

Optionally pre-download all pretrained backbone weights (useful on offline/cluster nodes):

```bash
python prefetch_torchvision_weights.py
```

---

## Usage Instructions

### Quick start — full pipeline

Place the data under `data/` as described above, then run the complete pipeline (build cache →
train gate → train experts → calibrate → evaluate):

```bash
python run_full_pipeline.py \
    --labels-path data/labels_brset_new.csv \
    --image-dir   data/fundus_photos \
    --checkpoint-dir checkpoints/demo \
    --gate-model-name efficientnet  --gate-model-size  medium \
    --expert-model-name efficientnet --expert-model-size small \
    --fusion-strategy scalar_calibration --fusion-only \
    --gate-epochs 10 --expert-epochs 10 --calibrate-epochs 5 \
    --batch-size-gate 128 --batch-size-expert 512 --calibrate-batch-size 16 \
    --eval-batch-size 64 --base-lr 1e-4 --calibrate-lr 1e-5 \
    --seed 42 --num-workers 12
```

Running `python run_full_pipeline.py` with **no arguments** executes a built-in default
configuration (EfficientNet, LoRA enabled, MLflow on) intended for notebook use.

Useful flags:

- `--use-lora` / `--use-qlora` — parameter-efficient fine-tuning for both gate and experts
  (`--gate-use-lora`, `--expert-use-lora`, etc. for finer control); `--lora-r` sets the rank.
- `--fusion-strategy {additive,scalar_calibration,soft_gating,topk}` and `--top-k`.
- `--use-mlflow --mlflow-uri <uri> --mlflow-experiment <name>` to enable experiment tracking.
- `--split-manifest-path <file>` to reuse a fixed train/val/test split across runs.

### Running phases individually

```bash
# Phase 0 — build the resized image cache (runs once)
python src/0_build_cache.py --labels-path data/labels_brset_new.csv --image-dir data/fundus_photos

# Phase 1 — train the multi-label gate
python src/1_train_gate.py    --labels-path data/labels_brset_new.csv --image-dir data/fundus_photos \
                              --output-dir checkpoints/gate --model-name efficientnet --model-size medium

# Phase 2 — train one expert (repeat per pathology, or use src/run_all_experts.sh)
python src/2_train_experts.py --labels-path data/labels_brset_new.csv --image-dir data/fundus_photos \
                              --target-label diabetic_retinopathy --output-dir checkpoints/experts \
                              --model-name efficientnet --model-size small

# Phase 3 — assemble and calibrate the Hybrid MoE
python src/3_calibrate_moe.py --labels-path data/labels_brset_new.csv --image-dir data/fundus_photos \
                              --gate-ckpt-path checkpoints/gate --expert-ckpt-dir checkpoints/experts \
                              --output-dir checkpoints/final_moe --fusion-strategy scalar_calibration

# Phase 4 — evaluate (threshold tuning + subgroup/fairness metrics)
python src/evaluate.py        --labels-path data/labels_brset_new.csv --image-dir data/fundus_photos \
                              --model-path checkpoints/final_moe/moe_calibrated_final.pth \
                              --subgroup-columns sex,camera --age-column age
```

> The phase scripts in `src/` resolve sibling modules by import (`from models import ...`), so run
> them from the repository root as shown (the orchestrator does this for you).

### Explainability (XAI)

Generate a Grad-CAM + saliency figure for a single expert checkpoint and image:

```bash
python src/visualize_xai.py \
    --model_name efficientnet --model_size small \
    --checkpoint_path checkpoints/experts/diabetic_retinopathy.pth \
    --image_path data/fundus_photos/<image_id>.png \
    --diagnosis diabetic_retinopathy \
    --output_path gradcam_dr.png
```

### Reproducing the full experiment suite

`run_all_paper_experiments.sh` is a **command catalog** (not a single run-all script): it lists
every gate baseline, MoE benchmark, fusion ablation, and LoRA ablation across backbones and seeds
(42/43/44), each line annotated with its purpose. Copy and run the commands you need. Use
`probe_batch_sizes.py` to choose safe batch sizes for your GPU before launching.

---

## Methodology

The pipeline proceeds in five phases:

1. **Cache building (`0_build_cache.py`).** All images are loaded once, resized to 256×256 in
   parallel, and serialised to `image_cache.pth` so subsequent phases avoid repeated disk I/O.

2. **Gate training (`1_train_gate.py`).** A single backbone is trained as a 14-way multi-label
   classifier using **focal loss** (to handle class imbalance), the AdamW optimiser, early stopping,
   and PyTorch Lightning. Data augmentation includes random crop (224×224), flips, and rotation;
   inputs are normalised with ImageNet statistics.

3. **Expert training (`2_train_experts.py`).** For each of the 14 pathologies, a binary classifier
   is trained on the same split, again with focal loss and macro-averaged validation metrics.

4. **MoE calibration (`3_calibrate_moe.py`).** The trained gate and experts are assembled into a
   `HybridMoE`. The chosen fusion strategy combines their logits; for `scalar_calibration`, a small
   set of per-class parameters (`alpha`, `beta`, `bias`) is learned at a low learning rate, optionally
   freezing the backbones (`--fusion-only`).

5. **Evaluation (`evaluate.py`).** Per-class decision thresholds are tuned on the **validation**
   split (grid search maximising F1), then applied to the **test** split. The script reports macro
   AUC, macro F1, per-class AUC/F1, and **subgroup metrics** for fairness auditing across metadata
   fields (e.g. sex, camera, age bins).

**Reproducibility safeguards:** global seeding across `random`, NumPy, PyTorch, and Lightning;
deterministic cuDNN; shared **split manifests** (JSON) so the gate, experts, calibration, and
evaluation all see identical train/val/test partitions; and a **run manifest** capturing arguments
and system/CUDA info for every pipeline launch.

**Parameter-efficient fine-tuning.** LoRA / Q-LoRA (via `peft` + `bitsandbytes`) can be enabled
independently for the gate and the experts, with a configurable rank, to study the accuracy/compute
trade-off against full fine-tuning.

---

## Citations

### This repository

If you use this code, please also cite the BRSET dataset and PhysioNet resources below.

### BRSET dataset (PhysioNet)

```bibtex
@misc{nakayama2024brset_physionet,
  author    = {Nakayama, L. F. and Goncalves, M. and Zago Ribeiro, L. and Santos, H. and
               Ferraz, D. and Malerbi, F. and Celi, L. A. and Regatieri, C.},
  title     = {A Brazilian Multilabel Ophthalmological Dataset (BRSET) (version 1.0.1)},
  year      = {2024},
  publisher = {PhysioNet},
  note      = {RRID:SCR\_007345},
  doi       = {10.13026/1pht-2b69},
  url       = {https://doi.org/10.13026/1pht-2b69}
}
```

### BRSET original publication

```bibtex
@article{nakayama2024brset_plos,
  author  = {Nakayama, Luis Filipe and Restrepo, David and Matos, Jo{\~a}o and
             Ribeiro, Lucas Zago and Malerbi, Fernando Korn and Celi, Leo Anthony and
             Regatieri, Caio Saito},
  title   = {{BRSET}: A Brazilian Multilabel Ophthalmological Dataset of Retina Fundus Photos},
  journal = {PLOS Digital Health},
  volume  = {3},
  number  = {7},
  pages   = {e0000454},
  year    = {2024},
  month   = {7},
  doi     = {10.1371/journal.pdig.0000454},
  note    = {PMID: 38991014; PMCID: PMC11239107}
}
```

### PhysioNet

```bibtex
@article{goldberger2000physionet,
  author  = {Goldberger, A. and Amaral, L. and Glass, L. and Hausdorff, J. and
             Ivanov, P. C. and Mark, R. and Mietus, J. E. and Moody, G. B. and
             Peng, C.-K. and Stanley, H. E.},
  title   = {{PhysioBank}, {PhysioToolkit}, and {PhysioNet}: Components of a New Research
             Resource for Complex Physiologic Signals},
  journal = {Circulation},
  volume  = {101},
  number  = {23},
  pages   = {e215--e220},
  year    = {2000},
  note    = {RRID:SCR\_007345}
}
```

When using BRSET, please cite **all three** references above, as requested by the data providers.

---

## License

This project is released under the **MIT License**. See the [`LICENSE`](LICENSE) file for the full
text.

> The MIT License applies to the **code in this repository only**. The BRSET dataset is distributed
> separately under its own PhysioNet license/terms — review and comply with those terms before use.

---

## Contribution Guidelines

Contributions are welcome. To propose a change:

1. Fork the repository and create a feature branch.
2. Keep changes focused; match the existing code style and file/phase conventions.
3. If you add a backbone, fusion strategy, or pipeline phase, update this README accordingly.
4. Do **not** commit data, images, model checkpoints, or credentials (these are git-ignored).
5. Open a pull request describing the change, its motivation, and how you tested it.

For questions or bug reports, please open an issue on the
[GitHub repository](https://github.com/matglima/XAI_medimg_diag).
