#!/bin/bash
# -----------------------------------------------------------------
# File: run_all_experts.sh
# -----------------------------------------------------------------
# Description:
# Helper script to run 2_train_experts.py for all pathologies.
# -----------------------------------------------------------------

# --- CONFIGURE THIS LIST ---
# This MUST match the list in 1_train_gate.py
PATHOLOGIES=(
    "diabetic_retinopathy"
    "age_related_macular_degeneration"
    "glaucoma"
    "hypertensive_retinopathy"
    "pathology_5"
    "pathology_6"
    "pathology_7"
    "pathology_8"
    "pathology_9"
    "pathology_10"
    "pathology_11"
    "pathology_12"
    "pathology_13"
    "pathology_14"
)

# --- CONFIGURE MODEL AND TRAINING PARAMS ---
MODEL_NAME="resnet"
MODEL_SIZE="small"
LABELS_CSV="data/labels_brset.csv"
IMAGE_DIR="data/fundus_photos"
EPOCHS=25
BATCH_SIZE=32
LR=1e-4
USE_LORA=false
USE_QLORA=false

# --- Build command flags ---
LORA_FLAGS=""
if [ "$USE_LORA" = true ]; then
    LORA_FLAGS="--use-lora"
    if [ "$USE_QLORA" = true ]; then
        LORA_FLAGS="$LORA_FLAGS --use-qlora"
    fi
fi

# --- Loop and Train ---
for pathology in "${PATHOLOGIES[@]}"; do
    echo "================================================================"
    echo "Starting training for expert: $pathology"
    echo "================================================================"
    
    python 2_train_experts.py \
        --labels-path $LABELS_CSV \
        --image-dir $IMAGE_DIR \
        --target-label "$pathology" \
        --model-name $MODEL_NAME \
        --model-size $MODEL_SIZE \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        $LORA_FLAGS

    echo "Finished training for $pathology."
    echo "================================================================"
done

echo "All expert training complete."