#!/bin/bash
# ablation_module 训练脚本
# 用法: bash train_ablation_module.sh
# 默认 8 GPU, batch_size=4

EXPERIMENTS=(
    "ablation_A0_baseline_full_gt"
    "ablation_A1_channel_split"
    "ablation_A2_cs_roi"
    "ablation_A3_full_method"
    "ablation_A4_roi_only"
    "ablation_A5_roi_scale"
)

EXP_CONFIG_DIR="labeldistill/exps/nuscenes/ablation_module"

GPUS=${1:-8}
BATCH_SIZE=${2:-4}

for EXP in "${EXPERIMENTS[@]}"
do
    echo "=========================================================="
    echo "Starting training for: $EXP"
    echo "  GPUs: $GPUS  Batch size: $BATCH_SIZE"
    echo "=========================================================="

    CONFIG_PATH="${EXP_CONFIG_DIR}/${EXP}.py"

    python "$CONFIG_PATH" --gpus "$GPUS" -b "$BATCH_SIZE"

    echo -e "Finished training $EXP\n"
done
