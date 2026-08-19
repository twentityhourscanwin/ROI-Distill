#!/bin/bash
# ablation_param 训练脚本
# 用法: bash train_ablation_param.sh
# 默认 8 GPU, batch_size=4

EXPERIMENTS=(
    "param_J1_wl02_wh05"
    "param_J2_wl03_wh06"
    "param_J3_wl035_wh07"
    "param_J4_wl05_wh08"
    "param_J5_wl03_wh08"
    "param_mu_010"
    "param_mu_020"
    "param_P3_lambda_020"
    "param_P3_lambda_040"
    "param_P3_lambda_050"
    "param_P3_lambda_055"
    "param_P3_lambda_065"
    "param_P3_lambda_070"
    "param_P3_lambda_080"
    "param_P3_lambda_100"
)

EXP_CONFIG_DIR="labeldistill/exps/nuscenes/ablation_param"

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
