#!/bin/bash

# 定义实验组名称
EXPERIMENTS=("ablation_A0_baseline_full_gt" 
             "ablation_A1_channel_split" 
             "ablation_A2_cs_roi" 
             "ablation_A3_full_method" 
             "ablation_A4_roi_only" 
             "ablation_A5_roi_scale")

# 基础路径配置
EXP_CONFIG_DIR="labeldistill/exps/nuscenes/ablation_module"
OUTPUT_BASE_DIR="outputs"
CHECKPOINT_NAME="epoch_epoch=23-v1.ckpt"

# 循环执行验证
for EXP in "${EXPERIMENTS[@]}"
do
    echo "=========================================================="
    echo "Starting evaluation for: $EXP"
    echo "=========================================================="

    # 拼接完整的路径
    CONFIG_PATH="${EXP_CONFIG_DIR}/${EXP}.py"
    CKPT_PATH="${OUTPUT_BASE_DIR}/${EXP}/checkpoints/${CHECKPOINT_NAME}"

    # 检查权重文件是否存在，防止中途报错
    if [ -f "$CKPT_PATH" ]; then
        python "$CONFIG_PATH" \
            --gpus 2 \
            --evaluate \
            --ckpt_path "$CKPT_PATH"
    else
        echo "Error: Checkpoint not found at $CKPT_PATH"
    fi

    echo -e "Finished $EXP\n"
done
