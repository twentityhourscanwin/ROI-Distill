#!/bin/bash
# 参数消融实验批量评估脚本
# 用法: bash run_ablation_param.sh

# 定义实验组名称
EXPERIMENTS=(
    "param_J1_wl02_wh05"
    "param_J2_wl03_wh06"
    "param_J3_wl035_wh07"
    "param_J4_wl05_wh08"
    "param_J5_wl03_wh08"
    "param_P3_lambda_020"
    "param_P3_lambda_040"
    "param_P3_lambda_050"
    "param_P3_lambda_070"
    "param_P3_lambda_080"
    "param_P3_lambda_100"
)

# 基础路径配置
EXP_CONFIG_DIR="labeldistill/exps/nuscenes/ablation_param"
OUTPUT_BASE_DIR="outputs"
CHECKPOINT_NAME="epoch_epoch=23.ckpt"

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