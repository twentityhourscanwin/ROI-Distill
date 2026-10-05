#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "${project_root}"
export PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

exec python tools/train.py \
  --config configs/experiments/b1_center_proposal_convnextb_896x1600.yaml \
  runtime.gpus=16 runtime.num_nodes=1 \
  runtime.batch_size_per_device=4 runtime.accumulate_grad_batches=2 \
  runtime.max_epochs=24 "$@"
