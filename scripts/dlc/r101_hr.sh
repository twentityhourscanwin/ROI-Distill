#!/usr/bin/env bash
set -euo pipefail

group="${1:-}"
case "${group}" in
  H1) suffix=f06_r10 ;;
  H2) suffix=f03_r10 ;;
  H3) suffix=f06_r05 ;;
  H4) suffix=f03_r05 ;;
  *) printf "Usage: bash scripts/dlc/r101_hr.sh H1|H2|H3|H4 [native overrides...]\n" >&2; exit 2 ;;
esac
shift

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "${project_root}"
export PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

exec python tools/train.py \
  --config "configs/experiments/b1_center_proposal_r101_512x1408_${suffix}.yaml" \
  runtime.gpus=16 runtime.num_nodes=1 \
  runtime.batch_size_per_device=8 runtime.accumulate_grad_batches=1 \
  runtime.max_epochs=24 "$@"
