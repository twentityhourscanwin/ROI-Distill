#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 1 ]] || { echo 'Usage: convnext_training_ablation.sh multistep|cosine36|bev256 [train overrides...]' >&2; exit 2; }
case "$1" in
  multistep) config=convnext_multistep_v1 ;;
  cosine36) config=convnext_cosine36_v1 ;;
  bev256) config=convnext_bev256_v1 ;;
  *) echo 'Unsupported experiment selector' >&2; exit 2 ;;
esac
shift
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "${project_root}"
export PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
exec python tools/train.py --config "configs/experiments/${config}.yaml" "$@"
