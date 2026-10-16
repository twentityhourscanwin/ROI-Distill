#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 1 ]] || { echo 'Usage: convnext_recipe_diagnostics.sh e24_lr025|e24_lr100|e36_lr025|e36_lr100 [train overrides...]' >&2; exit 2; }
case "$1" in
  e24_lr025|e24_lr100|e36_lr025|e36_lr100) config="convnext_recipe_${1}_v1" ;;
  *) echo 'Unsupported experiment selector' >&2; exit 2 ;;
esac
shift
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "${project_root}"
export PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
exec python tools/train.py --config "configs/experiments/${config}.yaml" "$@"
