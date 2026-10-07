#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo 'Usage: temporal_fallback.sh R50|R101|CONVNEXT current|previous_valid [train overrides...]' >&2
  exit 2
fi
arch="${1,,}"
mode="$2"
shift 2
case "${arch}" in
  r50)
    batch=16; accum=1
    [[ "${mode}" == current ]] || { echo 'R50 reuses its existing control.' >&2; exit 2; }
    ;;
  r101)
    batch=8; accum=1
    [[ "${mode}" == current ]] || { echo 'R101 reuses its existing control.' >&2; exit 2; }
    ;;
  convnext|convnextb)
    arch=convnextb; batch=4; accum=2
    [[ "${mode}" == current || "${mode}" == previous_valid ]] || exit 2
    ;;
  *) echo "Unsupported architecture: ${arch}" >&2; exit 2 ;;
esac

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "${project_root}"
export PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"

exec python tools/train.py \
  --config "configs/experiments/temporal_fallback_${arch}_${mode}.yaml" \
  runtime.gpus=16 runtime.num_nodes=1 \
  "runtime.batch_size_per_device=${batch}" "runtime.accumulate_grad_batches=${accum}" \
  runtime.max_epochs=24 "$@"
