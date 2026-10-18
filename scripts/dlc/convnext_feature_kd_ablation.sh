#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo 'Usage: convnext_feature_kd_ablation.sh off|f01' >&2
  exit 2
fi
case "$1" in
  off) config_name=convnext_feature_kd_off_v1 ;;
  f01) config_name=convnext_feature_kd_f01_v1 ;;
  *) echo 'Unsupported feature-KD selector' >&2; exit 2 ;;
esac

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "${project_root}"
export PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
IFS=, read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
if [[ ${#visible_devices[@]} -ne 16 ]]; then
  echo 'This experiment requires 16 visible GPUs for train and val' >&2
  exit 2
fi

run_id="${LABELDISTILL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
if [[ ! "${run_id}" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo 'LABELDISTILL_RUN_ID must use YYYYMMDD_HHMMSS' >&2
  exit 2
fi
export LABELDISTILL_RUN_ID="${run_id}"
checkpoint_root="${ROI_CHECKPOINT_ROOT:-/mnt/nas_data/guqiupeng/checkpoint_nes}"
run_name="${config_name}_${run_id}"
output_dir="${project_root}/outputs/${run_name}"
checkpoint_dir="${checkpoint_root}/${run_name}"
checkpoint="${checkpoint_dir}/epoch_23.ckpt"
if [[ -e "${output_dir}" || -e "${checkpoint_dir}" ]]; then
  echo "Run output already exists: ${run_name}" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo 'Tracked worktree changes are not allowed for a formal DLC run' >&2
  exit 2
fi
mkdir -p -- "${output_dir}"
{
  printf 'branch=%s\n' "$(git branch --show-current)"
  printf 'commit=%s\n' "$(git rev-parse HEAD)"
  printf 'run_id=%s\n' "${run_id}"
  printf 'config=%s\n' "configs/experiments/${config_name}.yaml"
  printf 'visible_gpus=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'checkpoint=%s\n' "${checkpoint}"
} > "${output_dir}/launcher_identity.txt"

if python tools/train.py \
  --config "configs/experiments/${config_name}.yaml" \
  "runtime.run_id=${run_id}" \
  "checkpoint.root_dir=${checkpoint_root}" \
  runtime.gpus=16 runtime.num_nodes=1 \
  runtime.batch_size_per_device=4 runtime.accumulate_grad_batches=2 \
  runtime.max_epochs=24; then
  printf '0\n' > "${output_dir}/train.exitcode"
else
  code=$?
  printf '%s\n' "${code}" > "${output_dir}/train.exitcode"
  exit "${code}"
fi

if [[ ! -s "${checkpoint}" || ! -s "${output_dir}/resolved_config.yaml" ]]; then
  echo 'Training returned zero but final checkpoint or resolved config is missing' >&2
  printf '1\n' > "${output_dir}/checkpoint_preflight.exitcode"
  exit 1
fi
printf '0\n' > "${output_dir}/checkpoint_preflight.exitcode"

if python scripts/dlc/evaluate_history.py \
  --history-fallback previous_valid \
  --config "${output_dir}/resolved_config.yaml" \
  --checkpoint "${checkpoint}" \
  --output-dir "${output_dir}/val_smoke" \
  --limit-test-batches 2 --prediction-only \
  runtime.gpus=16 runtime.num_nodes=1 runtime.batch_size_per_device=4 \
  evaluation.run_metrics=false; then
  printf '0\n' > "${output_dir}/val_smoke.exitcode"
else
  code=$?
  printf '%s\n' "${code}" > "${output_dir}/val_smoke.exitcode"
  exit "${code}"
fi

if python scripts/dlc/evaluate_history.py \
  --history-fallback previous_valid \
  --config "${output_dir}/resolved_config.yaml" \
  --checkpoint "${checkpoint}" \
  --output-dir "${output_dir}/val_full" \
  runtime.gpus=16 runtime.num_nodes=1 runtime.batch_size_per_device=4 \
  evaluation.run_metrics=true; then
  printf '0\n' > "${output_dir}/val_full.exitcode"
else
  code=$?
  printf '%s\n' "${code}" > "${output_dir}/val_full.exitcode"
  exit "${code}"
fi

if python scripts/dlc/verify_convnext_val.py \
  --config "${output_dir}/resolved_config.yaml" \
  --output-dir "${output_dir}/val_full" \
  --expected-count 6019; then
  printf '0\n' > "${output_dir}/val_audit.exitcode"
else
  code=$?
  printf '%s\n' "${code}" > "${output_dir}/val_audit.exitcode"
  exit "${code}"
fi
printf 'complete\n' > "${output_dir}/pipeline.status"
