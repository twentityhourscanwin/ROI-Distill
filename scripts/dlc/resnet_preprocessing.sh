#!/usr/bin/env bash
set -euo pipefail

group="${1:-}"
case "${group}" in
  R50) config="b1_center_proposal" ;;
  R101) config="b1_center_proposal_r101" ;;
  H1) config="b1_center_proposal_r101_512x1408_f06_r10" ;;
  H2) config="b1_center_proposal_r101_512x1408_f03_r10" ;;
  H3) config="b1_center_proposal_r101_512x1408_f06_r05" ;;
  H4) config="b1_center_proposal_r101_512x1408_f03_r05" ;;
  *) printf "Usage: bash scripts/dlc/resnet_preprocessing.sh R50|R101|H1|H2|H3|H4 [native overrides...]\n" >&2; exit 2 ;;
esac
shift
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "${project_root}"
export PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
exec python tools/train.py --config "configs/experiments/${config}.yaml" "$@"
