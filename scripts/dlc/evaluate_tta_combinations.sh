#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "${project_root}"
export PYTHONPATH="${project_root}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "${PYTHON:-python}" -u -m labeldistill.evaluation.tta_combinations "$@"
