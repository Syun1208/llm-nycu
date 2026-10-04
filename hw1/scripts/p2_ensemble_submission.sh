#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
RUN_DATE="${RUN_DATE:-$(date +%F)}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5

cd "${PROJECT_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_DATE}_p2_v9}"

"${PYTHON}" main.py p2 \
    --preset v9 \
    --epochs 20 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
