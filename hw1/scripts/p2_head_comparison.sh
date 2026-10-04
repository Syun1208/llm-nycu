#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
RUN_DATE="${RUN_DATE:-$(date +%F)}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6

cd "${PROJECT_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_DATE}_p2_head_comparison}"

"${PYTHON}" main.py p2-heads \
    --epochs 20 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
