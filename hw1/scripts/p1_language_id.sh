#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
RUN_DATE="${RUN_DATE:-$(date +%F)}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=7

cd "${PROJECT_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_DATE}_p1_language_id}"

"${PYTHON}" main.py p1 \
    --epochs 40 \
    --output-dir "${OUTPUT_DIR}" \
    "$@"
