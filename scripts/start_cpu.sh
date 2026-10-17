#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PERCEPCION_VENV="${PERCEPCION_VENV:-$HOME/.local/share/percepcion/venv}"
if [[ ! -x "$PERCEPCION_VENV/bin/python" ]]; then
    printf 'Falta el entorno permanente. Ejecuta primero: bash scripts/setup_cpu.sh\n' >&2
    exit 1
fi
for model in runs/yolov8s_zoom/weights/best.pt yolov8s.pt audio_fall_detector/models/fall_classifier.joblib; do
    if [[ ! -f "$model" ]]; then
        printf 'Falta el modelo: %s\n' "$model" >&2
        exit 1
    fi
done
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec "$PERCEPCION_VENV/bin/python" -u app/server.py --device cpu "$@"
