#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# /tmp se limpia; el entorno de uso diario vive en la carpeta del usuario.
PERCEPCION_VENV="${PERCEPCION_VENV:-$HOME/.local/share/percepcion/venv}"
PYTHON="${PERCEPCION_PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Se necesita Python 3.12; selecciona el ejecutable con PERCEPCION_PYTHON"'
if [[ ! -x "$PERCEPCION_VENV/bin/python" ]]; then
    "$PYTHON" -m venv --without-pip "$PERCEPCION_VENV"
fi
if ! "$PERCEPCION_VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$PERCEPCION_VENV/get-pip.py"
    "$PERCEPCION_VENV/bin/python" "$PERCEPCION_VENV/get-pip.py"
fi
"$PERCEPCION_VENV/bin/python" -m pip install --upgrade pip
"$PERCEPCION_VENV/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0+cpu torchvision==0.21.0+cpu
"$PERCEPCION_VENV/bin/python" -m pip install -r requirements-cpu.txt
"$PERCEPCION_VENV/bin/python" -m pip check
printf '\nInstalacion lista. Inicia con: bash scripts/start_cpu.sh\n'
