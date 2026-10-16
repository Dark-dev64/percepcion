#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

ROOT="$(pwd)"
echo "============================================================"
echo "  Setup - Sistema de Prevención de Caídas en Escaleras (V1)"
echo "  Rama: supuesto-ideal / tag: supuesto-ideal-v1.0"
echo "  Directorio: $ROOT"
echo "============================================================"

# --- Detectar GPU NVIDIA ---
HAS_NVIDIA=0
GPU_NAME=""
VRAM=0
if command -v nvidia-smi &>/dev/null; then
    HAS_NVIDIA=1
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [ "$VRAM" -lt 6000 ]; then
        echo "[ADVERTENCIA] VRAM detectada: ${VRAM} MB. Recomendado >= 6 GB."
    fi
    echo "✓ GPU detectada: $GPU_NAME (${VRAM} MB VRAM)"
else
    echo "⚠ No se detectó NVIDIA. Se usará modo CPU (más lento)."
    echo "  MediaPipe Pose ya corre en CPU por defecto."
fi

# --- Verificar Python 3.9+ ---
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python3 no encontrado."
    exit 1
fi
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Python: $PYVER"

# --- Crear entorno virtual ---
if [ ! -d venv ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
fi
echo "✓ venv listo"

PIP="$ROOT/venv/bin/pip"

# --- Instalar PyTorch (CUDA o CPU según GPU) ---
echo ""
if [ "$HAS_NVIDIA" -eq 1 ]; then
    echo "Instalando PyTorch + CUDA 12.8..."
    $PIP install --extra-index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.11.0+cu128" "torchvision==0.26.0+cu128" 2>&1 | tail -1
    echo "✓ PyTorch (CUDA) instalado"
else
    echo "Instalando PyTorch (CPU-only)..."
    $PIP install --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.6.0+cpu" "torchvision==0.21.0+cpu" 2>&1 | tail -1
    echo "✓ PyTorch (CPU) instalado"
fi

# --- Instalar dependencias (en orden para evitar conflicto protobuf/mediapipe) ---
echo ""
echo "Instalando dependencias..."
echo "  (nota: mediapipe 0.10.21 requiere protobuf<5; se instala por separado)"

# Paquetes comunes (sin torch/torchvision, ni protobuf/mediapipe)
$PIP install \
    "ultralytics==8.4.170" \
    "opencv-python==4.10.0.84" \
    "pillow==12.3.0" \
    "numpy==1.26.4" \
    "matplotlib==3.11.2" \
    "PyYAML==6.0.3" \
    "Flask==3.1.3" \
    "Werkzeug==3.1.5" \
    "imageio-ffmpeg==0.6.0" \
    "psutil==7.2.2" \
    "polars==1.44.2" \
    "polars-runtime-32==1.44.2" \
    "scipy==1.16.3" \
    "sympy==1.14.0" \
    "networkx==3.6.1" \
    "filelock==3.32.3" \
    "fsspec==2026.7.0" \
    "setuptools==78.1.0" \
    "six==1.17.0" \
    "Jinja2==3.1.6" \
    "MarkupSafe==3.0.3" \
    "mpmath==1.3.0" \
    "requests==2.34.2" \
    2>&1 | tail -1
echo "  ✓ Paquetes principales instalados"

# MediaPipe + protobuf compatible (<5)
$PIP install "mediapipe==0.10.21" "protobuf>=4.25.3,<5" 2>&1 | tail -1
echo "  ✓ MediaPipe + protobuf corregido"

# Real-ESRGAN super-resolution (build + runtime)
# onnxruntime se usaba antes, pero el modelo ONNX exportado tiene un bug de
# Resize (scales en slot de roi) que hace cuelgue en CUDA. onnx2torch convierte
# el ONNX a PyTorch FX y la inferencia se hace con PyTorch CUDA directamente.
$PIP install "onnx==1.20.0" "onnx2torch==1.5.15" "huggingface_hub==2.2.0" 2>&1 | tail -1
echo "  ✓ Real-ESRGAN deps (onnx + onnx2torch)"

# --- Descargar modelo Real-ESRGAN ONNX ---
if [ ! -f "$ROOT/runs/models/real_esrgan_x4.onnx" ]; then
    echo "Descargando modelo Real-ESRGAN x4 (ONNX, ~64 MB)..."
    "$ROOT/venv/bin/python" "$ROOT/app/build_esrgan.py" 2>&1 | tail -2
    if [ ! -f "$ROOT/runs/models/real_esrgan_x4.onnx" ]; then
        echo "  [ADVERTENCIA] No se pudo descargar el modelo Real-ESRGAN."
        echo "  Super-resolución desactivada. Use --esrgan para reactivarla."
    else
        echo "  ✓ Modelo Real-ESRGAN listo"
    fi
else
    echo "  ✓ Modelo Real-ESRGAN ya presente"
fi

# CUDA utilities (solo si hay GPU)
if [ "$HAS_NVIDIA" -eq 1 ]; then
    $PIP install "cuda-bindings==12.9.7" "cuda-pathfinder==1.6.0" "triton==3.6.0" 2>&1 | tail -1
    echo "  ✓ CUDA utilities"
fi

# --- Audio: mismo entorno del servidor, sin FastAPI ni PANNs ---
$PIP install -r "$ROOT/requirements-audio.txt"

# --- Verificar modelos ---
echo ""
echo "Verificando modelos..."
MISSING=""

if [ -f "$ROOT/yolov8s.pt" ]; then
    echo "  ✓ yolov8s.pt presente (modelo COCO para obstáculos)"
fi

if [ -f "$ROOT/runs/yolov8s_zoom/weights/best.pt" ]; then
    SIZE=$(du -h "$ROOT/runs/yolov8s_zoom/weights/best.pt" | cut -f1)
    echo "  ✓ runs/yolov8s_zoom/weights/best.pt presente ($SIZE, 6 clases)"
else
    echo "  ⨯ best.pt no encontrado - entrenar con:"
    echo "    ./venv/bin/python scripts/train.py --set name=yolov8s_zoom"
    MISSING="model"
fi

# --- Datasets (info para el usuario) ---
echo ""
echo "Datasets:"
echo "  stairs_fallhuman:  $( [ -d 'stairs_fallhuman' ] && echo '✓ presente' || echo '✗ clonar: git clone https://huggingface.co/datasets/andres31416/stairs_fallhuman' )"
echo "  Le2i-raw:          $( [ -d 'datasets/_external/Le2i-raw' ] && echo '✓ presente' || echo '✗ descargar con: ./venv/bin/python scripts/prepare_le2i.py --help' )"
echo "  Combined dataset:  $( [ -d 'datasets/combined' ] && echo '✓ generado' || echo '✗ generar con: ./venv/bin/python scripts/prepare_dataset.py --clean' )"

# --- Verificación final ---
echo ""
echo "============================================================"
if [ -z "$MISSING" ]; then
    DEVICE_FLAG="--device auto"
    ESRGAN_FLAG=""
    if [ "$HAS_NVIDIA" -eq 1 ] && [ -f "$ROOT/runs/models/real_esrgan_x4.onnx" ]; then
        ESRGAN_FLAG="--esrgan"
        echo "  ✓ Real-ESRGAN 4× activado (usa --esrgan para activar/desactivar)"
    fi
    if [ "$HAS_NVIDIA" -eq 0 ]; then
        DEVICE_FLAG="--device cpu"
        echo "  ✓ LISTO (modo CPU) - Inicie el servidor:"
    else
        echo "  ✓ LISTO (modo GPU) - Inicie el servidor:"
    fi
    echo ""
    echo "  ./venv/bin/python app/server.py $DEVICE_FLAG $ESRGAN_FLAG --host 127.0.0.1 --port 5000"
    echo ""
    echo "  Abra: http://127.0.0.1:5000"
    echo "============================================================"
    exit 0
else
    echo "  ✗ FALTA: modelo entrenado."
    echo "           Ejecute scripts/train.py antes de iniciar."
    echo "============================================================"
    exit 1
fi
