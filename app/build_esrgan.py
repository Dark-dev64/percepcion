"""Descarga el modelo ONNX de Real-ESRGAN x4.

Ejecutar una sola vez (o cuando falta el modelo::

    python app/build_esrgan.py

El modelo ONNX público proviene de ``SceneWorks/real-esrgan-onnx`` en
HuggingFace y se guarda en ``runs/models/real_esrgan_x4.onnx``.

La conversión ONNX → PyTorch FX la hace ``RealESRGAN`` en ``app/real_esrgan.py``
de forma *lazy* y cacheada, así que este script sólo descarga el ONNX.
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "models" / "real_esrgan_x4.onnx"
REPO = "SceneWorks/real-esrgan-onnx"
FILENAME = "real_esrgan_x4.onnx"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.is_file():
        print(f"✓ Modelo ya existe: {OUT}")
        return 0

    print(f"Descargando {REPO}/{FILENAME} ...")
    cached = hf_hub_download(REPO, FILENAME, force_download=True)
    OUT.write_bytes(Path(cached).read_bytes())
    size = OUT.stat().st_size / (1024 * 1024)
    print(f"✓ Guardado en {OUT}  ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
