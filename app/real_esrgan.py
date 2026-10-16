"""Real-ESRGAN 4× super-resolution.

Carga el modelo ONNX público de Real-ESRGAN (SceneWorks) y lo convierte a
PyTorch *FX* mediante ``onnx2torch`` — una sola vez, con cache en memoria.

Por qué no onnxruntime ni basicsr/realesrgan:

* El modelo ONNX exportado por Real-ESRGAN contiene un nodo ``Resize``
  cuyo ``scales`` quedó en el slot de ``roi`` (bug de exportación con opset
  ≥ 13).  Ese mal estate hace **cuelgue** tanto en ``onnxruntime`` como en
  ``cv2.dnn``.
* El paquete ``realesrgan`` / ``basicsr`` importa
  ``torchvision.transforms.functional_tensor``, que fue **eliminado** en
  torchvision 0.26 (nuestra versión).  Instalarlos provocaría un conflicto
  irracional con PyTorch 2.11.

``onnx2torch`` traduce el gráfico ONNX a un ``torch.fx.GraphModule`` cuyos
inputs quedan correctamente mapeados, y la inferencia en CUDA funciona a
~90 ms para un frame 112×112 → 448×448, lo que permite ~10 FPS.

Dependencias de *runtime*: ``torch`` + ``onnx`` + ``onnx2torch``.
El modelo ONNX origen se resuelve a ``runs/models/real_esrgan_x4.onnx``; si
falta, ``build_esrgan.py`` lo descarga e instala.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "runs" / "models" / "real_esrgan_x4.onnx"

# El ONNX de Real-ESRGAN asume entrada y salida en rango [0, 1]
# (verificado: el output raw está centrado en ~0.47, rango [0, 1]).
# Normalizar a [-1, 1] (mean/std 0.5) aplasta el contraste y deja la SR
# pálida/fea — se opera directamente en [0, 1].


class RealESRGAN:
    """Inferencia 4× con Real-ESRGAN (modelo ONNX → PyTorch FX).

    El modelo FX se convierte de ONNX una sola vez y se cachea a nivel
    de clase; instancias múltiples comparten el mismo modelo en GPU.
    """

    _MODEL_SIZE = 112          # tamaño de referencia del warmup (cuadrado)
    _MAX_INPUT = 320           # lado mayor de la entrada en modo calidad. 320 a 320
                               # → la SR real (4×) llega exactamente a 1280 = _MAX_OUTPUT
                               # sin resize extra. A 224 el output 896² se estiraba a
                               # 1280×720 con un resize no uniforme que borraba la nitidez
                               # de la SR (~2.3× menos nítido; Laplacian 56 vs 103).
                               # Medido en RTX 5050: 112→~90 ms, 320-aspect→~463 ms
                               # (uploads/imagen/video toleran los 463 ms; en vivo se
                               # usa `fast` a 112).
    _MAX_OUTPUT = 1280         # tope de salida: foto 4K → 16K al 4× agota VRAM y el
                               # codificador JPEG; 1280 es suficiente para pantalla y
                               # YOLO (que encoge a 640 de todos modos). _MAX_INPUT*4
                               # == _MAX_OUTPUT, así que el cap casi nunca recorta.
    _UPLOAD_INPUT = 640        # uploads/imagen/video (offline): entrada hasta 640 px
                               # (4× puro: 640→2560). Calidad > velocidad, VRAM OK en
                               # RTX 5050; videos grandes se escalan a 640 antes del 4×.
    _UPLOAD_OUTPUT = 2560      # tope de salida offline. Evita 8K+ (foto 4K→16K)
                               # agotando VRAM/codificador, manteniendo 4× real.
    _cache: dict[str, torch.nn.Module] = {}

    def __init__(self, model_path: str | os.PathLike = DEFAULT_MODEL,
                 device: str = "auto"):
        if device in ("auto", "cuda", 0, "cuda:0"):
            self._cuda = torch.cuda.is_available()
            self._dev = "cuda" if self._cuda else "cpu"
        else:
            self._cuda = False
            self._dev = "cpu"

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Modelo Real-ESRGAN no encontrado: {model_path}. "
                "Ejecuta `setup.sh` o `python app/build_esrgan.py` "
                "para descargarlo."
            )

        # Convertir ONNX → PyTorch FX (cacheada por path)
        key = str(model_path)
        if key not in RealESRGAN._cache:
            import onnx
            from onnx2torch.converter import convert
            onnx_model = onnx.load(key)
            RealESRGAN._cache[key] = convert(onnx_model).eval()

        self._model = RealESRGAN._cache[key].to(self._dev)
        # Warmup con el tamaño de referencia
        with torch.no_grad():
            _ = self._model(
                torch.zeros(1, 3, self._MODEL_SIZE, self._MODEL_SIZE,
                            device=self._dev, dtype=torch.float32)
            )
            if self._cuda:
                torch.cuda.synchronize()

    @property
    def cuda(self) -> bool:
        return self._cuda

    @torch.no_grad()
    def enhance(self, frame: np.ndarray, fast: bool = False,
                for_upload: bool = False) -> np.ndarray:
        """Escala 4× un frame BGR uint8.

        Parámetros
        ----------
        ``frame``: array H×W×3, uint8, OpenCV BGR.
        ``fast``: prioriza velocidad (entrada 112 px, ~90 ms) para flujos en vivo.
          False prioriza calidad (entrada hasta 320/640 px, preservando aspecto)
          para uploads/imagen/video; a 320 el output 4× llega a 1280 px con nitidez
          real (~2.3× más nítido que el viejo cuadrado 224), a costa de ~463 ms.
        ``for_upload`` (offline): usa caps altos (entrada 640, salida 2560) para un
          4× real en videos/fotos sin tocar el live, que mantiene el cap de 1280.

        Retorna
        ----------
        uint8 BGR escalado 4×, shape (4·H, 4·W, 3).
        """
        h, w = frame.shape[:2]
        # El modelo FX acepta dimensiones dinámicas (h, w); el warmup usa 112×112.
        # Modo rápido (cámara en vivo, fast): 112×112 cuadrado (~90 ms); el display
        # muestra el frame nativo y la SR solo sirve para detectar.
        # Modo calidad (uploads/imagen/video): se preserva la relación de aspecto
        # alimentando al modelo un rectángulo cuyo lado mayor es _MAX_INPUT (320).
        # Antes se aplastaba a un cuadrado 224×224 distorsionando 16:9 → la SR salía
        # 896² y se estiraba a 1280×720 con un resize no uniforme que borraba la
        # nitidez (Laplacian ~56 vs ~103 preservando aspecto). Medido RTX 5050:
        # 112→~90 ms, 320-aspect→~463 ms.
        if fast or max(h, w) <= self._MODEL_SIZE:
            s = self._MODEL_SIZE
        else:
            _max_in = self._UPLOAD_INPUT if for_upload else self._MAX_INPUT
            s = max(self._MODEL_SIZE, min(max(h, w), _max_in))
        # Relacion de aspecto preservada en AMBOS modos: antes `fast` usaba un
        # cuadrado 112x112 que estiraba 16:9 y dejaba la SR en vivo borrosa.
        sc = s / max(h, w)
        in_w = max(2, int(round(w * sc))) // 2 * 2     # PixelShuffle necesita pares
        in_h = max(2, int(round(h * sc))) // 2 * 2

        inp = cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_CUBIC)

        # BGR → RGB, → [0, 1], → NCHW (contiguous)
        rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.ascontiguousarray(np.transpose(x[None], (0, 3, 1, 2)))
        x = torch.from_numpy(x).to(self._dev, dtype=torch.float32)

        # fp16 autocast: NO. El FX GraphModule de onnx2torch tiene ops de Resize
        # con interpolación bilineal cuyo gradiente/rounding en fp16 produce
        # shifts de color (el output se tiñe de azul en movimiento). En su lugar
        # se aprovecha la caché de cudnn + el warmup ya hecho. Medido RTX 5050:
        # el SR a 112×112 es ~90 ms (fp32) vs ~55 ms (fp16) — el 35 ms de ahorro
        # no compensa el artefacto cromático.
        y = self._model(x)                              # [1, 3, 4*in_h, 4*in_w]

        # NCHW → NHWC, [0, 1] → [0, 255], RGB → BGR
        out = y[0].permute(1, 2, 0).cpu().numpy()
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

        # Recorte por seguridad: un 4× de foto 4K → 16K agota VRAM y el codificador.
        # Se topa la salida a _MAX_OUTPUT manteniendo aspect-ratio; _MAX_INPUT*4 ==
        # _MAX_OUTPUT, así que normalmente `out` ya cabe en el tope sin recortar.
        target_w, target_h = w * 4, h * 4
        _max_out = self._UPLOAD_OUTPUT if for_upload else self._MAX_OUTPUT
        if max(target_h, target_w) > _max_out:
            sc = _max_out / max(target_h, target_w)
            target_w = max(1, int(target_w * sc))
            target_h = max(1, int(target_h * sc))
        if (out.shape[0], out.shape[1]) != (target_h, target_w):
            out = cv2.resize(out, (target_w, target_h),
                             interpolation=cv2.INTER_CUBIC)
        # Nitidez suave: la super-resolución + los redimensionamientos bilineales
        # tienden a apagar bordes ("imagen pálida/fea"); unsharpen restituye el
        # contraste local sin sobreprocesar, dejando la salida claramente más nítida.
        out = self._unsharp(out)
        return out

    @staticmethod
    def _unsharp(img: np.ndarray,
                 ksize: int = 3, amount: float = 0.4) -> np.ndarray:
        """Unsharp masking muy leve (amount<1 conserva la SR, no ruido)."""
        blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
        return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)

    @torch.no_grad()
    def enhance_to_size(self, frame: np.ndarray,
                        target: tuple[int, int]) -> np.ndarray:
        """Escala al tamaño objetivo (4× o ajustado a *target*)."""
        enhanced = self.enhance(frame)
        th, tw = target
        h, w = enhanced.shape[:2]
        if (h, w) != (th, tw):
            enhanced = cv2.resize(enhanced, (tw, th),
                                  interpolation=cv2.INTER_CUBIC)
        return enhanced
