#!/usr/bin/env python3
"""
Plataforma web local para probar el detector de riesgo en escaleras.

Carga el checkpoint de Ultralytics y expone tres modos de prueba:

  * imagen  -> sube un frame y recibe el resultado anotado
  * video   -> sube un video y descarga la version anotada
  * camara  -> webcam en vivo por streaming MJPEG

Las detecciones se acompanan de un panel de riesgo que traduce cada clase a
la categoria de riesgo de `Caso.md` (RQF02), de forma que la prueba sirva
para validar el diseno y no solo la precision del detector.

Uso:
    ./venv/bin/python app/server.py
    # abrir http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import base64
import hmac
import json
import io
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from flask import (
    Flask, Response, jsonify, redirect, render_template, request, send_file,
)

ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Descripcion de cada clase, enlazada con los requisitos de `Caso.md`.
# `color` es BGR (como espera OpenCV).
# --------------------------------------------------------------------------- #
CLASS_INFO = {
    0: {
        "name": "Persona caída",
        "color": (60, 60, 235),        # rojo
        "risk": "ALTO",
        "req": "RQF02 (4) · RQNF05",
        "desc": "Persona tendida en el suelo tras una caída.",
    },
    1: {
        "name": "Postura no ergonómica",
        "color": (60, 170, 245),       # naranja
        "risk": "MEDIO",
        "req": "RQF03",
        "desc": "Postura sentada o agachada; no corresponde a tránsito de pie.",
    },
    2: {
        "name": "Postura erguida",
        "color": (90, 200, 90),        # verde
        "risk": "BAJO",
        "req": "RQF03",
        "desc": "Tránsito seguro: persona de pie o caminando.",
    },
    3: {
        "name": "Escalera",
        "color": (200, 120, 200),      # morado
        "risk": "CONTEXTO",
        "req": "—",
        "desc": "Tramo de escalera detectado en la escena.",
    },
    4: {
        "name": "Persona",
        "color": (220, 220, 220),      # blanco
        "risk": "NEUTRO",
        "req": "RQNF08",
        "desc": "Persona genérica. Es la entrada para el análisis de pose (MediaPipe).",
    },
    5: {
        "name": "Pérdida de equilibrio",
        "color": (40, 190, 255),       # amarillo
        "risk": "ALTO",
        "req": "RQF02 (4)",
        "desc": "Centro de gravedad inestable: pre-caída, antes de caer.",
    },
}

RISK_ORDER = {"ALTO": 3, "MEDIO": 2, "BAJO": 1, "NEUTRO": 0, "CONTEXTO": 0}

sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))
from risk_engine import RISK_COLOR, RiskEngine
from audio_live import install_audio

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024   # 512 MB
# Sin esto Flask cachea index.html en memoria y los cambios de la
# interfaz no se ven ni con recarga forzada del navegador: hay que
# reiniciar el servidor. Se deja activo porque la interfaz se itera
# mucho durante el desarrollo.
app.config["TEMPLATES_AUTO_RELOAD"] = True
audio_events, audio_sessions = install_audio(app, ROOT)

# El modelo se carga una sola vez y se protege con un lock: Flask atiende en
# varios hilos y dos inferencias simultaneas sobre la misma GPU pueden fallar.
class Inferencer:
    """Ejecuta inferencia Y dibujo en un unico hilo trabajador persistente.

    Por que no se hace directamente en el hilo de cada peticion: Flask atiende
    cada peticion en un hilo distinto (`threaded=True`, necesario para que el
    stream de la webcam no bloquee al resto), y en un hilo recien creado hay que
    reenlazar el contexto CUDA. Medido en esta maquina:

        inferencia   ~29 ms en hilo ya enlazado  ->  ~82 ms en hilo nuevo
        dibujo cv2    0.13 ms en hilo ya enlazado ->  ~23 ms en hilo nuevo

    Total: ~85 ms por peticion frente a ~27 ms con este hilo unico. El hilo se
    crea una vez al arrancar y attend todas las peticiones.

    Si ``enable_esrgan`` es True, cada frame se super-resuelve 4× con
    Real-ESRGAN antes de la detección.  La conversión ONNX → PyTorch FX es
    *lazy* y se hace dentro de este hilo para que el contexto CUDA quede
    ligado al worker (igual que el modelo YOLO).
    """

    def __init__(self, weights: str, device: int | str = "auto",
                 obstacle_weights: str = "yolov8s.pt", conf: float = 0.25,
                 enable_esrgan: bool = False):
        self._weights = weights
        self._device = device
        self._obstacle_weights = obstacle_weights
        self._conf = conf
        self._enable_esrgan = enable_esrgan
        self._esrgan: RealESRGAN | None = None
        self._last_sr_flag: bool = True  # para flush de cola al toggle SR
        # Estimación del ciclo SR (wall-clock entre key-frames) para programar
        self._q: queue.Queue = queue.Queue()
        # Modo asincrono para la camara en vivo: el navegador entrega el frame
        # y recibe ENSEGUIDA el ultimo resultado calculado, sin esperar a la
        # inferencia. Medido: dentro del servidor la inferencia tardaba 42 ms
        # frente a 26.7 ms en solitario, por la contencion con el hilo de
        # Flask; esperando de forma sincrona el navegador queda por debajo de
        # 20 fps aunque el motor de por si va a 37. Desacoplando, el navegador
        # no bloquea y la inferencia mantiene su ritmo.
        self._async = False
        self._last_async: dict | None = None
        # Lock único para el acceso al último resultado asíncrono.
        self._lock_async = threading.Lock()
        self._model = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()          # no se atiende hasta tener modelo

    def _run(self) -> None:
        from ultralytics import YOLO
        # Auto-detección de dispositivo: 'auto' usa CUDA si está disponible,
        # 'cpu' fuerza uso del procesador.
        if self._device == "auto":
            import torch
            self._device = 0 if torch.cuda.is_available() else "cpu"
        # el modelo se carga DENTRO del hilo: asi el contexto CUDA se crea y
        # queda ligado a este mismo hilo
        self._model = YOLO(self._weights)
        self._model.to(self._device)
        # RQF02 (1)(2)(3): motor de riesgo completo. Se construye aqui tambien
        # para que MediaPipe y el detector de obstaculos nauten en el hilo
        # enlazado, que es donde rinde bien.
        from risk_engine import RiskEngine
        self._engine = RiskEngine(
            self._weights, obstacle_weights=self._obstacle_weights,
            device=self._device, conf=self._conf, enable_pose=True,
            anonymize=True,
            pose_every=8, people_every=6,
            stairs_every=13, obstacle_every=11, classify_every=6,
            person_conf=0.20,
        )
        # Real-ESRGAN: super-resolución 4× (solo si --esrGAN).  Se carga
        # quede ligado a este mismo hilo (igual que YOLO).
        if self._enable_esrgan:
            from real_esrgan import RealESRGAN
            self._esrgan = RealESRGAN(device=self._device)
        try:
            import numpy as _np
            _wu = _np.full((540, 960, 3), 120, _np.uint8)
            _wu[130:460, 300:430] = 40
            self._engine.step(_wu, force=True, allow_alerts=False)
            self._engine.step(_wu, force=True, allow_alerts=False)
            self._engine.step(_wu, force=False, allow_alerts=False)
        except Exception as _e:
            print(f"  [aviso] warmup del motor fallo: {_e}", flush=True)
        self._ready.set()
        while True:
            job = self._q.get()
            if job is None:                       # señal de cierre
                return
            image, conf, imgsz, out = job
            try:
                if out.get("reset"):
                    self._engine.reset()
                    out["value"] = None
                else:
                    out["value"] = self._job(
                        image, conf, imgsz,
                        draw=not out.get("no_draw", False),
                        force=out.get("force", False),
                        allow_alerts=out.get("allow_alerts", True),
                        fast=out.get("fast", False),
                        sr=out.get("sr", True),
                        for_upload=out.get("for_upload", False))
                    # El ultimo resultado se guarda SIEMPRE, no solo en modo
                    # asincrono. Antes se guardaba bajo `if self._async`, pero
                    # la peticion restaura `_async = False` en su `finally`
                    # antes de que el worker termine: el cache se quedaba
                    # siempre vacio y la camara en vivo no devolvia NUNCA
                    # detecciones (siempre `pendiente: true`).
                    with self._lock_async:
                        self._last_async = out["value"]
            except BaseException as exc:          # noqa: BLE001
                # `finally` en vez de `except`: si `out["done"]` no fuera un
                # Event, el `.set()` lanzaba AttributeError y el hilo
                # terminaba. A partir de ahi TODAS las peticiones se
                # quedaban colgadas para siempre, porque nadie consumia la
                # cola. Con BaseException el hilo sobrevive a cualquier fallo.
                try:
                    out["error"] = exc
                finally:
                    try:
                        out["done"].set()
                    except Exception:
                        pass
            else:
                out["done"].set()

    def _job(self, image, conf, imgsz, draw=True, force=False,
             allow_alerts=True, fast=False, sr=True, for_upload=False):
        """Motor completo + dibujo, ambos dentro del hilo trabajador.

        El dibujo tambien se hace aqui a proposito: medir en esta maquina
        muestra que cualquier operacion de OpenCV ejecutada desde un hilo
        recien creado cuesta ~23 ms frente a 0.13 ms en el hilo ya enlazado.
        """
        if conf != self._conf:
            self._conf = conf
            self._engine.conf = conf
        t0 = time.perf_counter()
        self._engine._camara_connected = True
        # Super-resolución 4× (RQNF03-extra): aplicada ANTES de la detección para
        # que las bboxes coincidan con el frame escalado. En el live se pasa
        # `sr=False` para fluidar (~15-20 fps de cajas): el 4× calidad en RTX 5050
        # dura ~0.8-1.2 s/frame, incompatible con 30 fps. SR bajo demanda
        # (?sr=1) o en uploads/imagen vía /api/detect (donde sr=True por defecto).
        if self._esrgan is not None and sr:
            image = self._esrgan.enhance(image, fast=fast, for_upload=for_upload)
        result = self._engine.step(image, force=force,
                                   allow_alerts=allow_alerts)
        t_predict = (time.perf_counter() - t0) * 1000

        # En el modo `overlay=0` (camara en vivo) el navegador dibuja las cajas
        # sobre el frame que ya tiene: dibujar aqui es trabajo que se tira.
        # Medido, son ~5 ms por fotograma, y es el 10% del presupuesto.
        t1 = time.perf_counter()
        out = image.copy() if draw else image
        detections = draw_all(out, result) if draw else [
            {**d, "class_name": CLASS_INFO.get(d["class_id"], {}).get("name", d["class_name"]),
             "risk": d.get("riesgo", "NEUTRO")}
            for d in result["detecciones"]
        ] + [{"tipo": s["tipo"], "confidence": s["conf"], "risk": s["riesgo"],
              "ref": s["ref"], "detalle": s.get("detalle", "")}
             for s in result["postura"]] + [
            {**o, "confidence": o["conf"], "risk": o.get("riesgo", "CONTEXTO"),
             "bbox": o["bbox"], "ref": "RQF02 (1)"} for o in result["obstaculos"]]
        # La evidencia de las alertas se guarda YA DIBUJADA: dentro de
        # step() todavia no se han pintado las cajas.
        self._engine.flush_evidence(out)
        t_draw = (time.perf_counter() - t1) * 1000
        return out, result, detections, {
            "predict": round(t_predict, 1), "draw": round(t_draw, 1),
        }

    def engine_reset(self) -> None:
        """Pide al motor limpiar sus caches desde su propio hilo.

        El trabajo se encola con la MISMA forma que el resto: el cuarto
        elemento del tuple es el diccionario de resultado y su clave "done"
        debe ser el Event. Antes se encolaba un dict anidado
        ({"done": {"done": Event}}) y el worker hacia `out["done"].set()`
        sobre un dict: AttributeError que MATABA el hilo para siempre y
        dejaba cualquier peticion posterior colgada indefinidamente.
        """
        out = {"done": threading.Event(), "reset": True}
        self._q.put((None, None, None, out))
        out["done"].wait()

    def set_async(self, on: bool) -> None:
        self._async = on

    def detect(self, image: np.ndarray, conf: float, imgsz: int, draw: bool = True,
               force: bool = False, allow_alerts: bool = True, fast: bool = False,
               sr: bool = True, for_upload: bool = False):
        """Encola el trabajo y espera el resultado.

        En modo asincrono devuelve de inmediato el ultimo resultado disponible
        (o `None` si aun no hay ninguno). El frame se encola igualmente, asi
        que el motor sigue trabajando al ritmo que puede.

        ``fast`` fuerza la vía rápida de Real-ESRGAN (112 px) para que la
        cámara en vivo mantenga fluidez; uploads y video por lotes usan
        calidad (hasta 160 px) para que la salida 4× se vea nítida.
        ``sr`` (default True) aplica Real-ESRGAN; live pasa sr=False para fluidar
        a 30 fps (SR on-demand ?sr=1 o /api/detect de uploads).
        """
        out = {"done": threading.Event(), "no_draw": not draw, "force": force,
               "allow_alerts": allow_alerts, "fast": fast, "sr": sr,
               "for_upload": for_upload, "async": self._async}
        # "Solo el ultimo frame importa": si el motor no llega al ritmo de la
        # webcam, la cola crece y el resultado llega cada vez mas rancio. Se
        # descartan los frames en espera cuando se acumulan.
        while self._q.qsize() > 3:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        # Flush de cola al cambiar de modo SR↔normal: si el worker está
        # procesando un frame SR (~200 ms) y el browser ya envía normal
        # (sr=False), los jobs SR viejños encolados bloquean el worker y
        # la GIL contention durante ESGRAN/YOLO retrasa las respuestas async
        # → fps cae a 5 y las cajas aparecen en posiciones obsoletas.
        if self._last_sr_flag != sr:
            self._last_sr_flag = sr
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
        self._q.put((image, conf, imgsz, out))
        if self._async:
            # Devolver el último resultado ya calculado sin bloquear: el worker
            # sigue procesando el frame siguiente y el navegador siempre ve el
            # más reciente disponible. Un lock → read-modify-write atómico.
            with self._lock_async:
                return self._last_async
        out["done"].wait()
        if "error" in out:
            raise out["error"]
        return out["value"]

    def shutdown(self) -> None:
        self._q.put(None)


_inferencer = None
_settings = {"conf": 0.25, "imgsz": 640, "weights": "", "names": {}}




# --------------------------------------------------------------------------- #
# Codificacion
# --------------------------------------------------------------------------- #
def draw_all(frame: np.ndarray, result: dict) -> list[dict]:
    """Dibuja el resultado completo del motor de riesgo sobre el fotograma.

    `detecciones`   cajas del modelo propio (persona / escalera)
    `obstaculos`    objetos COCO sobre la escalera   -> RQF02 (1)
    `postura`       señales de keypoints             -> RQF02 (2)(3)
    `alertas`       historial de alertas confirmadas -> RQF07
    """
    out = []
    for d in result.get("detecciones", []):
        info = CLASS_INFO.get(
            d["class_id"],
            {"name": d["class_name"], "color": (200, 200, 200), "risk": "NEUTRO"},
        )
        # El color y el texto siguen el riesgo REAL de esa persona, que puede
        # venir de una señal de postura (tambaleo, distracción, pasamanos) y no
        # solo de su clase. Antes la caja decía "persona erguida" en verde
        # mientras el veredicto era ALTO.
        riesgo = d.get("riesgo") or info["risk"]
        color = RISK_COLOR.get(riesgo, info["color"])
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        tid = d.get("track_id")
        partes = [f'#{tid}'] if tid else []
        partes.append(info["name"])
        senales = d.get("senales") or []
        if senales:
            partes.append("+".join(s.replace("_", " ") for s in senales))
        partes.append(f'{d["confidence"]:.0%}')
        label = " ".join(partes)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        ty = max(y1, th + 6)
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 8, ty), color, -1)
        cv2.putText(frame, label, (x1 + 4, ty - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (20, 20, 20), 2, cv2.LINE_AA)
        out.append({**d, "class_name": info["name"], "risk": riesgo,
                    "color": color})

    for o in result.get("obstaculos", []):
        # El mobiliario se dibuja en color de contexto: es(scene), no riesgo.
        # Los objetos abandonados sobre el escalon van en rojo y si suman al
        # veredicto de riesgo.
        es_riesgo = o.get("riesgo") == "ALTO"
        col = (RISK_COLOR["MEDIO"] if o.get("tipo") == "telefono"
               else RISK_COLOR["ALTO"] if es_riesgo else RISK_COLOR["CONTEXTO"])
        prefijo = "TELEFONO" if o.get("tipo") == "telefono" else (
            "OBSTACULO" if es_riesgo else "ESCENA")
        x1, y1, x2, y2 = o["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        label = f'{prefijo} {o["objeto"]} {o["conf"]:.0%}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 8, y1),
                      col, -1)
        cv2.putText(frame, label, (x1 + 4, max(th + 2, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2, cv2.LINE_AA)
        out.append({"tipo": o["tipo"], "objeto": o["objeto"],
                    "confidence": o["conf"],
                    "risk": "ALTO" if es_riesgo else "CONTEXTO",
                    "bbox": o["bbox"],
                    "ref": "RQF02 (1)" if es_riesgo else "escena"})

    for s in result.get("postura", []):
        col = RISK_COLOR.get(s["riesgo"], (220, 220, 220))
        label = f'{s["tipo"].upper()} {s["conf"]:.0%} [{s["ref"]}]'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        y = frame.shape[0] - 6 - th
        cv2.rectangle(frame, (6, y - th - 6), (6 + tw + 8, y), col, -1)
        cv2.putText(frame, label, (10, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (20, 20, 20), 2, cv2.LINE_AA)
        out.append({"tipo": s["tipo"], "confidence": s["conf"],
                    "risk": s["riesgo"], "ref": s["ref"],
                    "detalle": s.get("detalle", "")})
    return out


def encode_jpeg(image: np.ndarray, quality: int = 82) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("no se pudo codificar la imagen")
    return buf.tobytes()


def encode_b64(image: np.ndarray, quality: int = 82) -> str:
    """JPEG base64. Se usa en el stream en vivo: servir ya base64 (en vez de
    hex) evita que el navegador haga un bucle parseInt de ~44 k chars por
    frame, que bloqueaba el hilo principal y destruía la fluidez."""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("no se pudo codificar la imagen")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# --------------------------------------------------------------------------- #
# Resumen de riesgo
# --------------------------------------------------------------------------- #
def respuesta(dets, result, timing, **extra) -> "Response":
    """Construye la respuesta de TODAS las rutas de analisis en un solo sitio.

    Antes el mismo JSON de 10 claves estaba escrito en tres rutas (`/api/detect`,
    `/api/frame` con y sin overlay) y fue el origen de una divergencia real: el
    movil salia como CONTEXTO en una y como MEDIO en otra, porque cada copia
    calculaba el riesgo a su manera. Con un unico constructor no puede volver a
    pasar.
    """
    cuerpo = {
        "detections": dets,
        "summary": summarise(dets),
        "riesgo": result["riesgo"],
        "postura": result["postura"],
        "obstaculos": result["obstaculos"],
        "alertas": result["alertas"],
        "tracks": result["tracks"],
        "sistema": result["sistema"],
        "timing": timing,
    }
    cuerpo.update(extra)
    return jsonify(cuerpo)


def summarise(detections: list[dict]) -> dict:
    """Traduce las detecciones a un veredicto de riesgo segun `Caso.md`."""
    # `detections` mezcla cajas de YOLO (con `class_name`), obstaculos y
    # señales de postura (que traen `tipo` en su lugar). Antes se accedia
    # `class_name` a pelo y reventaba con KeyError en cuanto habia un obstaculo.
    counts = Counter(d.get("class_name") or d.get("tipo", "?")
                     for d in detections)
    risks = {d["risk"] for d in detections
             if d.get("risk") not in ("CONTEXTO", "NEUTRO", None)}
    level = max(risks, key=lambda r: RISK_ORDER.get(r, 0), default="BAJO")

    notes = {
        "ALTO": "Se ha detectado una condición de caída o pérdida de equilibrio. "
                "Requiere alerta inmediata (RQF06).",
        "MEDIO": "Postura no ergonómica detectada. Conviene revisar (RQF04).",
        "BAJO": "No se detectan conductas de riesgo en la escena.",
    }
    return {
        "n": len(detections),
        "level": level,
        "riesgo": level,
        "counts": dict(counts),
        "message": notes.get(level, notes["BAJO"]),
    }


# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return render_template("index.html", classes=CLASS_INFO)


@app.post("/api/detect")
def api_detect():
    if "file" not in request.files:
        return jsonify({"error": "no se ha enviado ningún archivo"}), 400
    raw = request.files["file"].read()
    if not raw:
        return jsonify({"error": "archivo vacío"}), 400

    conf = float(request.form.get("conf", _settings["conf"]))
    imgsz = int(request.form.get("imgsz", _settings["imgsz"]))

    arr = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "no se pudo decodificar la imagen"}), 400

    # Cada peticion de imagen es una escena nueva: se limpian las cache de
    # fotograma para que no hereden obstaculos ni escalera de la imagen previa.
    _inferencer.engine_reset()
    t0 = time.perf_counter()
    # Imagen fija: se ejecutan pose, personas, escalera y obstaculos SIEMPRE.
    # Sin esto el movil se detectaba 1 de cada 15 analisis, porque los
    # obstaculos solo se evaluan en los fotogramas que tocan su cadencia, y al
    # recargar la misma imagen no cambia nada: la escena es la misma.
    annotated, result, dets, timing = _inferencer.detect(
        image, conf, imgsz, draw=True, force=True, allow_alerts=False)
    t_all = (time.perf_counter() - t0) * 1000
    t_enc = time.perf_counter()
    payload = encode_jpeg(annotated).hex()
    timing["encode"] = round((time.perf_counter() - t_enc) * 1000, 1)
    timing["total"] = round(t_all, 1)

    return respuesta(dets, result, timing,
                     inference_ms=timing["predict"],
                     size=[int(annotated.shape[1]), int(annotated.shape[0])],
                     image=payload)


@app.post("/api/frame")
def api_frame():
    """Inferencia sobre un unico frame JPEG.

    Es el endpoint que usa el modo «cámara de este equipo»: el navegador
    captura con `getUserMedia`, pinta el frame en un canvas y lo envia aqui
    frame a frame. Hace falta porque la webcam del usuario esta en su
    navegador, no en esta maquina (que no tiene ninguna: no existe
    /dev/video*). El stream MJPEG de /api/stream solo sirve si el servidor
    tiene camara fisica.
    """
    # Dos formas de enviar el frame:
    #   * cuerpo binario crudo con Content-Type image/jpeg  -> rápido
    #   * multipart/form-data con un campo `file`          -> compatible
    #
    # El camino crudo existe por rendimiento: Werkzeug parsea multipart en
    # Python y con un JPEG de 50 KB costaba ~17 ms por fotograma, casi la
    # mitad del presupuesto. Mandando el JPEG como cuerpo crudo y la
    # configuracion en la query string, esa cadena desaparece.
    raw_body = None
    if request.mimetype == "image/jpeg" and "file" not in request.files:
        raw_body = request.get_data(cache=False)
    if "file" in request.files:
        raw_body = request.files["file"].read()
    if not raw_body:
        return jsonify({"error": "no se ha enviado ningún frame"}), 400
    raw = raw_body

    conf = float(request.args.get("conf", request.form.get("conf", _settings["conf"])))
    imgsz = int(request.args.get("imgsz", request.form.get("imgsz", _settings["imgsz"])))
    overlay = request.args.get(
        "overlay", request.form.get("overlay", "1")) not in ("0", "false")

    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "no se pudo decodificar el frame"}), 400

    # Cámara en vivo: SIEMPRE asíncrono. Se purga el modo frame-by-frame
    # síncrono (sync=1): el navegador recibe el último SR+cajas calculado en
    # cuanto lo pide (~10 ms), dejando el display fluido a 30 fps con un
    # retardo de ~0.8-1 s (el productor SR 4× calidad tarda ~0.8 s/frame).
    # `sync=1` pasa a ser un no-op; sirve sólo para `/api/detect` (imagen fija).
    _inferencer.set_async(bool(not overlay))
    # SR on-demand en vivo: OFF por defecto (4× calidad ~0.8-1.2 s/frame rompe
    # 30 fps). ?sr=1 lo fuerza; /api/detect mantiene SR siempre.
    live_sr = overlay or request.args.get("sr", "0") == "1"
    try:
        got = _inferencer.detect(image, conf, imgsz, draw=overlay, fast=False,
                                 sr=live_sr)
    finally:
        _inferencer.set_async(False)
    if got is None:
        return jsonify({"detections": [], "summary": summarise([]),
                        "riesgo": "BAJO", "timing": {"predict": 0, "draw": 0},
                        "pendiente": True})
    annotated, result, dets, timing = got
    if overlay:
        annotate_status(annotated, dets, 0.0, result)
        return respuesta(dets, result, timing,
                         image=encode_jpeg(annotated, quality=72).hex())

    # Modo `overlay=0` (cámara en vivo): se devuelven las cajas normalizadas a
    # 0..1. El navegador ya tiene el frame nativo en su propio canvas (`small`)
    # y solo necesita las cajas para dibujarlas encima; por eso **no se envía
    # la imagen** en modo sin SR. Codificar + transferir + parsear + decodificar
    # una JPEG-base64 de ~40 KB por fotograma (~70% del JSON de respuesta) era
    # el cuello que hundía el live a mitad de cadencia: el cliente quedaba
    # esperando `readJson` + `new Image().onload` en cada frame, con lo que
    # `cam.busy` se mantenía bloqueado y se perdían fotogramas. Sin la imagen,
    # la respuesta es un JSON pequeño y el loop avanza a la cadencia del `setTimeout`
    # (30 fps reales).
    #
    # Solo cuando `live_sr` es True (SR on-demand, ?sr=1) el frame SR 4× es
    # diferente al nativo que el navegador posee, y hay que enviarlo para que el
    # display muestre la super-resolución. Las cajas vienen normalizadas sobre
    # `annotated`, que en modo sr=False coincide con el frame original (w×h),
    # idéntico al canvas `small`.
    h, w = annotated.shape[:2]
    # Solo las detecciones con caja se normalizan a 0..1. Las señales de
    # postura (`sin_pasamanos`, `distraccion`, `tambaleo`) NO tienen caja: son
    # condiciones descritas con keypoints, no objetos que se puedan encerrar,
    # y asumirlas todas revienta con KeyError en cuanto aparece una.
    norm = []
    for d in dets:
        if "bbox" in d:
            norm.append({**d, "bbox": [
                round(d["bbox"][0] / w, 4), round(d["bbox"][1] / h, 4),
                round(d["bbox"][2] / w, 4), round(d["bbox"][3] / h, 4),
            ]})
        else:
            norm.append(d)
    cuerpo = {
        "detections": norm,
        "summary": summarise(dets),
        "riesgo": result["riesgo"],
        "postura": result["postura"],
        "obstaculos": result["obstaculos"],
        "alertas": result["alertas"],
        "tracks": result["tracks"],
        "sistema": result["sistema"],
        "timing": timing,
        "size": [w, h],
    }
    # SR on-demand: enviar el frame super-resuelto porque es diferente al nativo.
    if live_sr:
        cuerpo["image"] = encode_b64(annotated, quality=72)
    return jsonify(cuerpo)


def annotate_status(frame: np.ndarray, dets: list[dict], fps: float = 0.0,
                    result: dict | None = None) -> None:
    """Superpone la banda de estado: FPS, nº de detecciones y nivel de riesgo.

    Si se pasa el resultado del motor, el veredicto es el suyo, que ya
    incorpora postura y obstaculos; si no, se deduce solo de las cajas.
    """
    summary = summarise(dets)
    if result:
        summary["n"] = len(dets) + len(result.get("obstaculos", [])) \
            + len(result.get("postura", []))
        summary["level"] = result["riesgo"]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (30, 30, 30), -1)
    color = {
        "ALTO": (60, 60, 235), "MEDIO": (60, 170, 245),
        "BAJO": (90, 200, 90), "NEUTRO": (90, 200, 90),
    }.get(summary["level"], (200, 200, 200))
    cv2.putText(frame, f'FPS: {fps:.1f}   detecciones: {summary["n"]}',
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255),
                1, cv2.LINE_AA)
    cv2.putText(frame, f'Riesgo: {summary["level"]}',
                (10, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


@app.get("/api/alerts")
def api_alerts():
    """Historial de alertas para el panel de supervision (RQF08).

    Lee `runs/alerts/<AAAA-MM-DD>/` y `alertas.jsonl`, de modo que el panel
    sobrevive a un reinicio del servidor.
    """
    d = RiskEngine.ALERT_DIR
    if not d.is_dir():
        rows = audio_events.read()[::-1]
        return jsonify({"alertas": rows[:100], "total": len(rows)})
    lineas = []
    jl = d / "alertas.jsonl"
    if jl.is_file():
        for ln in jl.read_text(encoding="utf-8", errors="replace").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                a = json.loads(ln)
            except Exception:
                continue
            ev = a.get("evidence") or ""
            rel = Path(ev).name
            dia = Path(ev).parent.name
            existe = rel and (Path(ev).exists())
            a["imagen"] = f"/api/alerta-imagen/{dia}/{rel}" if existe else None
            lineas.append(a)
    lineas.extend(audio_events.read())
    lineas.sort(key=lambda a: a.get("timestamp", ""), reverse=True)
    return jsonify({"alertas": lineas[:100], "total": len(lineas)})


def _alertes() -> list[dict]:
    """Lee el registro de alertas (reutilizado por las rutas de borrado)."""
    d = RiskEngine.ALERT_DIR
    jl = d / "alertas.jsonl"
    if not jl.is_file():
        return []
    out = []
    for ln in jl.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


@app.delete("/api/alertas")
def api_borrar_alertas():
    """Borra alertas.

    Sin cuerpo borra TODAS; con `{"ids": [...]}` borra solo las elegidas.
    Se eliminan tanto la imagen de evidencia como su linea del registro, para
    que no queden alertas apuntando a ficheros que ya no existen.
    """
    d = RiskEngine.ALERT_DIR
    cuerpo = request.get_json(silent=True) or {}
    ids = cuerpo.get("ids") or request.args.getlist("id") or None
    audio_deleted = audio_events.delete(ids)

    if not ids:
        # borrar todo
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        (d / "alertas.jsonl").touch()
        return jsonify({"borradas": "todas", "total": 0})

    borradas = audio_deleted
    conservadas = []
    for a in _alertes():
        if a.get("alert_id") not in ids:
            conservadas.append(a)
            continue
        ev = a.get("evidence") or ""
        try:
            destino = Path(ev).resolve()
            if str(destino).startswith(str(d.resolve()) + "/") and destino.is_file():
                destino.unlink()
        except Exception:
            pass          # la imagen ya no estaba: se ignora
        borradas += 1
    d.mkdir(parents=True, exist_ok=True)
    with (d / "alertas.jsonl").open("w", encoding="utf-8") as fh:
        for a in conservadas:
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")
    # limpia carpetas de fecha que se hayan quedado vacias
    for sub in d.iterdir() if d.is_dir() else []:
        if sub.is_dir() and not any(sub.iterdir()):
            try:
                sub.rmdir()
            except OSError:
                pass
    return jsonify({"borradas": borradas, "total": len(conservadas) + len(audio_events.read())})


@app.get("/api/alerta-imagen/<dia>/<nombre>")
def api_alerta_imagen(dia: str, nombre: str):
    """Sirve la imagen de una alerta, siempre dentro de `runs/alerts/`."""
    base = RiskEngine.ALERT_DIR.resolve()
    destino = (base / dia / nombre).resolve()
    if not str(destino).startswith(str(base) + "/") or not destino.is_file():
        return jsonify({"error": "no encontrada"}), 404
    return send_file(str(destino), mimetype="image/jpeg")


@app.get("/api/alertas.zip")
def api_alertas_zip():
    """Descarga TODAS las alertas en un unico ZIP (imagenes + jsonl).

    Un solo boton en la interfaz para llevarse el paquete completo, tal y como
    se pidio. Se genera al vuelo en memoria: no se deja ningun ZIP en disco.
    """
    import io as _io
    import zipfile

    d = RiskEngine.ALERT_DIR
    buf = _io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        n = 0
        if d.is_dir():
            for f in sorted(d.glob("*/*.jpg")):
                z.write(f, arcname=f"{f.parent.name}/{f.name}")
                n += 1
            jl = d / "alertas.jsonl"
            if jl.is_file():
                z.write(jl, arcname="alertas.jsonl")
        audio_rows = audio_events.read()
        if audio_rows:
            z.writestr("alertas_audio.jsonl", "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in audio_rows))
            n += len(audio_rows)
        z.writestr("README.txt",
                   "Paquete de alertas del sistema de prevencion de caidas.\n"
                   "Las imagenas estan anotadas con las detecciones y los rostros\n"
                   "pixelizados (RQNF14, Ley 29733).\n"
                   "alertas.jsonl lleva una linea por alerta con fecha, clase,\n"
                   "severidad, confianza, duracion e identidad.\n")
    buf.seek(0)
    if n == 0:
        return jsonify({"error": "aun no hay alertas registradas"}), 404
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name="alertas.zip")


@app.get("/api/cameras")
def api_cameras():
    """Informa de si el SERVIDOR tiene camara.

    La webcam del usuario la usa el navegador, asi que esto no la restringe:
    solo evita que se ofrezca un modo de camara que no puede funcionar.
    """
    found = []
    for i in range(4):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append({"index": i,
                          "name": cap.getBackendName() if hasattr(cap, "getBackendName") else f"cam {i}"})
        cap.release()
    return jsonify({"server_has_camera": bool(found), "devices": found})


def _transcode_h264(src: Path, dst: Path,
                    audio_source: Path | None = None) -> tuple[bool, str]:
    """Recodifica a H.264 con el ffmpeg empaquetado en imageio-ffmpeg.

    `libx264` con `+faststart` deja el indice al principio del archivo, que es
    lo que permite empezar a reproducir antes de que termine de descargar.

    Si `audio_source` apunta a un video con pista de audio, se extrae y
    se mezcla en la salida (video anotado con audio sincronizado).
    """
    try:
        import subprocess

        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()

        cmd = [exe, "-y", "-loglevel", "error"]
        if audio_source and audio_source.is_file():
            cmd += ["-i", str(src), "-i", str(audio_source)]
            cmd += [
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:a", "aac", "-b:a", "128k",
                str(dst),
            ]
        else:
            cmd += ["-i", str(src)]
            cmd += [
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst),
            ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return (r.returncode == 0 and dst.is_file()
                and dst.stat().st_size > 0, r.stderr[-200:])
    except Exception as exc:       # noqa: BLE001
        return False, str(exc)


@app.post("/api/video")
def api_video():
    if "file" not in request.files:
        return jsonify({"error": "no se ha enviado ningún archivo"}), 400
    raw = request.files["file"].read()
    if not raw:
        return jsonify({"error": "archivo vacío"}), 400

    conf = float(request.form.get("conf", _settings["conf"]))
    imgsz = int(request.form.get("imgsz", _settings["imgsz"]))

    # OpenCV 5 escribe siempre a disco: no acepta un buffer en memoria ni
    # como fuente (VideoCapture) ni como destino (VideoWriter). Se usan dos
    # temporales, uno para la entrada y otro para la salida, y se limpian al
    # terminar.
    suffix = Path(request.files["file"].filename or "entrada.mp4").suffix
    if suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
        suffix = ".mp4"
    tmp = Path(tempfile.gettempdir()) / f"sub_{uuid.uuid4().hex}{suffix}"
    tmp.write_bytes(raw)

    cap = cv2.VideoCapture(str(tmp))
    if not cap.isOpened():
        tmp.unlink(missing_ok=True)
        return jsonify({"error": "no se pudo decodificar el video"}), 400

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Codec de salida.
    #
    # OpenCV escribe `mp4v` (MPEG-4 Part 2) y los navegadores NO lo reproducen:
    # sale un reproductor con boton de play y pantalla negra.
    #
    # Este build de OpenCV (4.10, fijado por el conflicto con MediaPipe) no
    # trae ningun codificador que el navegador acepte: VP8 y VP9 abren el
    # VideoWriter pero escriben archivo vacio, y H.264 no existe. La solucion
    # es escribir con OpenCV y recodificar con el ffmpeg que trae
    # `imageio-ffmpeg`, que si incluye libx264.
    out_tmp = Path(tempfile.gettempdir()) / f"raw_{uuid.uuid4().hex}.mp4"
    final_tmp = Path(tempfile.gettempdir()) / f"out_{uuid.uuid4().hex}.mp4"
    # El writer se abre en DEMORA (lazy) sobre el PRIMER frame anotado: si
    # --esrgan está activo el frame es 4× mas grande que el original, y abrir el
    # writer con (w, h) original escribe un mp4 corrupto (pálido, estirado,
    # 0 frames). Usando annotated.shape[1::-1] el tamaño del writer siempre
    # coincide con el del frame que se escribe.
    writer = None
    out_size = (w, h)

    total_counts, frames, t_infer = Counter(), 0, 0.0

    engine = _inferencer._engine

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            annotated, result, _d, _t = _inferencer.detect(frame, conf, imgsz, sr=True, for_upload=True)
            engine._video_frame_no = frames  # frame actual (0-indexed)
            t_infer += time.perf_counter() - t0
            frames += 1
            total_counts.update(x.get("class_name") or x.get("tipo", "?")
                                 for x in _d)
            total_counts.update(o["tipo"] for o in result.get("obstaculos", []))
            if writer is None:
                out_size = (annotated.shape[1], annotated.shape[0])
                writer = cv2.VideoWriter(
                    str(out_tmp),
                    cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size,
                )
                if not writer.isOpened():
                    cap.release()
                    tmp.unlink(missing_ok=True)
                    return jsonify({
                        "error": f"no se pudo preparar la salida de video ({out_size[0]}×{out_size[1]})"
                    }), 400
            writer.write(annotated)
            if frames % 200 == 0:
                print(f"  {frames} frames...", flush=True)

        cap.release()
        if writer is not None:
            writer.release()
        # OpenCV deja MPEG-4 Part 2, que ningun navegador reproduce: se
        # recodifica a H.264, que si es universal.
        ok, err = _transcode_h264(out_tmp, final_tmp, audio_source=tmp)
        if not ok:
            print(f"  [aviso] ffmpeg fallo ({err}); se devuelve el intermedio")
        payload = (final_tmp if final_tmp.is_file() else out_tmp)
        payload = payload.read_bytes() if payload.is_file() else b""
    finally:
        if hasattr(engine, "_video_fps"):
            del engine._video_fps
        if hasattr(engine, "_video_frame_no"):
            del engine._video_frame_no
        tmp.unlink(missing_ok=True)
        out_tmp.unlink(missing_ok=True)
        final_tmp.unlink(missing_ok=True)

    if frames == 0 or not payload:
        return jsonify({"error": "el video no contenía frames utilizables"}), 400

    return send_file(
        io.BytesIO(payload), mimetype="video/mp4", as_attachment=True,
        download_name="resultado.mp4",
    )


def _mjpeg(camera_index: int, conf: float, imgsz: int):
    """Generador del stream de la webcam."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        yield b""
        return
    
    fps_t, fps_n, fps = time.perf_counter(), 0, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, result, dets, _t = _inferencer.detect(frame, conf, imgsz)
            fps_n += 1
            elapsed = time.perf_counter() - fps_t
            if elapsed >= 0.5:
                fps = fps_n / elapsed
                fps_n, fps_t = 0, time.perf_counter()

            annotate_status(annotated, dets, fps, result)

            frame_bytes = encode_jpeg(annotated, quality=80)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()


@app.get("/api/stream")
def api_stream():
    conf = float(request.args.get("conf", _settings["conf"]))
    imgsz = int(request.args.get("imgsz", _settings["imgsz"]))
    cam = int(request.args.get("cam", 0))
    probe = cv2.VideoCapture(cam)
    opened = probe.isOpened()
    probe.release()
    if not opened:
        return jsonify({
            "error": "El servidor no tiene cámara. Usa el modo "
                     "«Cámara de este equipo», que usa la webcam de tu navegador."
        }), 503
    return Response(
        _mjpeg(cam, conf, imgsz),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# --------------------------------------------------------------------------- #
# Autenticacion (RQNF19)
#
# `Caso.md` exige "autenticación de usuario para acceso al panel". Se
# implementa con un secreto compartido que se compara en tiempo constante.
# El secreto NUNCA se escribe en el repositorio: se lee de la variable de
# entorno PANEL_TOKEN y, si no esta, el panel arranca en modo local sin
# autenticacion y lo dice en pantalla en lugar de fallar en silencio.
# --------------------------------------------------------------------------- #
PANEL_TOKEN = os.environ.get("PANEL_TOKEN", "")
AUTH_ON = bool(PANEL_TOKEN)
TOKEN_TTL = 8 * 3600


def _is_public_request() -> bool:
    """El stream y las imagenes de la propia pagina no exigen token.

    Se aceptan solo peticiones desde el propio navegador de la maquina, que es
    el escenario de desarrollo; cualquier otro origen va con token.
    """
    return request.remote_addr in ("127.0.0.1", "::1")


@app.before_request
def _guard():
    if not AUTH_ON:
        return None
    if request.path in ("/", "/login", "/api/health"):
        return None
    if request.path.startswith("/static/"):
        return None
    if request.path == "/api/stream" and _is_public_request():
        return None
    supplied = request.headers.get("X-Panel-Token", "") or request.cookies.get("panel", "")
    if supplied and hmac.compare_digest(supplied, PANEL_TOKEN):
        return None
    if request.path.startswith("/api/"):
        return jsonify({"error": "token de panel requerido (RQNF19)"}), 401
    return redirect("/login")


@app.route("/login", methods=["GET", "POST"])
def login():
    if not AUTH_ON:
        return redirect("/")
    if request.method == "POST":
        sent = request.form.get("token", "")
        if sent and hmac.compare_digest(sent, PANEL_TOKEN):
            resp = redirect("/")
            resp.set_cookie("panel", PANEL_TOKEN, httponly=True,
                            max_age=TOKEN_TTL, samesite="Lax")
            return resp
        return render_template("login.html", error="Token incorrecto"), 401
    return render_template("login.html", error=None)


@app.get("/api/health")
def health():
    detalle = {}
    if _inferencer is not None and getattr(_inferencer, "_engine", None):
        detalle = _inferencer._engine.estado_detallado()
    return jsonify({
        "ok": True,
        "estado": detalle.get("estado", "DESCONOCIDO"),
        "detalle": detalle,
        "weights": _settings["weights"],
        "device": "cuda:0" if _cv_cuda() else "cpu",
        "classes": len(CLASS_INFO),
        "auth": "activo" if AUTH_ON else "desactivado (PANEL_TOKEN sin definir)",
        "esrgan": (
            "activo (cuda)" if getattr(_inferencer, "_esrgan", None) and _inferencer._esrgan.cuda
            else "activo (cpu)" if getattr(_inferencer, "_esrgan", None)
            else "desactivado (--esrgan)" if getattr(_inferencer, "_enable_esrgan", False)
            else "desactivado"
        ),
    })


def _cv_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/yolov8s_zoom/weights/best.pt")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--obstacle-weights", default="yolov8s.pt",
                    help="Modelo COCO para RQF02 (1): obstaculos en la escalera")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--device", default="auto",
                    help="Dispositivo: 'auto', 'cpu', 'cuda', o índice GPU (0, 1...)")
    ap.add_argument("--esrgan", action="store_true",
                    help="Super-resolución Real-ESRGAN 4× antes de la detección "
                         "(~+120 ms/frame en RTX 5050).")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = ROOT / weights
    if not weights.is_file():
        raise SystemExit(f"No se encuentra el checkpoint: {weights}")

    import yaml
    data_yaml = ROOT / "datasets/combined/data.yaml"
    names = {}
    if data_yaml.is_file():
        raw = yaml.safe_load(data_yaml.read_text()).get("names", {})
        names = {int(k): v for k, v in raw.items()} if isinstance(raw, dict) else dict(enumerate(raw))
    _settings.update(weights=str(weights), conf=args.conf, imgsz=args.imgsz, names=names)

    print("=" * 60)
    print("Plataforma de pruebas - Prevencion de Caidas en Escaleras")
    print("=" * 60)
    print(f"  pesos   : {weights.relative_to(ROOT) if weights.is_relative_to(ROOT) else weights}")
    print(f"  clases  : {len(names)} ({', '.join(names.values())})")
    print(f"  conf    : {args.conf}   imgsz: {args.imgsz}")
    print(f"  device  : {args.device}")
    print(f"  URL     : http://{args.host}:{args.port}")
    print("=" * 60)

    global _inferencer
    print("  cargando pesos en el hilo trabajador...")
    _inferencer = Inferencer(
        str(weights), conf=args.conf,
        obstacle_weights=args.obstacle_weights,
        device=args.device,
        enable_esrgan=args.esrgan,
    )
    print("  modelo listo")
    print("  precalentando detector acústico...")
    try:
        audio_sessions.warmup()
        print("  detector acústico listo (sin grabaciones)")
    except Exception as exc:
        # Audio opcional: su fallo nunca impide usar el módulo de video.
        print(f"  audio no disponible: {exc}")
    # Precalentar la inferencia fría: la primera llamada a RiskEngine inicializa
    # MediaPipe/pose (~10 s) y carga el grafo TFLite solo si el frame tiene
    # contenido real (un `np.zeros` no dispara la carga completa del intérprete
    # y el primer frame en vivo pagaba ese coste al usuario). Se fuerza un frame
    # con contenido (figura humana) a la resolución real del live (960×540) para
    # que el primer frame en vivo no haga stall.
    print("  precalentando inferencia (cold start)...")
    try:
        _wu = np.full((540, 960, 3), 120, np.uint8)
        _wu[130:460, 300:430] = 40
        _inferencer.detect(_wu, args.conf, args.imgsz,
                           draw=False, force=True, allow_alerts=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  warmup omitido: {exc}")
    # Limpio el frame de test del warmup: que el primer frame en vivo no devuelva
    # una SR de arranque (1280). Con SR OFF el live usa el nativo (480/640).
    with _inferencer._lock_async:
        _inferencer._last_async = None
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
