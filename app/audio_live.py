"""Sesiones de audio acotadas, independientes del trabajador de video.

PCM float32 LE mono, 16 kHz, bloques de 500 ms. Los IDs aleatorios son
capacidades efímeras; todas las rutas heredan la autenticación del panel.
No se almacena audio. El registro JSONL contiene solamente metadatos.
"""
from __future__ import annotations

import atexit
import json
import logging
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request

LOGGER = logging.getLogger(__name__)
MESSAGE = 'ALERTA: Se ha detectado un sonido compatible con una posible caída.'


class AudioEventStore:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()

    def read(self):
        with self.lock:
            if not self.path.exists():
                return []
            rows = []
            for line in self.path.read_text(encoding='utf-8').splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
            return rows

    def append(self, event):
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')

    def delete(self, ids=None):
        with self.lock:
            rows = self.read()
            keep = [r for r in rows if ids is not None and r['alert_id'] not in ids]
            if self.path.exists():
                self.path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in keep), encoding='utf-8')
            return len(rows) - len(keep)


class AudioSession:
    def __init__(self, detector_factory, store, idle_timeout=15):
        self.id = uuid.uuid4().hex
        self.store = store
        self.idle_timeout = idle_timeout
        self.last_seen = time.monotonic()
        self.stopped = threading.Event()
        self.lock = threading.RLock()
        self.queue = queue.Queue(maxsize=4)
        self.state = 'iniciando'
        self.error = None
        self.last_sequence = -1
        self.processed = 0
        self.dropped = 0
        self.processing_ms = 0.0
        self.rms = 0.0
        self.last_event_offset = -float('inf')
        self.worker = threading.Thread(target=self._run, args=(detector_factory,), daemon=True,
                                       name=f'audio-{self.id[:8]}')
        self.worker.start()

    def submit(self, sequence, offset, samples):
        with self.lock:
            if self.stopped.is_set():
                raise KeyError(self.id)
            if sequence <= self.last_sequence:
                raise ValueError('Secuencia duplicada o fuera de orden')
            self.last_sequence = sequence
            self.last_seen = time.monotonic()
            try:
                self.queue.put_nowait((sequence, offset, samples))
            except queue.Full:
                self.dropped += 1
                return False
            return True

    def snapshot(self):
        with self.lock:
            return dict(stream_id=self.id, state=self.state, error=self.error,
                        processed_chunks=self.processed, dropped_chunks=self.dropped,
                        processing_ms=round(self.processing_ms, 2), rms=round(self.rms, 6))

    def stop(self):
        with self.lock:
            self.stopped.set()
            self.state = 'detenido'
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass

    def _run(self, factory):
        detector = None
        try:
            detector = factory()
            expected = None
            origin = 0.0
            with self.lock:
                if not self.stopped.is_set():
                    self.state = 'esperando_audio'
            while not self.stopped.is_set():
                try:
                    item = self.queue.get(timeout=1)
                except queue.Empty:
                    if time.monotonic() - self.last_seen > self.idle_timeout:
                        self.stop()
                    continue
                if item is None:
                    break
                seq, offset, samples = item
                # Una pérdida de red/cola nunca se interpreta como silencio real.
                if expected is None or offset != expected:
                    detector.reset()
                    origin = offset / 16000
                expected = offset + len(samples)
                started = time.perf_counter()
                events = detector.process(samples)
                with self.lock:
                    if self.stopped.is_set():
                        break
                    self.processed += 1
                    self.processing_ms = (time.perf_counter() - started) * 1000
                    self.rms = float(np.sqrt(np.mean(samples ** 2)))
                    self.state = 'silencio' if self.rms < 1e-5 else 'analizando'
                    for result in events:
                        impact_offset = origin + result['impact_offset_s']
                        if impact_offset - self.last_event_offset < 30:
                            continue  # Conserva cooldown aunque un hueco reinicie el detector.
                        event_id = 'audio-' + uuid.uuid4().hex
                        self.store.append(dict(
                            alert_id=event_id, event_id=event_id, type='posible_caida',
                            risk_type='posible_caida_acustica', source='audio',
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            stream_id=self.id, severity='MEDIO', description=MESSAGE,
                            impact_offset_s=round(impact_offset, 3),
                            risk_ref='Detección acústica; requiere verificación humana',
                            technical=result, imagen=None))
                        self.last_event_offset = impact_offset
        except Exception as exc:
            LOGGER.exception('Fallo en sesión acústica %s', self.id)
            with self.lock:
                if not self.stopped.is_set():
                    self.state = 'error'
                    self.error = f'{type(exc).__name__}: {exc}'
        finally:
            if detector is not None:
                detector.reset()


class AudioSessionManager:
    def __init__(self, store, detector_factory=None, max_sessions=4, idle_timeout=15):
        self.store = store
        self.detector_factory = detector_factory or self._make_detector
        self.sessions = {}
        self.lock = threading.Lock()
        self.max_sessions = max_sessions
        self.idle_timeout = idle_timeout
        self.classifier = None
        self.classifier_lock = threading.Lock()
        self.closed = threading.Event()
        self.reaper = None

    def _make_detector(self):
        # Importaciones y carga costosas fuera de las peticiones/video.
        from audio_fall_detector.streaming import StreamingFallDetector, runtime_config
        from audio_fall_detector.classifier import FallClassifierHead
        from threadpoolctl import threadpool_limits
        with self.classifier_lock:
            if self.classifier is None:
                self.classifier = FallClassifierHead(runtime_config())
        detector = StreamingFallDetector(self.classifier)
        original = detector.process

        def process(samples):
            # Una inferencia espectral a la vez y BLAS acotado para proteger video.
            with self.classifier_lock, threadpool_limits(limits=1, user_api='blas'):
                return original(samples)
        detector.process = process
        return detector

    def warmup(self):
        """Carga y compila kernels acústicos antes de aceptar video en main()."""
        from threadpoolctl import threadpool_limits
        detector = self._make_detector()
        with self.classifier_lock, threadpool_limits(limits=1, user_api='blas'):
            # Mel/MFCC y pYIN usan kernels JIT en su primera ejecución.
            self.classifier.predict_proba(np.zeros(24000, dtype=np.float32))
            t = np.arange(8000, dtype=np.float32) / 16000
            detector.confirmator.detect_vocal_distress((.1*np.sin(2*np.pi*450*t)).astype(np.float32))
        detector.reset()

    def _reap(self):
        while not self.closed.wait(2):
            with self.lock:
                for key, value in list(self.sessions.items()):
                    if value.stopped.is_set() or time.monotonic() - value.last_seen > self.idle_timeout:
                        value.stop()
                        del self.sessions[key]

    def create(self):
        with self.lock:
            if len(self.sessions) >= self.max_sessions:
                raise RuntimeError('Límite de sesiones de audio alcanzado')
            if self.reaper is None:
                self.reaper = threading.Thread(target=self._reap, daemon=True, name='audio-cleanup')
                self.reaper.start()
            session = AudioSession(self.detector_factory, self.store, self.idle_timeout)
            self.sessions[session.id] = session
            return session

    def get(self, key):
        with self.lock:
            return self.sessions[key]

    def stop(self, key):
        with self.lock:
            session = self.sessions.pop(key, None)
        if session:
            session.stop()

    def close(self):
        self.closed.set()
        with self.lock:
            for session in self.sessions.values():
                session.stop()
            self.sessions.clear()


def audio_blueprint(manager):
    bp = Blueprint('audio_live', __name__)

    @bp.post('/api/audio/sessions')
    def create():
        try:
            return jsonify(manager.create().snapshot()), 201
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 429

    @bp.get('/api/audio/sessions/<key>')
    def status(key):
        try:
            return jsonify(manager.get(key).snapshot())
        except KeyError:
            return jsonify(error='Sesión finalizada'), 404

    @bp.post('/api/audio/sessions/<key>/chunks')
    def chunk(key):
        if request.content_length != 8000 * 4:
            return jsonify(error='Se requieren 8000 muestras float32 LE mono a 16000 Hz'), 400
        try:
            seq = int(request.headers.get('X-Audio-Sequence', ''))
            offset = int(request.headers.get('X-Audio-Offset', ''))
            if seq < 0 or offset != seq * 8000:
                raise ValueError('Secuencia u offset inválidos')
            samples = np.frombuffer(request.get_data(), dtype='<f4').copy()
            if not np.all(np.isfinite(samples)) or np.max(np.abs(samples)) > 1.01:
                raise ValueError('PCM inválido')
            session = manager.get(key)
            if session.snapshot()['state'] == 'error':
                return jsonify(session.snapshot()), 503
            accepted = session.submit(seq, offset, samples)
            return jsonify(session.snapshot()), 202 if accepted else 429
        except KeyError:
            return jsonify(error='Sesión finalizada'), 404
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @bp.post('/api/audio/sessions/<key>/stop')
    def stop(key):
        manager.stop(key)
        return jsonify(state='detenido')

    return bp


def install_audio(app, root):
    store = AudioEventStore(Path(root) / 'runs' / 'audio_alerts' / 'alertas.jsonl')
    manager = AudioSessionManager(store)
    app.register_blueprint(audio_blueprint(manager))
    atexit.register(manager.close)
    return store, manager
