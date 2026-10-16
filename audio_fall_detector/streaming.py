"""Adaptación temporal del detector original para PCM continuo a 16 kHz.

El disparador trabaja cada 50 ms. Un candidato espera los siete segundos
posteriores reales antes de evaluar las etapas originales de clasificación,
confirmación y decisión. Nunca se usa audio futuro rellenado con ceros.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import sosfilt

from .classifier import FallClassifierHead
from .confirm import PostImpactConfirmator
from .decision import DecisionEngine
from .preprocess import AudioPreprocessor, CircularAudioBuffer, load_config
from .trigger import ImpactTrigger


def runtime_config():
    config = load_config()
    config['classifier']['model_save_path'] = str(
        Path(__file__).parent / 'models' / 'fall_classifier.joblib')
    # El modelo incluido utiliza características espectrales, no CNN14.
    config['classifier']['embedding_backend'] = 'spectral'
    # train.py extrae features de clips crudos de 1.5 s desde su inicio.
    # El trigger marca el FINAL del bloque de 50 ms: retroceder ese bloque
    # conserva la alineación de esos clips, sin añadir 300 ms de silencio.
    config['classifier']['pre_impact_offset_sec'] = config['trigger']['short_window_ms'] / 1000
    config['decision']['save_event_clip'] = False
    return config


class StreamingFallDetector:
    def __init__(self, classifier=None):
        self.config = runtime_config()
        self.sr = self.config['audio']['sample_rate']
        self.preprocessor = AudioPreprocessor(self.config)
        self.trigger = ImpactTrigger(self.config)
        self.classifier = classifier or FallClassifierHead(self.config)
        if self.classifier.model is None:
            raise RuntimeError('Falta el modelo acústico; no se habilita una heurística de respaldo.')
        self.confirmator = PostImpactConfirmator(self.config)
        self.decision = DecisionEngine(self.config)
        self.buffer = CircularAudioBuffer(capacity_sec=20, sample_rate=self.sr)
        self.raw_buffer = CircularAudioBuffer(capacity_sec=20, sample_rate=self.sr)
        self.reset()

    def reset(self):
        self.trigger.reset()
        self.decision.reset()
        self.buffer.reset()
        self.raw_buffer.reset()
        self.pending = []
        self.remainder = np.empty(0, dtype=np.float32)
        self.raw_remainder = np.empty(0, dtype=np.float32)
        self.filter_state = np.zeros((len(self.preprocessor.sos), 2))
        self.last_alert_time = -float('inf')

    def process(self, samples):
        filtered, self.filter_state = sosfilt(
            self.preprocessor.sos, samples, zi=self.filter_state)
        self.remainder = np.concatenate((self.remainder, filtered)).astype(np.float32)
        self.raw_remainder = np.concatenate((self.raw_remainder, samples)).astype(np.float32)
        events = []
        size = self.trigger.short_samples
        while len(self.remainder) >= size:
            block, self.remainder = self.remainder[:size], self.remainder[size:]
            raw_block, self.raw_remainder = self.raw_remainder[:size], self.raw_remainder[size:]
            self.buffer.append(block)
            self.raw_buffer.append(raw_block)
            now = self.buffer.total_samples_written / self.sr
            trig = self.trigger.evaluate_window(block, now)
            if trig.detected:
                self.pending.append(trig)
            wait = self.confirmator.start_offset + self.confirmator.duration
            while self.pending and now >= self.pending[0].timestamp_sec + wait:
                candidate = self.pending.pop(0)
                if candidate.timestamp_sec - self.last_alert_time < self.decision.cooldown_sec:
                    continue
                audio = self.buffer.get_recent(20)
                origin = now - len(audio) / self.sr
                relative = candidate.timestamp_sec - origin
                start = max(0, int((relative - self.config['classifier']['pre_impact_offset_sec']) * self.sr))
                length = int(self.config['classifier']['window_around_impact_sec'] * self.sr)
                raw_audio = self.raw_buffer.get_recent(20)
                segment = raw_audio[start:start + length]
                if len(segment) < length:
                    continue
                # train.py original usa clip crudo; filtrar/normalizar aquí
                # cambia la distribución de features y pierde sensibilidad.
                probs = self.classifier.predict_proba(segment)
                if (probs.get('caida', 0) < .20 or
                        max(probs.get('portazo', 0), probs.get('objeto_cae', 0)) >= self.decision.slam_rejection):
                    continue  # veto original; evita pYIN para sonidos rechazados
                confirmation = self.confirmator.evaluate_post_impact(
                    audio, relative, candidate.baseline_rms)
                result = self.decision.evaluate(
                    replace(candidate, timestamp_sec=relative), probs, confirmation,
                    current_time_sec=candidate.timestamp_sec).to_dict()
                if result['state'] != 'IDLE':
                    self.last_alert_time = candidate.timestamp_sec
                    # La fusión es un índice interno, no confianza de caída real.
                    events.append({'impact_offset_s': candidate.timestamp_sec,
                                   'scores': result['scores'],
                                   'rules': [rule.replace('CONFIRMACION_APROBADA', 'UMBRAL_ACUSTICO_ALTO_SUPERADO')
                                             for rule in result['reglas_activadas']],
                                   'predicted_class': result['clase_predicha']})
        return events
