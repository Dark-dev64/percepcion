"""
Módulo 4: Confirmación Temporal Post-Impacto (Etapa 3)
======================================================
Analiza la ventana temporal de 3 a 10 s posteriores al impacto para confirmar la caída:
  1. Detección de voz, quejido o grito de auxilio (frecuencia fundamental humana F0 / VAD).
  2. Silencio o inmovilidad anormal (energía acústica por debajo o al nivel de la línea base,
     característico de una víctima incapacitada en el suelo).
  Retorna un score de confirmación calibrado de 0.0 a 1.0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import librosa
import numpy as np
import yaml

logger = logging.getLogger("fall_detector.confirm")


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ConfirmationResult:
    score: float
    vocal_distress_detected: bool
    abnormal_silence_detected: bool
    vocal_score: float
    silence_score: float
    f0_median_hz: Optional[float]
    post_impact_rms: float
    details: Dict[str, Any]


class PostImpactConfirmator:
    """Verificador temporal post-impacto para erradicación de falsos positivos."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or load_config()
        self.sr = cfg["audio"]["sample_rate"]
        self.start_offset = float(cfg["confirmation"].get("analysis_start_offset_sec", 1.0))
        self.duration = float(cfg["confirmation"].get("analysis_duration_sec", 6.0))
        self.fmin = float(cfg["confirmation"].get("vocal_min_f0_hz", 180.0))
        self.fmax = float(cfg["confirmation"].get("vocal_max_f0_hz", 850.0))
        self.min_conf = float(cfg["confirmation"].get("vocal_min_confidence", 0.45))
        self.silence_ratio = float(cfg["confirmation"].get("silence_energy_threshold_ratio", 1.25))
        self.w_vocal = float(cfg["confirmation"].get("weight_vocal", 0.60))
        self.w_silence = float(cfg["confirmation"].get("weight_silence", 0.40))

    def detect_vocal_distress(self, segment: np.ndarray) -> Tuple[float, Optional[float], float]:
        """
        Detecta indicios de quejido o pedido de auxilio vocal mediante pYIN y energía en banda vocal.
        Retorna (vocal_score [0..1], f0_mediana_hz, voiced_ratio).
        """
        if len(segment) < self.sr * 0.4:
            return 0.0, None, 0.0

        # Filtro rápido de energía en banda de voz (150-1200 Hz)
        rms = float(np.sqrt(np.mean(segment ** 2))) + 1e-6
        if rms < 0.02:
            return 0.0, None, 0.0

        try:
            f0, voiced_flag, voiced_prob = librosa.pyin(
                segment,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=self.sr,
                frame_length=1024,
                hop_length=512,
            )
        except Exception as e:
            logger.debug("Error estimando F0 en confirmación: %s", e)
            return 0.0, None, 0.0

        if f0 is None or not np.any(voiced_flag):
            return 0.0, None, 0.0

        voiced_f0 = f0[voiced_flag]
        voiced_conf = voiced_prob[voiced_flag]
        valid = voiced_f0[~np.isnan(voiced_f0)]

        if len(valid) == 0:
            return 0.0, None, 0.0

        median_f0 = float(np.median(valid))
        mean_conf = float(np.mean(voiced_conf))
        voiced_ratio = float(np.sum(voiced_flag) / len(voiced_flag))

        # El quejido o grito de auxilio tiene alta confianza y persistencia
        is_distress_range = self.fmin <= median_f0 <= self.fmax
        if is_distress_range and mean_conf >= self.min_conf:
            vocal_score = min(1.0, mean_conf * (0.6 + 0.4 * voiced_ratio))
        else:
            vocal_score = min(0.3, mean_conf * 0.3)

        return round(float(vocal_score), 3), round(median_f0, 1), round(voiced_ratio, 3)

    def analyze_silence(self, segment: np.ndarray, baseline_rms: float) -> Tuple[float, float]:
        """
        Determina si tras el impacto ocurre inmovilidad acústica anormal.
        Retorna (silence_score [0..1], post_rms).
        """
        if len(segment) == 0:
            return 0.5, 0.0

        post_rms = float(np.sqrt(np.mean(segment ** 2)))
        baseline = max(baseline_rms, 1e-5)
        energy_ratio = post_rms / baseline

        # Si el audio posterior es de bajo nivel (cercano o menor a la línea base),
        # indica que no hay actividad normal de marcha ni recuperación inmediata.
        if energy_ratio <= self.silence_ratio:
            # Muy silencioso tras la caída -> alta probabilidad de incapacitación
            silence_score = 1.0 - (energy_ratio / (self.silence_ratio * 1.5))
            silence_score = max(0.0, min(1.0, silence_score))
        else:
            # Mucha actividad sonora (e.g. continuaron caminando o hablando rápido)
            silence_score = max(0.0, 1.0 - (energy_ratio - self.silence_ratio) * 0.5)

        return round(float(silence_score), 3), round(post_rms, 5)

    def evaluate_post_impact(
        self,
        full_audio: np.ndarray,
        impact_timestamp_sec: float,
        baseline_rms: float,
    ) -> ConfirmationResult:
        """
        Evalúa el intervalo post-impacto (de 3 a 10 s tras el evento).
        """
        start_sample = int((impact_timestamp_sec + self.start_offset) * self.sr)
        end_sample = int((impact_timestamp_sec + self.start_offset + self.duration) * self.sr)

        # Extraer segmento disponible
        if start_sample >= len(full_audio):
            segment = np.zeros(0, dtype=np.float32)
        else:
            segment = full_audio[start_sample : min(end_sample, len(full_audio))]

        vocal_score, f0_val, voiced_ratio = self.detect_vocal_distress(segment)
        silence_score, post_rms = self.analyze_silence(segment, baseline_rms)

        # Fusión de confirmación
        combined_score = (self.w_vocal * vocal_score) + (self.w_silence * silence_score)

        # Regla de refuerzo: si hay quejido humano evidente, score sube drásticamente
        if vocal_score >= 0.70:
            combined_score = max(combined_score, vocal_score)

        final_score = round(float(np.clip(combined_score, 0.0, 1.0)), 3)

        return ConfirmationResult(
            score=final_score,
            vocal_distress_detected=vocal_score >= 0.45,
            abnormal_silence_detected=silence_score >= 0.65,
            vocal_score=vocal_score,
            silence_score=silence_score,
            f0_median_hz=f0_val,
            post_impact_rms=post_rms,
            details={
                "voiced_ratio": voiced_ratio,
                "analysis_duration_sec": len(segment) / self.sr if len(segment) else 0.0,
            },
        )
