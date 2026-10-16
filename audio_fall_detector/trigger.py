"""
Módulo 2: Disparador de Impacto Acústico (Etapa 1)
==================================================
Objetivo de bajo consumo y filtro anti-falsos positivos:
  Detecta impacto cuando el RMS de ventana corta (50 ms) supera k veces el RMS de línea base
  (mediana móvil de los últimos 5.0 s).
  Si NO hay impacto, las etapas pesadas posteriores (clasificador y confirmación) no se ejecutan.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class TriggerEvent:
    detected: bool
    timestamp_sec: float
    peak_rms: float
    baseline_rms: float
    ratio: float
    k_threshold: float
    details: Dict[str, Any]


class ImpactTrigger:
    """Detector de transitorios de impacto mediante comparación de energía relativa."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or load_config()
        self.sr = cfg["audio"]["sample_rate"]
        self.short_win_ms = cfg["trigger"]["short_window_ms"]
        self.baseline_sec = cfg["trigger"]["baseline_window_sec"]
        self.k = float(cfg["trigger"]["k_factor"])
        self.min_rms = float(cfg["trigger"]["min_absolute_rms"])
        self.cooldown_sec = float(cfg["trigger"].get("cooldown_sec", 1.5))

        self.short_samples = int((self.short_win_ms / 1000.0) * self.sr)
        self.max_history_items = int(self.baseline_sec / (self.short_win_ms / 1000.0))

        # Cola circular para almacenar los RMS de 50 ms de los últimos 5 s
        self.rms_history: deque[float] = deque(maxlen=self.max_history_items)
        self.last_trigger_time = -999.0

    @staticmethod
    def compute_rms(samples: np.ndarray) -> float:
        """Calcula Root Mean Square de un bloque de audio."""
        if len(samples) == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))

    def update_baseline(self, rms_val: float) -> None:
        """Agrega un nuevo valor de RMS a la memoria de línea base."""
        self.rms_history.append(rms_val)

    def get_baseline_rms(self) -> float:
        """Calcula la mediana móvil de los RMS de fondo (robusta a ruidos esporádicos)."""
        if not self.rms_history:
            return self.min_rms
        # Usamos mediana para que transitorios breves no inflen la línea base
        return float(np.median(self.rms_history))

    def evaluate_window(self, samples: np.ndarray, timestamp_sec: float) -> TriggerEvent:
        """
        Evalúa un segmento de audio (habitualmente 50 ms).
        Determina si ocurre un impacto transitorio que amerite activar Etapa 2.
        """
        short_rms = self.compute_rms(samples)
        baseline_rms = max(self.get_baseline_rms(), 1e-6)
        ratio = short_rms / baseline_rms

        # Verificar condición de impacto con histéresis y cooldown
        in_cooldown = (timestamp_sec - self.last_trigger_time) < self.cooldown_sec
        impact_detected = (
            (short_rms >= self.min_rms)
            and (ratio >= self.k)
            and (not in_cooldown)
        )

        if impact_detected:
            self.last_trigger_time = timestamp_sec

        # Actualizar línea base con el valor actual si no es un transitorio violento
        if short_rms < (self.k * 0.8 * baseline_rms):
            self.update_baseline(short_rms)
        else:
            # En impacto, repetimos la mediana para no distorsionar la línea base posterior
            self.update_baseline(baseline_rms)

        return TriggerEvent(
            detected=impact_detected,
            timestamp_sec=round(timestamp_sec, 3),
            peak_rms=round(short_rms, 5),
            baseline_rms=round(baseline_rms, 5),
            ratio=round(ratio, 2),
            k_threshold=self.k,
            details={
                "in_cooldown": in_cooldown,
                "history_length": len(self.rms_history),
                "min_rms_met": short_rms >= self.min_rms,
            },
        )

    def scan_audio(self, audio: np.ndarray) -> List[TriggerEvent]:
        """
        Escanea una grabación continua completa buscando todos los eventos de impacto.
        Útil para pruebas batch y evaluación de falsas alarmas por hora.
        """
        events = []
        hop = self.short_samples
        total_samples = len(audio)

        for i in range(0, total_samples - self.short_samples + 1, hop):
            chunk = audio[i : i + self.short_samples]
            t_sec = (i + self.short_samples) / self.sr
            evt = self.evaluate_window(chunk, t_sec)
            if evt.detected:
                events.append(evt)

        return events

    def reset(self) -> None:
        """Reinicia el estado del detector y la historia de línea base."""
        self.rms_history.clear()
        self.last_trigger_time = -999.0
