"""
Módulo 1: Preprocesamiento y Buffer Circular para Streaming
============================================================
Pipeline:
  1. Conversión a Mono y 16 kHz.
  2. Filtro Pasa-Altos Butterworth a 80 Hz (orden 5).
  3. Reducción de Ruido Estacionario (noisereduce).
  4. Normalización a pico objetivo.
  5. Buffer Circular para ventana deslizante de 2.0 s y salto de 0.5 s.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple, Union

import librosa
import noisereduce as nr
import numpy as np
import scipy.signal as signal
import soundfile as sf
import yaml

logger = logging.getLogger("fall_detector.preprocess")


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Carga la configuración centralizada desde config.yaml."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AudioPreprocessor:
    """Aplica la cadena de filtrado, reducción de ruido y normalización acústica."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.sr = self.config["audio"]["sample_rate"]
        self.cutoff_hz = self.config["preprocess"]["highpass_cutoff_hz"]
        self.order = self.config["preprocess"]["highpass_order"]
        self.noise_reduction = self.config["preprocess"]["stationary_noise_reduction"]
        self.nr_prop = self.config["preprocess"]["noise_reduction_prop_decrease"]
        self.normalize = self.config["preprocess"]["normalize_peak"]
        self.target_peak = self.config["preprocess"]["target_peak_amplitude"]

        # Precalcular coeficientes SOS del filtro pasa-altos Butterworth
        self.sos = signal.butter(
            self.order, self.cutoff_hz, btype="highpass", fs=self.sr, output="sos"
        )

    def load_audio(
        self, source: Union[str, Path, bytes, np.ndarray], original_sr: Optional[int] = None
    ) -> np.ndarray:
        """Carga cualquier fuente de audio, garantizando canal mono y 16 kHz en float32."""
        if isinstance(source, np.ndarray):
            audio = source.astype(np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)
            if original_sr is not None and original_sr != self.sr:
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.sr)
            return audio

        if isinstance(source, bytes):
            buf = io.BytesIO(source)
            audio, sr_orig = sf.read(buf, dtype="float32")
        else:
            audio, sr_orig = sf.read(str(source), dtype="float32")

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        if sr_orig != self.sr:
            audio = librosa.resample(audio, orig_sr=sr_orig, target_sr=self.sr)

        return audio.astype(np.float32)

    def apply_highpass(self, audio: np.ndarray) -> np.ndarray:
        """Aplica filtro pasa-altos Butterworth a 80 Hz usando SOS para estabilidad numérica."""
        if len(audio) < self.order * 3:
            return audio
        filtered = signal.sosfiltfilt(self.sos, audio)
        return filtered.astype(np.float32)

    def apply_noise_reduction(
        self, audio: np.ndarray, noise_profile: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Aplica reducción de ruido espectral estacionario."""
        if not self.noise_reduction or len(audio) < 1024:
            return audio
        try:
            cleaned = nr.reduce_noise(
                y=audio,
                sr=self.sr,
                prop_decrease=self.nr_prop,
                stationary=True,
                y_noise=noise_profile,
            )
            return cleaned.astype(np.float32)
        except Exception as e:
            logger.warning("Fallo en reducción de ruido (%s), se preserva señal original.", e)
            return audio

    def apply_normalization(self, audio: np.ndarray) -> np.ndarray:
        """Normaliza el pico de amplitud para mantener rango dinámico consistente."""
        if not self.normalize or len(audio) == 0:
            return audio
        peak = float(np.max(np.abs(audio))) + 1e-9
        if peak > 1e-5:
            scale = self.target_peak / peak
            return (audio * scale).astype(np.float32)
        return audio

    def process(
        self,
        audio: np.ndarray,
        noise_profile: Optional[np.ndarray] = None,
        apply_nr: bool = True,
    ) -> np.ndarray:
        """Ejecuta el pipeline completo de preprocesamiento."""
        x = self.apply_highpass(audio)
        if apply_nr:
            x = self.apply_noise_reduction(x, noise_profile=noise_profile)
        x = self.apply_normalization(x)
        return x


class CircularAudioBuffer:
    """
    Buffer circular para streaming continuo de audio en tiempo real.
    Permite procesar con ventana deslizante de 2.0 s y salto de 0.5 s sin pérdida de transitorios.
    """

    def __init__(
        self,
        capacity_sec: float = 20.0,
        sample_rate: int = 16000,
        window_sec: float = 2.0,
        hop_sec: float = 0.5,
    ):
        self.sr = sample_rate
        self.capacity_samples = int(capacity_sec * sample_rate)
        self.window_samples = int(window_sec * sample_rate)
        self.hop_samples = int(hop_sec * sample_rate)

        self.buffer = np.zeros(self.capacity_samples, dtype=np.float32)
        self.write_idx = 0
        self.total_samples_written = 0
        self.last_window_sample = 0

    def append(self, samples: np.ndarray) -> None:
        """Agrega un bloque de muestras de audio al buffer circular."""
        samples = samples.astype(np.float32)
        n = len(samples)
        if n == 0:
            return

        if n >= self.capacity_samples:
            # Si el bloque supera la capacidad, conservar solo los últimos n muestras
            samples = samples[-self.capacity_samples :]
            n = len(samples)
            self.buffer[:] = samples
            self.write_idx = 0
            self.total_samples_written += n
            return

        end_idx = self.write_idx + n
        if end_idx <= self.capacity_samples:
            self.buffer[self.write_idx : end_idx] = samples
        else:
            first_part = self.capacity_samples - self.write_idx
            self.buffer[self.write_idx :] = samples[:first_part]
            self.buffer[: end_idx - self.capacity_samples] = samples[first_part:]

        self.write_idx = (self.write_idx + n) % self.capacity_samples
        self.total_samples_written += n

    def get_recent(self, duration_sec: float) -> np.ndarray:
        """Obtiene los últimos duration_sec segundos del buffer en orden cronológico."""
        n = int(duration_sec * self.sr)
        n = min(n, self.capacity_samples, self.total_samples_written)
        if n <= 0:
            return np.zeros(0, dtype=np.float32)

        start = (self.write_idx - n) % self.capacity_samples
        if start < self.write_idx:
            return self.buffer[start : self.write_idx].copy()
        else:
            return np.concatenate((self.buffer[start:], self.buffer[: self.write_idx]))

    def get_window_at_time(self, t_center_sec: float, window_sec: float) -> np.ndarray:
        """Obtiene una ventana centrada en un timestamp absoluto (en segundos)."""
        w_samples = int(window_sec * self.sr)
        center_sample = int(t_center_sec * self.sr)
        start_sample = center_sample - (w_samples // 2)

        available_start = max(0, self.total_samples_written - self.capacity_samples)
        if start_sample < available_start or (start_sample + w_samples) > self.total_samples_written:
            # Ventana parcialmente fuera del buffer
            recent = self.get_recent(window_sec)
            if len(recent) < w_samples:
                pad = np.zeros(w_samples - len(recent), dtype=np.float32)
                return np.concatenate((pad, recent))
            return recent

        # Calcular offset desde el final
        offset_from_end = self.total_samples_written - (start_sample + w_samples)
        recent_chunk = self.get_recent((offset_from_end + w_samples) / self.sr)
        return recent_chunk[:w_samples]

    def ready_windows(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Generador que entrega ventanas completas de 2.0 s cada vez que se avanza hop_sec (0.5 s).
        Devuelve (segmento_audio, timestamp_fin_sec).
        """
        while (self.total_samples_written - self.last_window_sample) >= self.window_samples:
            target_end_sample = self.last_window_sample + self.window_samples
            offset_from_end = self.total_samples_written - target_end_sample

            chunk = self.get_recent((offset_from_end + self.window_samples) / self.sr)
            window = chunk[: self.window_samples]
            t_end_sec = target_end_sample / self.sr

            yield window, t_end_sec
            self.last_window_sample += self.hop_samples

    def reset(self) -> None:
        """Reinicia el buffer circular."""
        self.buffer.fill(0)
        self.write_idx = 0
        self.total_samples_written = 0
        self.last_window_sample = 0
