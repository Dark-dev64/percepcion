"""
Generador y Gestor de Dataset para Detección de Caídas Acústicas
================================================================
Permite generar datasets sintéticos realistas con 6 clases y metadatos de sesión
para validación estricta y entrenamiento sin fuga de datos (data leakage).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal as signal
import soundfile as sf


def generate_synthesized_clip(
    category: str,
    duration_sec: float = 1.5,
    sr: int = 16000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Genera un clip acústico sintético representativo de una de las 6 clases."""
    rng = np.random.default_rng(seed)
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    y = np.zeros(n_samples, dtype=np.float32)

    # Ruido ambiente base en todos los clips
    ambient = rng.normal(0, 0.015, n_samples).astype(np.float32)

    if category == "caida":
        # Impacto sordo pesado: pulso de baja frecuencia (50-120 Hz) + decaimiento exponencial
        t_impact = duration_sec * 0.35
        idx_impact = int(t_impact * sr)
        
        # Pulso transitorio
        decay = np.exp(-14.0 * np.maximum(0, t - t_impact))
        freq = rng.uniform(55, 95)
        impact = np.sin(2 * np.pi * freq * (t - t_impact)) * decay * rng.uniform(0.75, 0.98)
        
        # Resonancia corporal y de suelo
        thud = rng.normal(0, 0.25, n_samples) * decay
        b, a = signal.butter(4, [40, 220], btype="bandpass", fs=sr)
        thud_filt = signal.lfilter(b, a, thud)
        
        y = ambient + impact + thud_filt

    elif category == "portazo":
        # Golpe agudo de alta energía con reverberación acústica
        t_impact = duration_sec * 0.3
        decay = np.exp(-22.0 * np.maximum(0, t - t_impact))
        slam = rng.normal(0, 0.85, n_samples) * decay
        b, a = signal.butter(3, [150, 4000], btype="bandpass", fs=sr)
        y = ambient + signal.lfilter(b, a, slam) * 0.95

    elif category == "objeto_cae":
        # Múltiples rebotes ligeros (e.g. celular, llaves, vaso de plástico)
        y = ambient.copy()
        n_rebotes = rng.integers(3, 6)
        t_base = duration_sec * 0.25
        dt = 0.08
        for b_idx in range(n_rebotes):
            t_r = t_base + b_idx * dt
            amp_r = (0.55 / (b_idx + 1)) * rng.uniform(0.8, 1.2)
            decay_r = np.exp(-65.0 * np.maximum(0, t - t_r))
            ping = np.sin(2 * np.pi * rng.uniform(1200, 2800) * (t - t_r)) * decay_r * amp_r
            y += ping.astype(np.float32)

    elif category == "pasos":
        # Cadencia de pisadas rítmicas
        cadence_steps = int(duration_sec * rng.uniform(1.8, 3.2))
        y = ambient.copy()
        for s_idx in range(max(1, cadence_steps)):
            t_s = (s_idx + 1) * (duration_sec / (cadence_steps + 1))
            decay_s = np.exp(-35.0 * np.maximum(0, t - t_s))
            step = rng.normal(0, 0.18, n_samples) * decay_s
            b, a = signal.butter(2, [80, 500], btype="bandpass", fs=sr)
            y += signal.lfilter(b, a, step).astype(np.float32)

    elif category == "voz_tv":
        # Formantes de habla humana
        f0 = rng.uniform(140, 280)
        speech = np.sin(2 * np.pi * f0 * t) * 0.25
        speech += np.sin(2 * np.pi * f0 * 2 * t) * 0.18
        speech += np.sin(2 * np.pi * f0 * 3 * t) * 0.12
        env = 0.5 * (1 + np.sin(2 * np.pi * rng.uniform(2, 5) * t))
        y = ambient + (speech * env).astype(np.float32)

    else:  # ruido_fondo
        # Ruido continuo filtrado (HVAC o zumbido)
        b, a = signal.butter(2, 600, btype="lowpass", fs=sr)
        y = ambient + signal.lfilter(b, a, rng.normal(0, 0.05, n_samples)).astype(np.float32)

    # Normalizar a rango [-0.98, 0.98]
    peak = np.max(np.abs(y)) + 1e-9
    if peak > 0:
        y = y / peak * 0.95

    return y.astype(np.float32)


def create_synthetic_dataset(
    output_dir: Path,
    n_samples_per_class: int = 30,
    sr: int = 16000,
    duration_sec: float = 1.5,
) -> List[Dict[str, Any]]:
    """Crea y guarda en disco un dataset balanceado con metadatos de sesión para entrenamiento."""
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = ["caida", "portazo", "objeto_cae", "pasos", "voz_tv", "ruido_fondo"]
    manifest = []

    for cl in classes:
        cl_dir = output_dir / cl
        cl_dir.mkdir(exist_ok=True)
        for i in range(n_samples_per_class):
            session_id = f"session_{i % 5}"  # 5 sesiones distintas para GroupKFold
            clip = generate_synthesized_clip(cl, duration_sec=duration_sec, sr=sr, seed=1000 + i)
            file_name = f"{cl}_{i:03d}.wav"
            file_path = cl_dir / file_name
            sf.write(str(file_path), clip, sr)
            manifest.append({
                "file_path": str(file_path),
                "label": cl,
                "session_id": session_id,
            })

    return manifest
