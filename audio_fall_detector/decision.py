"""
Módulo 5: Motor de Decisión, Fusión Ponderada y Máquina de Estados
===================================================================
Pipeline:
  1. Fusión ponderada: score_final = w1 * p_caida + w2 * confirmacion
  2. Máquina de estados: IDLE -> POSIBLE_CAIDA -> CAIDA_CONFIRMADA
  3. Histéresis y cooldown de 30 s entre alertas confirmadas
  4. Reglas duras anti-falsos positivos (rechazo si portazo/objeto_cae > umbral)
  5. Guardado de clip de audio de 10 s (3 s pre + 7 s post impacto) para auditoría y reentrenamiento
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import yaml

from audio_fall_detector.confirm import ConfirmationResult
from audio_fall_detector.trigger import TriggerEvent

logger = logging.getLogger("fall_detector.decision")


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class FallState(str, Enum):
    IDLE = "IDLE"
    POSIBLE_CAIDA = "POSIBLE_CAIDA"
    CAIDA_CONFIRMADA = "CAIDA_CONFIRMADA"


@dataclass
class DecisionAlert:
    timestamp: str
    event_id: str
    state: str
    nivel: str
    scores: Dict[str, float]
    clase_predicha: str
    clip_path: Optional[str]
    reglas_activadas: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DecisionEngine:
    """Motor de decisión probabilística con máquina de estados finitos y filtrado de falsas alarmas."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or load_config()
        self.config = cfg
        self.sr = cfg["audio"]["sample_rate"]
        self.w1 = float(cfg["decision"].get("weight_classifier_caida", 0.65))
        self.w2 = float(cfg["decision"].get("weight_confirmation", 0.35))
        self.th_confirmed = float(cfg["decision"].get("fall_threshold_confirmed", 0.68))
        self.th_possible = float(cfg["decision"].get("fall_threshold_possible", 0.48))
        self.slam_rejection = float(cfg["decision"].get("slam_rejection_prob", 0.58))
        self.cooldown_sec = float(cfg["decision"].get("state_cooldown_sec", 30.0))
        self.save_clip_enabled = bool(cfg["decision"].get("save_event_clip", True))
        self.pre_impact_sec = float(cfg["decision"].get("clip_pre_impact_sec", 3.0))
        self.post_impact_sec = float(cfg["decision"].get("clip_post_impact_sec", 7.0))
        self.clips_dir = Path(cfg["decision"].get("clips_directory", "audio_fall_detector/recordings/events"))
        if self.save_clip_enabled:
            self.clips_dir.mkdir(parents=True, exist_ok=True)

        self.current_state = FallState.IDLE
        self.last_confirmed_time = -999.0
        self.event_counter = 0

    def evaluate(
        self,
        trigger_evt: TriggerEvent,
        class_probs: Dict[str, float],
        confirm_res: ConfirmationResult,
        full_audio: Optional[np.ndarray] = None,
        current_time_sec: Optional[float] = None,
    ) -> DecisionAlert:
        """
        Ejecuta la máquina de estados y las reglas de decisión para emitir la alerta final en JSON.
        """
        t_now = current_time_sec if current_time_sec is not None else trigger_evt.timestamp_sec
        self.event_counter += 1
        iso_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        event_id = f"evt-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')}-{self.event_counter:03d}"

        p_caida = class_probs.get("caida", 0.0)
        p_portazo = class_probs.get("portazo", 0.0)
        p_objeto_cae = class_probs.get("objeto_cae", 0.0)
        winner_class = max(class_probs, key=class_probs.get)

        # 1. Fusión ponderada
        score_final = (self.w1 * p_caida) + (self.w2 * confirm_res.score)
        score_final = round(float(np.clip(score_final, 0.0, 1.0)), 3)

        reglas: List[str] = []

        # 2. Comprobación de Cooldown
        in_cooldown = (t_now - self.last_confirmed_time) < self.cooldown_sec
        if in_cooldown:
            reglas.append(f"COOLDOWN_ACTIVO: Restan {round(self.cooldown_sec - (t_now - self.last_confirmed_time), 1)}s")

        # 3. Reglas explícitas de rechazo anti-falsos positivos
        slam_detected = max(p_portazo, p_objeto_cae) >= self.slam_rejection
        if slam_detected:
            reglas.append(f"RECHAZO_PORTAZO_OBJETO: prob={max(p_portazo, p_objeto_cae):.2f} >= {self.slam_rejection}")

        low_impact_class = p_caida < 0.20
        if low_impact_class:
            reglas.append("RECHAZO_CLASIFICADOR: p_caida insuficiente")

        # 4. Transición en la máquina de estados
        nuevo_estado = FallState.IDLE
        nivel = "BAJO"

        if not in_cooldown and not slam_detected and not low_impact_class:
            if score_final >= self.th_confirmed:
                nuevo_estado = FallState.CAIDA_CONFIRMADA
                nivel = "ALTO"
                self.last_confirmed_time = t_now
                reglas.append(f"CONFIRMACION_APROBADA: score_final={score_final:.3f} >= {self.th_confirmed}")
            elif score_final >= self.th_possible:
                nuevo_estado = FallState.POSIBLE_CAIDA
                nivel = "MEDIO"
                reglas.append(f"POSIBLE_CAIDA_DETECTADA: score_final={score_final:.3f} >= {self.th_possible}")
            else:
                nuevo_estado = FallState.IDLE
                nivel = "BAJO"
                reglas.append("SCORE_FINAL_INSUFICIENTE")
        else:
            nuevo_estado = FallState.IDLE
            nivel = "BAJO"

        self.current_state = nuevo_estado

        # 5. Guardado de clip de 10 s alrededor del impacto si es un evento relevante
        clip_path_str: Optional[str] = None
        if self.save_clip_enabled and full_audio is not None and nuevo_estado != FallState.IDLE:
            clip_path = self._save_clip(full_audio, trigger_evt.timestamp_sec, event_id)
            clip_path_str = str(clip_path)

        # 6. Construir objeto de respuesta JSON estructurado
        scores_dict = {
            "trigger_ratio": trigger_evt.ratio,
            "p_caida": round(p_caida, 4),
            "p_portazo": round(p_portazo, 4),
            "p_objeto_cae": round(p_objeto_cae, 4),
            "p_pasos": round(class_probs.get("pasos", 0.0), 4),
            "p_voz_tv": round(class_probs.get("voz_tv", 0.0), 4),
            "p_ruido_fondo": round(class_probs.get("ruido_fondo", 0.0), 4),
            "score_confirmacion": confirm_res.score,
            "score_final": score_final,
        }

        alert = DecisionAlert(
            timestamp=iso_timestamp,
            event_id=event_id,
            state=nuevo_estado.value,
            nivel=nivel,
            scores=scores_dict,
            clase_predicha=winner_class,
            clip_path=clip_path_str,
            reglas_activadas=reglas,
        )

        return alert

    def _save_clip(self, full_audio: np.ndarray, t_impact_sec: float, event_id: str) -> Path:
        """Guarda 10 segundos alrededor del impacto (3s antes y 7s después)."""
        start_samp = max(0, int((t_impact_sec - self.pre_impact_sec) * self.sr))
        end_samp = min(len(full_audio), int((t_impact_sec + self.post_impact_sec) * self.sr))
        clip = full_audio[start_samp:end_samp]

        target_len = int((self.pre_impact_sec + self.post_impact_sec) * self.sr)
        if len(clip) < target_len:
            pad = np.zeros(target_len - len(clip), dtype=np.float32)
            clip = np.concatenate((clip, pad))

        out_path = self.clips_dir / f"{event_id}.wav"
        sf.write(str(out_path), clip, self.sr)
        logger.info("Clip de audio del evento guardado para reentrenamiento: %s", out_path)
        return out_path

    def reset(self) -> None:
        """Reinicia la máquina de estados y contadores."""
        self.current_state = FallState.IDLE
        self.last_confirmed_time = -999.0
