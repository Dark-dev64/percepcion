"""Pruebas temporales del algoritmo reutilizado. Ver prueba real de modelo separada."""
from pathlib import Path
import time

import numpy as np
import pytest

from audio_fall_detector.streaming import StreamingFallDetector


class ControlledClassifier:
    model = object()
    def predict_proba(self, segment):
        assert len(segment) == 24000
        return {'caida':.95, 'portazo':.01, 'objeto_cae':.01, 'pasos':.01, 'voz_tv':.01, 'ruido_fondo':.01}


def feed(detector, audio):
    events = []
    for start in range(0, len(audio), 8000):
        events.extend(detector.process(audio[start:start+8000]))
    return events


def impact_audio(seconds=12):
    rng = np.random.default_rng(42)
    audio = rng.normal(0, .002, int(seconds*16000)).astype(np.float32)
    t = np.arange(2400) / 16000
    audio[32000:34400] += (.8*np.sin(2*np.pi*100*t)*np.exp(-15*t)).astype(np.float32)
    return audio


def test_waits_for_actual_post_audio_no_duplicate_and_reset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    detector = StreamingFallDetector(ControlledClassifier())
    audio = impact_audio(16)
    assert not feed(detector, audio[:8*16000])
    events = feed(detector, audio[8*16000:])
    assert len(events) == 1 and 2 <= events[0]['impact_offset_s'] <= 2.2
    assert events[0]['scores']['score_final'] >= .48
    assert all('CONFIRMACION_APROBADA' not in rule for rule in events[0]['rules'])
    assert not list(tmp_path.rglob('*.wav'))
    detector.reset()
    assert detector.buffer.total_samples_written == 0
    assert len(feed(detector, audio)) == 1


@pytest.mark.parametrize('kind', ['silence','ambient','tone'])
def test_normal_audio_does_not_alert(kind):
    detector = StreamingFallDetector(ControlledClassifier())
    rng = np.random.default_rng(123)
    audio = np.zeros(16000*12, np.float32)
    if kind == 'ambient': audio += rng.normal(0,.003,len(audio)).astype(np.float32)
    if kind == 'tone': audio += (.01*np.sin(2*np.pi*440*np.arange(len(audio))/16000)).astype(np.float32)
    assert not feed(detector, audio)


def test_missing_model_disables_detector():
    classifier = ControlledClassifier()
    classifier.model = None
    with pytest.raises(RuntimeError, match='Falta el modelo'):
        StreamingFallDetector(classifier)
