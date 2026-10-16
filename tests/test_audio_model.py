"""Modelo REAL migrado, entradas sintéticas; no representa caídas humanas reales."""
from pathlib import Path
import hashlib

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from audio_fall_detector.dataset import generate_synthesized_clip
from audio_fall_detector.streaming import StreamingFallDetector


@pytest.fixture(scope='module')
def detector():
    return StreamingFallDetector()


@pytest.mark.parametrize('category', ['caida','portazo','objeto_cae','pasos','voz_tv','ruido_fondo'])
def test_existing_model_streaming_classification(detector, category):
    detector.reset()
    rng = np.random.default_rng(42)
    audio = rng.normal(0,.002,16000*13).astype(np.float32)
    clip = generate_synthesized_clip(category,seed=42)
    audio[32000:32000+len(clip)] += clip
    results = []
    with threadpool_limits(1):
        for start in range(0,len(audio),8000):
            new = detector.process(audio[start:start+8000])
            if new:
                assert (start+8000)/16000 >= 9.05  # no evalúa audio futuro
            results.extend(new)
    if category == 'caida':
        assert len(results) == 1
        assert results[0]['predicted_class'] == 'caida'
        assert results[0]['scores']['score_final'] >= .48
    else:
        assert results == []


def test_original_artifact_bytes_unchanged():
    path = Path(__file__).resolve().parents[1] / 'audio_fall_detector/models/fall_classifier.joblib'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '68527b88f75142693c56ee798f981611febb5d25fd46316e6c799aca7e2eb9da'


def test_numpy_compatibility_preserves_native_weights_and_predictions(detector):
    # Referencia obtenida cargando este mismo joblib con NumPy 2.5.4.
    digest = hashlib.sha256()
    for head in detector.classifier.model.calibrated_classifiers_:
        for array in head.estimator.coefs_ + head.estimator.intercepts_:
            digest.update(array.tobytes())
    assert digest.hexdigest() == '7d7620cb8a007ddf93863224c27cbec4ad25892fe1898fe7ce9e1d53e8031678'
    matrix = np.array([np.arange(2048,dtype=np.float32)/2048])
    with threadpool_limits(1):
        actual = detector.classifier.model.predict_proba(matrix)[0]
    expected = [1.3278604611827312e-05,.0006363047550485587,.00042505577802180077,
                .00964882395519496,.1524455030220725,.8368310338850504]
    np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5)


def test_startup_warmup_does_not_create_sessions_or_recordings(tmp_path):
    from audio_live import AudioEventStore, AudioSessionManager
    manager = AudioSessionManager(AudioEventStore(tmp_path / 'events.jsonl'))
    try:
        manager.warmup()
        assert manager.classifier.model is not None
        assert manager.sessions == {} and manager.reaper is None
        assert not list(tmp_path.iterdir())
    finally:
        manager.close()
