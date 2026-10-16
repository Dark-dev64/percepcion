"""Diagnóstico local reproducible del artefacto migrado; datos sintéticos en RAM."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from threadpoolctl import threadpool_limits
from audio_fall_detector.dataset import generate_synthesized_clip
from audio_fall_detector.streaming import StreamingFallDetector
from audio_fall_detector.model_compat import load_original_model

with threadpool_limits(1):
    detector = StreamingFallDetector()
    print('CLASSES', detector.classifier.model.classes_, flush=True)
    print('ARTIFACT_CONFIG', load_original_model(detector.classifier.save_path)['config']['classifier'], flush=True)
    for category in ['caida', 'portazo', 'objeto_cae', 'pasos', 'voz_tv', 'ruido_fondo']:
        detector.reset()
        clip = generate_synthesized_clip(category, seed=42)
        print('RAW_CLASS', category, detector.classifier.predict_proba(clip), flush=True)
        classify = detector.classifier.predict_proba
        def traced(segment):
            probs = classify(segment)
            print('WINDOW_CLASS', category, probs, flush=True)
            return probs
        detector.classifier.predict_proba = traced
        audio = np.random.default_rng(42).normal(0,.002,16000*13).astype(np.float32)
        audio[32000:32000+len(clip)] += clip
        results, times = [], []
        for start in range(0,len(audio),8000):
            t = time.perf_counter()
            results.extend(detector.process(audio[start:start+8000]))
            times.append((time.perf_counter()-t)*1000)
        print('STREAM', category, results, 'max_ms',max(times),'median_ms',np.median(times), flush=True)
        detector.classifier.predict_proba = classify
