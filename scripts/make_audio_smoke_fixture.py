"""Señal sintética en formato PCM para la prueba del navegador. No es una grabación."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from audio_fall_detector.dataset import generate_synthesized_clip

audio = np.random.default_rng(42).normal(0,.002,16000*14).astype('<f4')
clip = generate_synthesized_clip('caida',seed=42)
audio[32000:32000+len(clip)] += clip
# Señal vocal TONAL sintética posterior: verifica también pYIN/fusión,
# no representa semántica de auxilio ni una grabación humana.
t = np.arange(4*16000)/16000
audio[4*16000:8*16000] += (.12*np.sin(2*np.pi*450*t)+.04*np.sin(2*np.pi*900*t)).astype('<f4')
target = ROOT / '.test-runtime' / 'fall-smoke.f32'
target.parent.mkdir(exist_ok=True)
target.write_bytes(audio.tobytes())
print(target)
