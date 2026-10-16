"""Compara pesos/predicciones del joblib nativo con el cargador compatible.

Ejecutar con --native en un entorno NumPy 2; sin flag en NumPy 1.26.
La salida JSON debe coincidir en los hashes y en las predicciones.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits

path = ROOT / 'audio_fall_detector/models/fall_classifier.joblib'
if '--native' in sys.argv:
    import joblib
    payload = joblib.load(path)
else:
    from audio_fall_detector.model_compat import load_original_model
    payload = load_original_model(path)
model = payload['model']
digest = hashlib.sha256()
for head in model.calibrated_classifiers_:
    for array in head.estimator.coefs_ + head.estimator.intercepts_:
        digest.update(array.tobytes())
matrix = np.array([np.arange(2048, dtype=np.float32)/2048, np.ones(2048,dtype=np.float32)/2048])
with threadpool_limits(1):
    predictions = model.predict_proba(matrix)
print(json.dumps(dict(numpy=np.__version__, weights_sha256=digest.hexdigest(),
                      probabilities=predictions.tolist(), classes=model.classes_.tolist())))
