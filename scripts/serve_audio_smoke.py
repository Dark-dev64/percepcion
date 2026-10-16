"""Servidor SOLO para pruebas locales: detector real, sin cargar YOLO.

El navegador de pruebas simula la respuesta /api/frame y las fuentes multimedia.
Este script no sustituye app/server.py para usar el sistema real.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
sys.path.insert(0, str(ROOT))
import server
from audio_live import AudioEventStore

if __name__ == '__main__':
    server.AUTH_ON = False
    server.audio_events = AudioEventStore(ROOT / '.test-runtime' / 'smoke_alerts.jsonl')
    server.audio_sessions.store = server.audio_events
    server.app.run(host='127.0.0.1', port=5057, threaded=True, use_reloader=False)
