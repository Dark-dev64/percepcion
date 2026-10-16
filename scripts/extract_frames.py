#!/usr/bin/env python3
"""
Extrae fotogramas de clips de vídeo para construir un conjunto de datos real
de escaleras, y los REPARTE POR VÍDEO.

POR QUÉ POR VÍDEO Y NO POR FOTOGRAMA
------------------------------------
Fotogramas consecutivos de un vídeo son casi idénticos: la misma persona, la
misma escalera, la misma luz, el mismo ángulo. Si uno cae en train y otro en
val, el modelo memoriza el fotograma y las métricas salen infladas sin que el
sistema sepa hacer nada nuevo. Es el mismo error que ya se corrigió dos veces
en Le2i.

Con pocos clips el reparto por vídeo deja los splits muy pequeños, asi
que además se submuestrea con una cadencia: de 30 fps a 2 fps el contenido
aporta lo mismo y el riesgo de sobreajuste baja.

QUEDA FIJA para que el resultado sea reproducible.

Uso:
    python3 scripts/extract_frames.py --stride 15
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT  # los clips estan en la raiz
OUT = ROOT / "datasets" / "_video_frames"

# Reparto por video: la unidad es el clip, nunca el fotograma.
#   train -> se entrena
#   val   -> seleccion de checkpoint
#   test  -> medida final, NO se toca al entrenar
VIDEO_SPLIT = {
    "videoplayback.mp4": "train",
    "videoplayback (1).mp4": "train",
    "WhatsApp Video 2026-09-26 at 10.44.01 AM.mp4": "val",
    "A primera vista, este video puede parecer gracioso.Una caída torpe, un "
    "momento inesperado… y la .mp4": "test",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--stride", type=int, default=15,
                    help="Tomar 1 de cada N frames (30 fps con 15 -> 2 fps)")
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "frames").mkdir(parents=True, exist_ok=True)

    clips = sorted(p for p in args.src.glob("*.mp4"))
    if not clips:
        raise SystemExit(f"No hay vídeos MP4 en {args.src}")

    rows = []
    for clip in clips:
        split = VIDEO_SPLIT.get(clip.name)
        if split is None:
            # Sin asignacion manual se cae a train, pero se avisa: un clip
            # colocado en train sin querer inflaria train y no valeria para medir.
            split = "train"
            print(f"  [aviso] '{clip.name[:40]}' sin split asignado -> train")
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            print(f"  [aviso] no se pudo abrir {clip.name}")
            continue
        n = 0
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % args.stride == 0:
                name = f"{clip.stem.replace(' ', '_')[:40]}_f{i:05d}.jpg"
                cv2.imwrite(str(args.out / "frames" / name), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                rows.append({"video": clip.name, "split": split,
                             "frame": i, "name": name})
                n += 1
            i += 1
        cap.release()
        print(f"  {clip.name[:46]:46} -> {split:5} {n:>4} fotogramas")

    with (args.out / "frames.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["video", "split", "frame", "name"])
        w.writeheader()
        w.writerows(rows)

    c = Counter((r["split"], r["video"]) for r in rows)
    print(f"\nTotal: {len(rows)} fotogramas de {len(rows and set(r['video'] for r in rows))} vídeos")
    for split in ("train", "val", "test"):
        n = sum(1 for r in rows if r["split"] == split)
        v = len({r["video"] for r in rows if r["split"] == split})
        print(f"  {split:6}: {n:>4} fotogramas de {v} vídeo(s)")

    (args.out / "config.json").write_text(
        json.dumps({"stride": args.stride, "video_split": VIDEO_SPLIT},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFotogramas en {args.out / 'frames'}")
    print("Estos ficheros son TEMPORALES: se borran al terminar el entrenamiento.")


if __name__ == "__main__":
    main()