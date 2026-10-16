#!/usr/bin/env python3
"""
Etiqueta los fotogramas extraidos de los clips con caja (COCO) y postura
(geometria de pose), y construye la variante de entrenamiento.

DISEÑO ANTI-SOBREAJUSTE
-----------------------
1. **Reparto por VÍDEO, nunca por fotograma.** Los fotogramas de un clip son
   casi idénticos: si uno va a train y otro a val, el modelo memoriza el
   fotograma y las métricas salen infladas. Aqui un clip entero cae en un
   unico split.
2. **Clips como unico origen de las cajas de video.** Los fotogramas se
   etiquetan por separado pero el reparto se hereda del clip, de modo que no
   se puede colar el mismo instante en dos splits.
3. **Techo de cajas por clase en el entrenamiento** (`max_boxes_per_class`),
   ya presente en `scripts/train.py`: aunque los clips sumen 68 imagenes, si
   metieran una clase mayoritaria no podrían dominar el gradiente.
4. **El test del dataset BASE no se toca.** La comparacion final se hace
   contra ese test intacto, que no contiene ningun fotograma de estos clips.

Caja: detector COCO de 80 clases, que en escenas de interior con una persona
larga acierta de forma fiable (medido 0.92 en fotos reales de escalera).
Postura: geometria de la pose de MediaPipe.

Uso:
    python scripts/label_video_frames.py
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "datasets" / "_video_frames"
BASE = ROOT / "datasets" / "combined"
OUT = ROOT / "datasets" / "combined_video"
WEIGHTS = ROOT / "runs" / "yolov8s_zoom" / "weights" / "best.pt"


def link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=FRAMES)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    sys_path = str(ROOT / "app")
    import sys
    sys.path.insert(0, sys_path)
    from risk_engine import RiskEngine

    print("Cargando motor para etiquetar...")
    eng = RiskEngine(str(WEIGHTS), "yolov8s.pt",
                     obstacle_every=10, pose_every=1, people_every=1)

    if args.out.exists():
        shutil.rmtree(args.out)
    for split in ("train", "val", "test"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (args.out / "images" / "train_zoom").mkdir(parents=True, exist_ok=True)
    (args.out / "labels" / "train_zoom").mkdir(parents=True, exist_ok=True)
    # Los fotogramas del clip de TEST no van a `test/`: ese directorio tiene
    # que seguir identico al de la base para que la comparacion con el modelo
    # entregado sea justa. Se guardan aparte, en `domains/`, y se evaluan por
    # separado como medida de dominio real.
    (args.out / "domains" / "video_test" / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "domains" / "video_test" / "labels").mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader((args.frames / "frames.csv").open()))
    counters: Counter = Counter()
    sin_persona = 0
    detalle = []

    for r in rows:
        path = args.frames / "frames" / r["name"]
        img = cv2.imread(str(path))
        if img is None:
            continue
        split = r["split"]
        eng.reset()
        result = eng.step(img)

        people = [d for d in result["detecciones"]
                  if d["class_name"].startswith("persona")
                  and d["class_name"] != "persona"]
        if not people:
            # Sin postura reconocible: se descarta. Preferimos menos datos a
            # meter etiquetas inventadas, que es lo que arruino el dataset de
            # Kaggle.
            sin_persona += 1
            continue
        d = max(people, key=lambda x: (x["bbox"][2] - x["bbox"][0])
                * (x["bbox"][3] - x["bbox"][1]))
        cid = d["class_id"]
        x1, y1, x2, y2 = d["bbox"]
        h, w = img.shape[:2]
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        if bw <= 0 or bh <= 0:
            sin_persona += 1
            continue

        name = f"vid_{r['name']}"
        # El clip de test va a `domains/video_test/`, NO a `test/`.
        # El resto va a SU carpeta de split: la disposicion del dataset es
        # `images/<split>/` y `labels/<split>/`, y `train_zoom` va aparte.
        if split == "test":
            img_dir = args.out / "domains" / "video_test" / "images"
            lbl_dir = args.out / "domains" / "video_test" / "labels"
        elif split == "train_zoom":
            img_dir = args.out / "images" / "train_zoom"
            lbl_dir = args.out / "labels" / "train_zoom"
        else:
            img_dir = args.out / "images" / split
            lbl_dir = args.out / "labels" / split
        link(path, img_dir / name)
        (lbl_dir / f"{Path(name).stem}.txt").write_text(
            f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
        )
        counters[(split, d["class_name"])] += 1
        detalle.append({"split": split, "video": r["video"],
                        "frame": r["frame"], "clase": d["class_name"],
                        "origen": d.get("origen", "-")})

    # base intacta + train_zoom
    for split in ("train", "val", "test"):
        for f in (BASE / "images" / split).iterdir():
            link(f, args.out / "images" / split / f.name)
        for f in (BASE / "labels" / split).iterdir():
            link(f, args.out / "labels" / split / f.name)
    if (BASE / "images" / "train_zoom").is_dir():
        (args.out / "images" / "train_zoom").mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / "train_zoom").mkdir(parents=True, exist_ok=True)
        for f in (BASE / "images" / "train_zoom").iterdir():
            link(f, args.out / "images" / "train_zoom" / f.name)
        for f in (BASE / "labels" / "train_zoom").iterdir():
            link(f, args.out / "labels" / "train_zoom" / f.name)

    names = yaml.safe_load((BASE / "data.yaml").read_text())["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    txt = ("# Variante con fotogramas de clips propios, repartidos POR VIDEO.\n"
           "# El test es el de la base, intacto: no contiene estos clips.\n"
           f"path: {args.out.resolve()}\n"
           "train:\n  - images/train\n  - images/train_zoom\n"
           "val: images/val\ntest: images/test\n\nnames:\n"
           + "".join(f"  {i}: {n}\n" for i, n in names.items()))
    (args.out / "data.yaml").write_text(txt, encoding="utf-8")
    (args.out / "data_zoom.yaml").write_text(txt, encoding="utf-8")

    with (args.out / "video_frames.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "video", "frame",
                                           "clase", "origen"])
        w.writeheader()
        w.writerows(detalle)

    print(f"\nFotogramas etiquetados: {len(detalle)}")
    print(f"Descartados por no identificar persona: {sin_persona}")
    print("\n  reparto:")
    for (split, cls), n in sorted(counters.items()):
        print(f"    {split:6} {cls:22} {n:>4}")
    print(f"\nVariante en {args.out}")
    for split in ("train", "val", "test"):
        print(f"  {split:6}: {len(list((args.out / 'images' / split).iterdir()))} imagenes")


if __name__ == "__main__":
    main()