#!/usr/bin/env python3
"""
Construye una VARIANTE del dataset unificado añadiendo el dataset sintetico
de Kaggle (Simuletic, CCTV incident), y solo a train/val.

El test NO se toca: es exactamente el mismo que usa el modelo entregado, para
que la comparacion sea justa. Si la variante no mejora en ese test, no sirve
y se descarta.

Contexto de por que esto es dudoso y hay que medirlo:
  * las 111 imagenes son TODAS de la clase "tumbado": no hay ninguna de pie,
    asi que no aporta contraste, solo refuerza `persona_caido`
  * son sintéticas y de un parking con camara cenital y viñeteado, nada
    parecido a una escalera vista de frente
  * por eso el script solo anade y luego hay que COMPARAR

Uso:
    python3 scripts/build_kaggle_variant.py
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "datasets" / "combined"
KAGGLE = ROOT / "datasets" / "_external" / "kaggle_fall" / "laying_dataset"
OUT = ROOT / "datasets" / "combined_kaggle"

# clase del dataset de Kaggle -> clase de la taxonomia unificada
KAGGLE_TO_UNIFIED = {0: 0}      # 0 = laying -> persona_caido


def link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--val-ratio", type=float, default=0.2,
                    help="Fraccion de las imagenes de Kaggle que va a val")
    args = ap.parse_args()
    out = args.out

    if out.exists():
        shutil.rmtree(out)

    # ------------------------------------------------------------------ #
    # 1. Copia la base con symlinks: solo cambia train/val
    # ------------------------------------------------------------------ #
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        src_i = BASE / "images" / split
        src_l = BASE / "labels" / split
        for f in src_i.iterdir():
            link(f, out / "images" / split / f.name)
        for f in src_l.iterdir():
            link(f, out / "labels" / split / f.name)

    # ------------------------------------------------------------------ #
    # 2. Anade las imagenes de Kaggle SOLO a train/val
    # ------------------------------------------------------------------ #
    import random
    rng = random.Random(42)
    names = yaml.safe_load((BASE / "data.yaml").read_text())["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))

    added = Counter()
    k_labels = sorted((KAGGLE / "labels").glob("*.txt"))
    stems = [l.stem for l in k_labels]
    rng.shuffle(stems)
    n_val = int(len(stems) * args.val_ratio)

    for i, stem in enumerate(stems):
        lbl = KAGGLE / "labels" / f"{stem}.txt"
        img = next((KAGGLE / "images" / f"{stem}{e}" for e in (".png", ".jpg")
                    if (KAGGLE / "images" / f"{stem}{e}").is_file()), None)
        if img is None:
            continue
        parts = lbl.read_text().split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls not in KAGGLE_TO_UNIFIED:
            continue
        cx, cy, w, h = (float(x) for x in parts[1:5])
        unified = KAGGLE_TO_UNIFIED[cls]

        split = "val" if i < n_val else "train"
        out_name = f"kaggle_{stem}{img.suffix}"
        link(img, out / "images" / split / out_name)
        (out / "labels" / split / f"kaggle_{stem}.txt").write_text(
            f"{unified} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
        )
        added[(split, unified)] += 1

    # train_zoom se reference igual que en la base
    zoom_img = BASE / "images" / "train_zoom"
    if zoom_img.is_dir():
        (out / "images" / "train_zoom").mkdir(parents=True, exist_ok=True)
        (out / "labels" / "train_zoom").mkdir(parents=True, exist_ok=True)
        for f in zoom_img.iterdir():
            link(f, out / "images" / "train_zoom" / f.name)
        for f in (BASE / "labels" / "train_zoom").iterdir():
            link(f, out / "labels" / "train_zoom" / f.name)

    (out / "data.yaml").write_text(
        "# Variante con el dataset sintetico de Kaggle SOLO en train/val.\n"
        "# El test es identico al de la base para poder comparar.\n"
        f"path: {out.resolve()}\n"
        "train:\n  - images/train\n  - images/train_zoom\n"
        "val: images/val\n"
        "test: images/test\n"
        "\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in names.items()),
        encoding="utf-8",
    )
    (out / "data_zoom.yaml").write_text((out / "data.yaml").read_text(),
                                         encoding="utf-8")

    print(f"Variante creada en {out}")
    print(f"\n  imagenes anadidas de Kaggle:")
    for (split, cid), n in sorted(added.items()):
        print(f"    {split:6} {names[cid]:22} {n:>4}")
    for split in ("train", "val", "test"):
        n = len(list((out / "images" / split).iterdir()))
        print(f"  {split:6}: {n} imagenes (test intacto: {split == 'test'})")


if __name__ == "__main__":
    main()