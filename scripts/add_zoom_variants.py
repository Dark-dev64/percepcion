#!/usr/bin/env python3
"""
Genera variantes de primer plano a partir de las imagenes ya anotadas.

POR QUE
------
El modelo detecta bien a la gente lejos y falla cuando esta cerca o de
espaldas. La causa es medible en los datos de entrenamiento:

    clase                   cajas   recortadas   area mediana
    persona_desequilibrio      357        0.0%          0.069
    persona_erguida            824       19.8%          0.080
    persona (CCTV)            4570        8.1%          0.010

`persona` se aprdio sobre personas pequenas y lejanas (area 1% del encuadre),
y las clases de postura no tienen ni un solo ejemplo de primer plano: el 100%
viene de Le2i, cinco escenas con camara fija y plano abierto.

COMO SE SOLUCIONA
-----------------
Un recorte de la imagen es una transformacion geometrica de las cajas: si se
recorta una region y se reescala, las cajas se transforman de forma exacta y
conservan su etiqueta. Se pueden generar primeros planos REALES sin volver a
anotar nada, y de paso se generan los casos de persona parcialmente fuera de
cuadro (el caso tipico de alguien que pasa muy cerca de la camara).

Ademas cubre "de espaldas" en la medida en que los recortes conservan los
angulos que ya hay en el dataset.

Uso:
    python3 scripts/add_zoom_variants.py --split train
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "datasets" / "combined"

# Clases que representan una persona. `escalera` no se recorta porque al
# ampliar la imagen la escalera se sale de cuadro y la etiqueta pierde sentido.
# Taxonomia de 6 clases: la 3 es `escalera` y queda fuera.
PERSON_CLASSES = {0, 1, 2, 4, 5}

# Fraccion del area final que debe ocupar la caja principal. Valores altos
# producen primeros planos; 0.30-0.45 deja ver el cuerpo entero con la
# persona cerca, que es el caso de una escalera vista de cerca.
TARGET_FILL = (0.30, 0.55)

# Una caja recortada se conserva si sigue siendo visible en su mayor parte.
MIN_VISIBLE = 0.60


def read_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) >= 5:
            rows.append((int(p[0]), *[float(x) for x in p[1:5]]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=None,
                    help="Carpeta de salida; por defecto <split>_zoom")
    ap.add_argument("--per-image", type=int, default=1,
                    help="Variantes por imagen (0 = solo las que tengan sentido)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    img_dir = DATA / "images" / args.split
    lbl_dir = DATA / "labels" / args.split
    out_name = args.out or f"{args.split}_zoom"
    o_img = DATA / "images" / out_name
    o_lbl = DATA / "labels" / out_name
    o_img.mkdir(parents=True, exist_ok=True)
    o_lbl.mkdir(parents=True, exist_ok=True)

    made = 0
    stats = defaultdict(lambda: {"n": 0, "trunc": 0, "area": []})
    manifest = []

    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        img_path = next(
            (img_dir / f"{lbl_path.stem}{ext}"
             for ext in (".jpg", ".jpeg", ".png")
             if (img_dir / f"{lbl_path.stem}{ext}").is_file()),
            None,
        )
        if img_path is None:
            continue
        rows = read_label(lbl_path)
        people = [r for r in rows if r[0] in PERSON_CLASSES]
        if not people:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        H, W = image.shape[:2]
        if min(H, W) < 80:
            continue

        # se toma la persona mas grande como referencia del encuadre
        cls, cx, cy, w, h = max(people, key=lambda r: r[2] * r[3])
        px1, py1 = (cx - w / 2) * W, (cy - h / 2) * H
        px2, py2 = (cx + w / 2) * W, (cy + h / 2) * H
        if px2 - px1 < 24 or py2 - py1 < 24:
            continue

        for k in range(max(1, args.per_image)):
            fill = rng.uniform(*TARGET_FILL)
            # lado del recorte para que la persona ocupe `fill` del encuadre
            side = max(px2 - px1, py2 - py1) / fill
            side = min(side, min(H, W) * 1.9)   # no mas alla del lienzo
            side = max(side, 32.0)

            ox = rng.uniform(px1 - (side - (px2 - px1)) / 2,
                             px2 - (side - (px2 - px1)) / 2)
            oy = rng.uniform(py1 - (side - (py2 - py1)) / 2,
                             py2 - (side - (py2 - py1)) / 2)
            # recorte al lienzo
            ox = min(max(ox, 0.0), max(0.0, W - side))
            oy = min(max(oy, 0.0), max(0.0, H - side))
            side = min(side, W - ox, H - oy)
            if side < 32:
                continue

            x1, y1 = int(round(ox)), int(round(oy))
            x2, y2 = int(round(ox + side)), int(round(oy + side))

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            # se reescala a un tamanoutil: ni tan pequeno que pierda detalle
            scale_out = min(2.0, 640 / max(crop.shape[0], crop.shape[1]))
            nh, nw = max(32, int(crop.shape[0] * scale_out)), \
                     max(32, int(crop.shape[1] * scale_out))
            crop = cv2.resize(crop, (nw, nh))

            # transformacion exacta de las cajas al nuevo encuadre
            new_rows = []
            for c, bx, by, bw, bh in rows:
                nx1 = (bx - bw / 2) * W
                ny1 = (by - bh / 2) * H
                nx2 = (bx + bw / 2) * W
                ny2 = (by + bh / 2) * H
                ix1, iy1 = max(nx1, x1), max(ny1, y1)
                ix2, iy2 = min(nx2, x2), min(ny2, y2)
                iw, ih = ix2 - ix1, iy2 - iy1
                if iw <= 0 or ih <= 0:
                    continue                     # la caja quedo fuera
                orig_area = max((nx2 - nx1) * (ny2 - ny1), 1e-6)
                visible = (iw * ih) / orig_area
                if visible < MIN_VISIBLE:
                    continue                     # demasiado recortada
                ncx = ((ix1 + ix2) / 2 - x1) / side
                ncy = ((iy1 + iy2) / 2 - y1) / side
                nbw = iw / side
                nbh = ih / side
                new_rows.append((c, ncx, ncy, nbw, nbh))
                if c in PERSON_CLASSES:
                    s = stats[c]
                    s["n"] += 1
                    s["area"].append(nbw * nbh)
                    if ncx - nbw / 2 < 0.02 or ncx + nbw / 2 > 0.98 \
                       or ncy - nbh / 2 < 0.02 or ncy + nbh / 2 > 0.98:
                        s["trunc"] += 1

            if not new_rows:
                continue
            # se descarta si el encuadre no gana nada frente al original
            gain = max(r[2] * r[3] for r in new_rows) / max(
                r[2] * r[3] for r in rows
            )
            if gain < 1.25:
                continue

            name = f"z{img_path.stem}_{k}.jpg"
            cv2.imwrite(str(o_img / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (o_lbl / f"z{img_path.stem}_{k}.txt").write_text(
                "".join(f"{c} {a:.6f} {b:.6f} {w2:.6f} {h2:.6f}\n"
                        for c, a, b, w2, h2 in new_rows)
            )
            made += 1
            manifest.append({"out_name": name, "src": img_path.name,
                             "gain": round(gain, 2), "split": args.split})

    names = {0: "persona_caido", 1: "persona_sentado", 2: "persona_erguida",
             4: "persona", 5: "persona_desequilibrio"}
    print(f"variantes de primer plano generadas: {made}  -> {o_img}")
    print(f"\n{'clase':22} {'n':>5} {'recortadas':>11} {'%':>7} {'area med':>9}")
    for c in sorted(stats):
        s = stats[c]
        if not s["n"]:
            continue
        med = float(np.median(s["area"]))
        print(f"  {names.get(c, c):20} {s['n']:>5} {s['trunc']:>11} "
              f"{100*s['trunc']/s['n']:>6.1f}% {med:>9.3f}")

    with (DATA / f"manifest_{out_name}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "src", "out_name", "gain"])
        w.writeheader()
        w.writerows(manifest)

    print(f"\nPara usarlo en el entrenamiento, `datasets/combined/data_zoom.yaml`")
    print(f"incluye esta carpeta automaticamente (prepare_dataset.py la escribe).")


if __name__ == "__main__":
    main()