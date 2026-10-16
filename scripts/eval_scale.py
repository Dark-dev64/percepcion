#!/usr/bin/env python3
"""
Evaluacion por ESCALA y por RECORTE de la persona.

La metrica global (mAP50) no explica por que el modelo falla con gente cerca
o de espaldas. Este script mide justo eso:

  * tasa de deteccion por tamano de la persona en el encuadre
  * tasa de deteccion segun la persona este recortada por el borde o no
  * reparto de clases predichas, para detectar si el modelo se colapsa sobre
    una clase concreta

Uso:
    python3 scripts/eval_scale.py --weights runs/yolov8s_zoom/weights/best.pt
    python3 scripts/eval_scale.py --weights A.pt --weights B.pt --compare
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
PERSON = {0, 1, 2, 4, 5}

# cortes de area relativa de la caja (fraccion del encuadre)
SIZE_BINS = [(0.0, 0.02, "diminuta <2%"),
             (0.02, 0.08, "lejana 2-8%"),
             (0.08, 0.25, "media 8-25%"),
             (0.25, 1.01, "cercana >25%")]


def load_cases(data_yaml: Path):
    """Devuelve [(ruta, clases_reales, cajas, recortada)] del split de test."""
    import yaml
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg["path"])
    test = cfg.get("test", "images/test")
    # `test: images/test` -> las etiquetas estan en `<path>/labels/test`
    lbl_dir = root / "labels" / Path(test).name
    img_dir = root / test
    if not lbl_dir.is_dir():
        raise SystemExit(f"No se encuentra el directorio de labels: {lbl_dir}")
    cases = []
    for lbl in sorted(lbl_dir.glob("*.txt")):
        img = next((img_dir / f"{lbl.stem}{e}"
                    for e in (".jpg", ".jpeg", ".png")
                    if (img_dir / f"{lbl.stem}{e}").is_file()), None)
        if img is None:
            continue
        boxes = []
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                boxes.append((int(p[0]), *[float(x) for x in p[1:5]]))
        if not any(b[0] in PERSON for b in boxes):
            continue
        p = next(b for b in boxes if b[0] in PERSON)
        cx, cy, w, h = p[1], p[2], p[3], p[4]
        trunc = (cx - w / 2 < 0.02 or cx + w / 2 > 0.98
                 or cy - h / 2 < 0.02 or cy + h / 2 > 0.98)
        cases.append((str(img), {b[0] for b in boxes}, p[3] * p[4], trunc))
    return cases


def evaluate(weights: str, cases, conf: float, imgsz: int, label: str):
    from ultralytics import YOLO
    m = YOLO(weights)
    names = m.names

    by_size = defaultdict(lambda: [0, 0])     # [con deteccion, total]
    by_trunc = defaultdict(lambda: [0, 0])
    pred_counter = Counter()
    matched_posture = 0
    total_person = 0

    for path, real_cls, area, trunc in cases:
        im = cv2.imread(path)
        if im is None:
            continue
        r = m.predict(im, conf=conf, imgsz=imgsz, device=0, verbose=False)[0]
        got = {int(c) for c in r.boxes.cls} if r.boxes is not None else set()

        bin_name = next(n for lo, hi, n in SIZE_BINS if lo <= area < hi)
        by_size[bin_name][1] += 1
        by_trunc["recortada" if trunc else "completa"][1] += 1
        if got & PERSON:
            by_size[bin_name][0] += 1
            by_trunc["recortada" if trunc else "completa"][0] += 1
        total_person += 1
        # la postura solo cuenta si es una de las 4 clases de postura
        if got & {0, 1, 2, 5}:
            matched_posture += 1
        for c in got:
            pred_counter[names[c]] += 1

    print(f"\n{'=' * 64}\n{label}\n  pesos: {weights}\n{'=' * 64}")
    print("\nDeteccion de persona segun su tamano en el encuadre:")
    print(f"  {'rango':18} {'detectada':>10}")
    for _, _, name in SIZE_BINS:
        hit, tot = by_size[name]
        if tot:
            bar = "#" * int(30 * hit / tot)
            print(f"  {name:18} {hit:>4}/{tot:<5} {100*hit/tot:>3.0f}%  {bar}")

    print("\nDeteccion segun la persona este recortada por el borde:")
    print(f"  {'caso':18} {'detectada':>10}")
    for name in ("completa", "recortada"):
        hit, tot = by_trunc[name]
        if tot:
            bar = "#" * int(30 * hit / tot)
            print(f"  {name:18} {hit:>4}/{tot:<5} {100*hit/tot:>3.0f}%  {bar}")

    tot_pred = sum(pred_counter.values())
    print("\nClases predichas:")
    for k, v in pred_counter.most_common():
        share = 100 * v / tot_pred if tot_pred else 0
        flag = "  <-- colapsada" if share > 60 else ""
        print(f"  {k:22} {v:>5}  {share:>5.1f}%{flag}")
    if total_person:
        print(f"\n  recibe clase de postura el {100*matched_posture/total_person:.1f}% "
              f"de las imagenes con persona")
    return {
        "size": {n: by_size[n][0] / by_size[n][1] for n in by_size if by_size[n][1]},
        "trunc": {n: by_trunc[n][0] / by_trunc[n][1]
                  for n in by_trunc if by_trunc[n][1]},
        "posture_rate": matched_posture / total_person if total_person else 0,
        "pred_dist": {k: v / tot_pred for k, v in pred_counter.items()} if tot_pred else {},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", action="append", required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "datasets/combined/data.yaml")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-cases", type=int, default=400)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    cases = load_cases(args.data)
    rng = __import__("random").Random(42)
    rng.shuffle(cases)
    cases = cases[:args.max_cases]
    print(f"casos de evaluacion: {len(cases)}")

    results = []
    for w in args.weights:
        p = Path(w)
        if not p.is_absolute():
            p = ROOT / p
        results.append((w, evaluate(str(p), cases, args.conf, args.imgsz, p.parent.parent.name)))

    if args.compare and len(results) == 2:
        (_, a), (_, b) = results
        print(f"\n{'=' * 64}\nCOMPARATIVA\n{'=' * 64}")
        print(f"\n{'metrica':32} {'antes':>10} {'ahora':>10}  delta")
        for key in ("posture_rate",):
            print(f"  {key:30} {a[key]:>9.1%} {b[key]:>9.1%}  "
                  f"{b[key]-a[key]:+.1%}")
        for group in ("size", "trunc"):
            for n in a[group]:
                if n in b[group]:
                    d = b[group][n] - a[group][n]
                    mark = "mejora" if d > 0.005 else ("igual" if abs(d) <= 0.005 else "PEOR")
                    print(f"  {group+'/'+n:30} {a[group][n]:>9.1%} {b[group][n]:>9.1%}  "
                          f"{d:+.1%}  {mark}")


if __name__ == "__main__":
    main()