#!/usr/bin/env python3
"""
Evaluacion aislada de un checkpoint YOLOv8 ya entrenado.

Permite obtener metricas sin repetir el entrenamiento (util si el proceso
se interrumpio, o para reevaluar con otra configuracion).

Uso:
    python3 scripts/evaluate.py --weights runs/yolov8n_balanced/weights/best.pt
    python3 scripts/evaluate.py --weights ... --imgsz 960
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from train import evaluate, save_report  # noqa: E402  (reutiliza la logica)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "datasets/combined/data_balanced.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", default=None,
                    help="Carpeta de salida; por defecto la del propio run.")
    args = ap.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import torch
    from ultralytics import YOLO

    weights = args.weights if args.weights.is_absolute() else ROOT / args.weights
    if not weights.is_file():
        raise SystemExit(f"No existe el checkpoint: {weights}")

    out_dir = (
        (ROOT / args.name) if args.name else weights.parent.parent
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Modelo  : {weights.name}")
    print(f"Datos   : {args.data}")
    print(f"Salida  : {out_dir}")

    model = YOLO(str(weights))
    metrics = {}
    for split in ("val", "test"):
        metrics[split] = evaluate(
            model, str(args.data), split, args.device, args.imgsz,
            args.batch, out_dir,
        )

    cfg = {
        "model": weights.stem,
        "imgsz": args.imgsz, "batch": args.batch, "seed": seed,
        "data": str(args.data), "data_final": str(args.data),
        "epochs": "checkpoint previo", "weights": str(weights),
    }
    save_report(out_dir / "INFORME.txt", cfg, metrics)
    (out_dir / "metricas.json").write_text(
        json.dumps({"config": cfg, "metricas": metrics}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 64)
    print("RESULTADOS")
    print("=" * 64)
    for split in ("val", "test"):
        m = metrics[split]
        print(f"\n{split.upper()}")
        print(f"  precision {m['precision']:.4f} | recall {m['recall']:.4f} | "
              f"mAP50 {m['mAP50']:.4f} | mAP50-95 {m['mAP50-95']:.4f}")
        for name, v in sorted(m["per_class"].items(), key=lambda kv: kv[1]["mAP50"]):
            print(f"    {name:20} mAP50={v['mAP50']:.4f}  mAP50-95={v['mAP50-95']:.4f}")
        worst = min(m["per_class"].items(), key=lambda kv: kv[1]["mAP50"])
        print(f"    -> peor clase: {worst[0]} (mAP50={worst[1]['mAP50']:.4f})")

    # RQNF10 de Caso.md exige mAP@0.5 > 75%
    for split in ("val", "test"):
        ok = "CUMPLE" if metrics[split]["mAP50"] > 0.75 else "NO CUMPLE"
        print(f"\nRQNF10 (mAP50 > 0.75) en {split}: {ok} "
              f"({metrics[split]['mAP50']:.4f})")


if __name__ == "__main__":
    sys.exit(main())