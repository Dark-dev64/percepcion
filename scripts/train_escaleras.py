#!/usr/bin/env python3
"""
Fine-tuning de YOLOv8 con el dataset de videos de Escaleras.

Particion:
  - Estándar (ubicación de cámara): frames particionados train/val/test.
  - Extra (suplementario): todos a train.

El modelo parte de runs/yolov8s_zoom/weights/best.pt (transfer learning).

Uso:
    python3 scripts/train_escaleras.py
    python3 scripts/train_escaleras.py --epochs 100 --batch 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path,
                    default=ROOT / "configs" / "train_escaleras.yaml")
    ap.add_argument("--set", nargs="*", default=[],
                    help="Sobrescribe claves, p.ej. --set epochs=150")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--resume", type=Path, default=None,
                    help="Reanuda desde un checkpoint last.pt")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    for kv in args.set:
        k, _, v = kv.partition("=")
        cur = cfg.get(k)
        if isinstance(cur, bool):
            cfg[k] = v.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            cfg[k] = int(v)
        elif isinstance(cur, float):
            cfg[k] = float(v)
        else:
            cfg[k] = v

    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch:
        cfg["batch"] = args.batch
    if args.imgsz:
        cfg["imgsz"] = args.imgsz

    # Reproducibilidad
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    project = str(ROOT / cfg["project"])
    name = cfg["name"]
    run_dir = Path(project) / name

    print("=" * 64)
    print("FINE-TUNING YOLOv8 - Dataset Escaleras")
    print("=" * 64)
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"dataset: {cfg['data']}")
    print(f"modelo base: {cfg['model']}")
    print(f"epocas: {cfg['epochs']} | imgsz: {cfg['imgsz']} | batch: {cfg['batch']}")
    print(f"seed: {seed}")
    print(f"salida: {run_dir}")

    # Cargar modelo existente para fine-tuning
    model_path = cfg["model"]
    if not model_path.endswith(".pt"):
        model_path = f"{model_path}.pt"
    weights = Path(model_path)
    if not weights.is_absolute():
        weights = ROOT / weights
    if not weights.is_file():
        raise SystemExit(f"No existe el modelo base: {weights}")

    print(f"\nCargando {weights}...")
    model = YOLO(str(weights))

    # Entrenamiento
    print("\nEntrenando...")
    resume_path = None
    if args.resume:
        resume_path = args.resume if args.resume.is_absolute() else ROOT / args.resume
        if not resume_path.is_file():
            raise SystemExit(f"No existe el checkpoint: {resume_path}")
        print(f"  reanudando desde {resume_path}")
    results = model.train(
        data=str(ROOT / cfg["data"]),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        seed=seed,
        device=cfg["device"],
        workers=cfg["workers"],
        patience=cfg["patience"],
        project=project,
        name=name,
        exist_ok=True,
        resume=resume_path,
        hsv_h=cfg["hsv_h"], hsv_s=cfg["hsv_s"], hsv_v=cfg["hsv_v"],
        degrees=cfg["degrees"], translate=cfg["translate"], scale=cfg["scale"],
        shear=cfg["shear"], flipud=cfg["flipud"], fliplr=cfg["fliplr"],
        mosaic=cfg["mosaic"], close_mosaic=cfg["close_mosaic"], mixup=cfg["mixup"],
        lr0=cfg["lr0"], lrf=cfg["lrf"], momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"], warmup_epochs=cfg["warmup_epochs"],
        optimizer=cfg["optimizer"], cos_lr=cfg["cos_lr"],
        val=cfg["val"], plots=cfg["plots"], save_period=cfg["save_period"],
        verbose=True,
    )

    # Evaluación
    print("\n" + "=" * 64)
    print("EVALUACION")
    print("=" * 64)

    best_path = run_dir / "weights" / "best.pt"
    if not best_path.exists():
        best_path = run_dir / "weights" / "last.pt"

    print(f"Cargando mejor modelo: {best_path}")
    best = YOLO(str(best_path))

    metrics: dict = {}
    for split in ("val", "test"):
        print(f"\n--- {split.upper()} ---")
        res = best.val(
            data=str(ROOT / cfg["data"]), split=split,
            device=cfg["device"], imgsz=cfg["imgsz"],
            batch=cfg["batch"], plots=True, verbose=False,
            project=str(run_dir), name=f"eval_{split}",
        )
        box = res.box
        names = res.names

        per_class = {}
        for i, cls_idx in enumerate(box.ap_class_index):
            cid = int(cls_idx)
            per_class[names[cid]] = {
                "mAP50": round(float(box.ap50[i]), 4),
                "mAP50-95": round(float(box.ap[i]), 4),
            }

        m = {
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
            "mAP50": round(float(box.map50), 4),
            "mAP50-95": round(float(box.map), 4),
            "per_class": per_class,
        }
        metrics[split] = m

        print(f"  precision {m['precision']:.4f} | recall {m['recall']:.4f} | "
              f"mAP50 {m['mAP50']:.4f} | mAP50-95 {m['mAP50-95']:.4f}")
        for name_p, v in sorted(per_class.items(), key=lambda kv: kv[1]["mAP50"]):
            print(f"    {name_p:20} mAP50={v['mAP50']:.4f}  mAP50-95={v['mAP50-95']:.4f}")

    # Guardar métricas
    cfg_serializable = {k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()}
    (run_dir / "metricas.json").write_text(
        json.dumps({"config": cfg_serializable, "metricas": metrics},
                    indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 64)
    print(f"Artefactos: {run_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
