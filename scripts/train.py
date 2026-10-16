#!/usr/bin/env python3
"""
Paso 3 - Entrenamiento y evaluacion de YOLOv8.

Lee `configs/train.yaml`, entrena el detector y evalua sobre `val` y `test`,
guardando pesos, metricas y graficas. Todo el entrenamiento es reproducible:
semilla fija y configuracion versionada.

Uso:
    python3 scripts/train.py
    python3 scripts/train.py --model yolov8s --epochs 150
    python3 scripts/train.py --config configs/train.yaml --set use_balanced=false
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Balanceo de clases
# --------------------------------------------------------------------------- #
def build_balanced_dataset(cfg: dict) -> Path | None:
    """Crea un subconjunto de entrenamiento con las clases dominantes recortadas.

    El objetivo es que ninguna clase domine el gradiente. `persona` aporta
    ~4.5k cajas de CCTV frente a ~1.4k de todas las clases de postura juntas, y
    el modelo colapsa sobre ella: medido, predecía `persona` el 96% de las
    veces y apenas repartia el resto.

    IMPORTANTE (correccion de un bug anterior): el recorte se decide por
    CONTEO DE CAJAS, no de imagenes. Contar imagenes seleccionaba
    `persona_erguida` --que aparecia en mas imagenes por el peso de Le2i
    Stand-- y la recortaba a la mitad, justo la clase que mas falta hacia
    falta, mientras `persona` se dejaba intacta. El desbalance real esta en
    las cajas, no en las imagenes.

    Como un recorte es una transformacion geometrica exacta de las cajas,
    aplicar el mismo recorte a las variantes de primer plano mantiene las
    etiquetas correctas.

    Devuelve la ruta del data.yaml generado, o None si no aplica.
    """
    data_path = Path(cfg["data"])
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    root = data_path.parent
    names = _read_names(data_path)

    train_dirs = _train_dirs(root, cfg)          # p.ej. train y train_zoom
    target_max = int(cfg.get("max_boxes_per_class", 900))

    for d in train_dirs:
        (root / "images" / d).mkdir(parents=True, exist_ok=True)
        (root / "labels" / d).mkdir(parents=True, exist_ok=True)
    # salida: symlinks, nunca copias, para no duplicar los ~600 MB
    for d in ("train_balanced",):
        for sub in ("images", "labels"):
            shutil.rmtree(root / sub / d, ignore_errors=True)
            (root / sub / d).mkdir(parents=True, exist_ok=True)

    # ---- inventario: cajas e imagenes por clase ------------------------- #
    img_classes: dict[tuple[str, str], set] = {}
    box_counts: Counter = Counter()
    for d in train_dirs:
        img_dir, lbl_dir = root / "images" / d.name, root / "labels" / d.name
        if not lbl_dir.is_dir():
            continue
        for lbl in sorted(lbl_dir.glob("*.txt")):
            stem = lbl.stem
            img = next(
                (img_dir / f"{stem}{e}" for e in (".jpg", ".jpeg", ".png")
                 if (img_dir / f"{stem}{e}").is_file()), None,
            )
            if img is None:
                continue
            cls = {int(float(l.split()[0]))
                   for l in lbl.read_text().splitlines() if l.strip()}
            img_classes[(d.name, stem)] = cls
            # se cuentan CAJAS reales, no clases-imagen: una foto de CCTV con
            # cinco pedestrians aporta cinco cajas de `persona`, y eso es
            # justo lo que compite en el gradiente
            for line in lbl.read_text().splitlines():
                if line.strip():
                    box_counts[int(float(line.split()[0]))] += 1

    if not box_counts:
        print("  [aviso] no hay cajas; no se balancea")
        return None

    # clases que superan el objetivo, con su tasa de conservacion necesaria
    # para llegar al techo. Es adaptativa por clase porque un recorte fijo
    # penaliza a las clases menos numerosas, que son justo las que mas
    # necesitan senal.
    keep_rate: dict[int, float] = {}
    for c, n in box_counts.items():
        keep_rate[c] = 1.0 if n <= target_max else target_max / n
    over = [c for c, n in box_counts.items() if n > target_max]
    if not over:
        print(f"  balanceo: ninguna clase supera {target_max} cajas; se usa tal cual")
    major = box_counts.most_common(1)[0][0]
    rng = random.Random(int(cfg["seed"]))

    kept, dropped = 0, 0
    # El recuento final se acumula AL ESCRIBIR cada etiqueta, no en una
    # segunda pasada: antes se repetia la decision con `rng.random()`, pero
    # `rng` es un unico generador, asi que la segunda vez sorteaba numeros
    # nuevos y el recuento impreso no correspondia a `train_balanced`.
    final: Counter = Counter()
    for (dirn, stem), cls in sorted(img_classes.items()):
        # Tasa de la imagen = la mas generosa entre sus clases, para no
        # descartar una imagen que aporta a una clase que aun necesita datos.
        # Solo se recortan imagenes cuya clase unica esta por encima del techo.
        if cls and len(cls) == 1:
            only = next(iter(cls))
            if keep_rate.get(only, 1.0) < 1.0 and rng.random() > keep_rate[only]:
                dropped += 1
                continue
        src = next(
            (root / "images" / dirn / f"{stem}{e}" for e in (".jpg", ".jpeg", ".png")
             if (root / "images" / dirn / f"{stem}{e}").is_file()), None,
        )
        if src is None:
            continue
        _link(src, root / "images" / "train_balanced" / f"{dirn}__{stem}{src.suffix}")
        _link(root / "labels" / dirn / f"{stem}.txt",
              root / "labels" / "train_balanced" / f"{dirn}__{stem}.txt")
        kept += 1
        lbl = root / "labels" / dirn / f"{stem}.txt"
        if lbl.is_file():
            for line in lbl.read_text().splitlines():
                if line.strip():
                    final[int(float(line.split()[0]))] += 1

    print(
        f"  balanceo por cajas (objetivo {target_max}/clase): "
        f"train {kept} imgs, {dropped} descartadas"
    )
    print("    cajas antes -> despues (contadas sobre lo escrito):")
    for c in sorted(box_counts):
        flag = " <- recortada" if box_counts[c] > target_max else ""
        print(f"      {names.get(c, c):22} {box_counts[c]:>6} -> {final[c]:>6}{flag}")

    out_yaml = root / "data_balanced.yaml"
    out_yaml.write_text(
        "# Generado por scripts/train.py (NO editar a mano).\n"
        f"path: {root.resolve()}\n"
        f"train: images/train_balanced\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in names.items()),
        encoding="utf-8",
    )
    return out_yaml


def _train_dirs(root: Path, cfg: dict) -> list[Path]:
    """Directorios de entrenamiento declarados en el yaml (admite lista)."""
    import yaml as _y
    data = _y.safe_load(Path(cfg["data"]).read_text()) \
        if Path(cfg["data"]).is_absolute() else \
        _y.safe_load((ROOT / cfg["data"]).read_text())
    train = data.get("train", "images/train")
    entries = train if isinstance(train, list) else [train]
    # se queda solo con el nombre del directorio, relativo a `path` del yaml
    return [Path(Path(e).name) for e in entries]


def _link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(os.path.relpath(src, dst.parent))


def _read_names(data_path: Path) -> dict[int, str]:
    data = yaml.safe_load(data_path.read_text())
    raw = data.get("names", {})
    if isinstance(raw, list):
        return dict(enumerate(raw))
    return {int(k): v for k, v in raw.items()}


# --------------------------------------------------------------------------- #
# Evaluacion
# --------------------------------------------------------------------------- #
# Procedencia de las etiquetas de test, clase por clase:
#
#   persona_caido         `fall` (GT humano) + Le2i `Lie` (pseudo, validada 91%)
#   persona_sentado       `fall` (GT humano)
#   persona_erguida       `fall` (GT humano) + Le2i `Stand` (pseudo) +
#                         `tracking` (pseudo-etiquetas de detector)
#   escalera              `stairs` (GT humano)
#   persona_desequilibrio Le2i `Likefall` (pseudo, validada 100%)
#
# `persona_erguida` es la unica que MEZCLA anotacion humana con
# pseudo-etiquetas, porque en la parte 2 se fusiono ahi la clase `persona`
# de CCTV para eliminar la clase generica que se tragaba el 76% de las
# predicciones. Por eso se reporta tambien un mAP50 restringido a las clases
# de etiqueta inequivocamente humana.
GT_HUMAN_CLASSES = ("persona_sentado", "escalera")


def evaluate(model, data: str, split: str, device, imgsz, batch, out_dir: Path) -> dict:
    """Evalua el modelo y devuelve un diccionario de metricas.

    `out_dir` recibe las graficas de cada evaluacion para que pesos,
    metricas y figuras queden juntos en la carpeta de resultados.

    Ademas de la metrica global se calcula `mAP50_gt_humano`: el promedio de
    AP50 restringido a las clases con anotacion humana, excluyendo `persona`.
    """
    res = model.val(
        data=data, split=split, device=device, imgsz=imgsz,
        batch=batch, plots=True, verbose=False,
        project=str(out_dir), name=f"eval_{split}",
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

    honest = [v for k, v in per_class.items() if k in GT_HUMAN_CLASSES]
    out = {
        "split": split,
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "per_class": per_class,
    }
    if honest:
        out["mAP50_gt_humano"] = round(
            sum(v["mAP50"] for v in honest) / len(honest), 4
        )
        out["mAP50_95_gt_humano"] = round(
            sum(v["mAP50-95"] for v in honest) / len(honest), 4
        )
    return out


def rank_worst(per_class: dict, metric: str = "mAP50") -> list[tuple[str, float]]:
    """Ordena las clases de peor a mejor para la metrica indicada."""
    return sorted(
        ((k, v[metric]) for k, v in per_class.items()),
        key=lambda kv: kv[1],
    )


def save_report(path: Path, cfg: dict, metrics: dict) -> None:
    """Volca un informe legible junto a los artefactos del entrenamiento."""
    lines = [
        "Informe de evaluacion - Prevencion de Caidas y Riesgos en Escaleras",
        "=" * 64,
        f"modelo        : {cfg['model']}",
        f"epocas        : {cfg['epochs']}  imgsz: {cfg['imgsz']}  batch: {cfg['batch']}",
        f"seed          : {cfg['seed']}",
        f"dataset       : {cfg.get('data_final', cfg['data'])}",
        "",
    ]
    for split in ("val", "test"):
        m = metrics.get(split)
        if not m:
            continue
        lines += [
            f"--- {split.upper()} ---",
            f"  precision : {m['precision']:.4f}",
            f"  recall    : {m['recall']:.4f}",
            f"  mAP50     : {m['mAP50']:.4f}",
            f"  mAP50-95  : {m['mAP50-95']:.4f}",
        ]
        if "mAP50_gt_humano" in m:
            lines += [
                f"  mAP50 (solo clases con GT humano inequivocable): "
                f"{m['mAP50_gt_humano']:.4f}",
                f"  mAP50-95 (ídem)                                   : "
                f"{m['mAP50_95_gt_humano']:.4f}",
            ]
        lines += ["", "  clase                 mAP50   mAP50-95"]
        for name, v in sorted(m["per_class"].items(), key=lambda kv: kv[1]["mAP50"]):
            lines.append(f"  {name:20} {v['mAP50']:6.4f}   {v['mAP50-95']:7.4f}")
        worst = rank_worst(m["per_class"])
        if worst:
            lines += ["", f"  peor clase ({split}): {worst[0][0]} ({worst[0][1]:.4f})"]
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "train.yaml")
    ap.add_argument("--set", nargs="*", default=[],
                    help="Sobrescribe claves, p.ej. --set epochs=50 batch=8")
    ap.add_argument("--model", help="Sobrescribe el modelo, p.ej. yolov8s")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--no-balanced", action="store_true")
    ap.add_argument("--resume", type=Path, default=None,
                    help="Reanuda desde un checkpoint last.pt (p.ej. tras "
                         "interrumpir el entrenamiento). El resto de "
                         "argumentos se recupera del propio checkpoint.")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    resuming = args.resume is not None
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
    if args.model:
        cfg["model"] = args.model
        cfg["name"] = f"{args.model}_balanced" if cfg.get("use_balanced") else args.model
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch:
        cfg["batch"] = args.batch
    if args.imgsz:
        cfg["imgsz"] = args.imgsz
    if args.no_balanced:
        cfg["use_balanced"] = False

    # reproducibilidad
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    import torch
    from ultralytics import YOLO

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("=" * 64)
    print("ENTRENAMIENTO YOLOv8 - Prevencion de Caidas en Escaleras")
    print("=" * 64)
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # --- datos ------------------------------------------------------------ #
    data = cfg["data"]
    if resuming:
        # Al reanudar, Ultralytics restaura dataset, batch y optimizador desde
        # el checkpoint. Aqui solo se reconstruye el directorio balanceado para
        # que las rutas sigan existiendo.
        print("\n[1/4] Reanudando: se reconstruye el dataset balanceado...")
        if cfg.get("use_balanced"):
            build_balanced_dataset(cfg)
        data = cfg["balanced_data"] if cfg.get("use_balanced") else cfg["data"]
    elif cfg.get("use_balanced"):
        print("\n[1/4] Preparando dataset balanceado...")
        alt = build_balanced_dataset(cfg)
        if alt:
            data = str(alt)
    cfg["data_final"] = data

    print(f"\n[2/4] Dataset: {data}")

    # --- entrenamiento ---------------------------------------------------- #
    run_dir = Path(ROOT / cfg["project"]) / cfg["name"]

    if resuming:
        ckpt = args.resume if args.resume.is_absolute() else ROOT / args.resume
        if not ckpt.is_file():
            raise SystemExit(f"No existe el checkpoint: {ckpt}")
        print(f"\n[3/4] Reanudando desde {ckpt}...")
        model = YOLO(str(ckpt))
        model.train(resume=True)   # el resto sale del checkpoint
    else:
        print(f"\n[3/4] Entrenando {cfg['model']} ({cfg['epochs']} épocas, "
              f"imgsz={cfg['imgsz']}, batch={cfg['batch']})...")
        model = YOLO(f"{cfg['model']}.pt")
        model.train(
            data=data,
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            seed=seed,
            device=cfg["device"],
            workers=cfg["workers"],
            patience=cfg["patience"],
            project=str(ROOT / cfg["project"]),
            name=cfg["name"],
            exist_ok=True,
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

    # --- evaluacion ------------------------------------------------------- #
    print("\n[4/4] Evaluando el mejor modelo...")
    best_path = run_dir / "weights" / "best.pt"
    out_dir = best_path.parent.parent
    best = YOLO(str(best_path))

    metrics = {}
    for split in ("val", "test"):
        metrics[split] = evaluate(
            best, data, split, cfg["device"], cfg["imgsz"], cfg["batch"], out_dir
        )

    save_report(out_dir / "INFORME.txt", cfg, metrics)
    (out_dir / "metricas.json").write_text(
        json.dumps({"config": cfg, "metricas": metrics},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # resumen en consola
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
    print(f"\nArtefactos: {out_dir}")


if __name__ == "__main__":
    sys.exit(main())