#!/usr/bin/env python3
"""
Extracción y etiquetado de fotogramas de los videos de Escaleras.

PARTICIONAMIENTO (por video, nunca por fotograma — evita fuga de datos):
  - Estándar (ubicación real de la cámara): 4 train, 1 val, 1 test.
  - Extra (suplementario): todos a train.

Los fotogramas se extracción con submuestreo temporal (stride) para evitar
redundancia, y se etiquetan con el modelo existente (yolov8s_zoom/best.pt)
usando pseudo-etiquetado con umbral de confianza.

Uso:
    python3 scripts/extract_escaleras_frames.py
    python3 scripts/extract_escaleras_frames.py --stride 30 --conf 0.45
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent

# --- Configuración de partición por video -------------------------------------
# Estándar: los videos 01, 04, 05, 06 → train; 02 → val; 03 → test.
# Extra: todos → train (datos suplementarios).
VIDEO_SPLIT_ESTANDAR = {
    "01.mp4": "train",
    "02.mp4": "val",
    "03.mp4": "test",
    "04.mp4": "train",
    "05.mp4": "train",
    "06.mp4": "train",
}

# Resolución nativa: 832x464 (Estándar, 60fps) y 848x478 (Extra, 30fps).
# imgsz del modelo es 640: el video entra tal cual, YOLO lo redimensiona.
ESTANDAR_STRIDE = 15  # 60fps / 15 = 4 fps para train
ESTANDAR_VALSTRIDE = 8   # 60fps / 8 = 7.5 fps para val/test (mas frames para evaluar)
EXTRA_STRIDE = 4     # 30fps / 4 ≈ 7.5 fps

# Umbral de confianza para aceptar pseudo-etiquetas.
DEFAULT_CONF = 0.40

# Clases que el modelo detecta (debe coincidir con yolov8s_zoom).
CLASS_NAMES = [
    "persona_caido",      # 0
    "persona_sentado",    # 1
    "persona_erguida",    # 2
    "escalera",           # 3
    "persona",            # 4
    "persona_desequilibrio",  # 5
]

# Solo etiquetamos frames que contengan al menos una detección válida.
# Los frames sin detecciones se descartan (no aportan al entrenamiento).


def extract_and_label(
    video_path: Path,
    split: str,
    stride: int,
    model,
    out_root: Path,
    rows: list,
    counters: Counter,
    conf_thres: float = DEFAULT_CONF,
) -> None:
    """Extrae frames de un video, los etiqueta con el modelo y los guarda."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [aviso] no se pudo abrir {video_path.name}")
        return

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{video_path.parent.name.lower()}_{video_path.stem.replace(' ', '_')[:30]}"
    n_saved = 0
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            name = f"{prefix}_f{i:05d}"
            _write_frame_and_label(frame, name, model, img_dir, lbl_dir, split, video_path.name, i, rows, counters, conf_thres)
            n_saved += 1
        i += 1
    cap.release()
    print(f"  {video_path.name[:46]:46} -> {split:5} {n_saved:>4} frames "
          f"(de {n_frames} @ {fps:.1f}fps, cada {stride})")


def _write_frame_and_label(
    frame, name, model, img_dir, lbl_dir, split, video, frame_idx, rows, counters, conf_thres
) -> None:
    """Etiqueta un frame con YOLO y escribe imagen + label si hay detecciones."""
    img_path = img_dir / f"{name}.jpg"
    cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

    results = model(img_path, conf=conf_thres, iou=0.5, verbose=False)[0]
    h, w = frame.shape[:2]

    label_lines = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        counters[(split, class_name)] += 1

        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        rows.append({
            "video": video, "split": split, "frame": frame_idx,
            "name": f"{name}.jpg", "class": class_name, "conf": f"{conf:.3f}",
        })

    if not label_lines:
        img_path.unlink(missing_ok=True)
        return

    lbl_path = lbl_dir / f"{name}.txt"
    lbl_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "Escaleras")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "escaleras_dataset")
    ap.add_argument("--stride", type=int, default=None,
                    help="Stride global (sobreescribe por tipo). Default: 30 Estándar / 8 Extra.")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF,
                    help="Umbral de confianza para pseudo-etiquetas")
    ap.add_argument("--weights", type=Path,
                    default=ROOT / "runs" / "yolov8s_zoom" / "weights" / "best.pt")
    args = ap.parse_args()

    weights = args.weights.resolve()
    src = args.src.resolve()
    if not src.is_dir():
        raise SystemExit(f"No existe: {src}")

    if not weights.is_file():
        raise SystemExit(f"No existe el modelo: {weights}")

    if args.out.exists():
        shutil.rmtree(args.out)
    for split in ("train", "val", "test"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Extracción y pseudo-etiquetado - Videos Escaleras")
    print(f"  Modelo: {weights.name} (conf={args.conf})")
    print(f"  Salida: {args.out}")
    print("=" * 64)

    from ultralytics import YOLO
    model = YOLO(str(weights))
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    rows: list = []
    counters: Counter = Counter()

    stride = args.stride
    estandar_stride = stride if stride else ESTANDAR_STRIDE
    extra_stride = stride if stride else EXTRA_STRIDE

    # --- Estándar ---
    print("\n--- Estándar (ubicación de cámara) ---")
    est_dir = src / "Estándar"
    for video in sorted(est_dir.glob("*.mp4")):
        split = VIDEO_SPLIT_ESTANDAR.get(video.name, "train")
        if split is None:
            split = "train"
            print(f"  [aviso] '{video.name}' sin split -> train")
        # Para val/test se usa un stride menor para obtener mas frames
        vstride = estandar_stride if split == "train" else ESTANDAR_VALSTRIDE
        extract_and_label(video, split, vstride, model, args.out, rows, counters, args.conf)

    # --- Extra ---
    print("\n--- Extra (suplementario → train) ---")
    extra_dir = src / "Extra"
    for video in sorted(extra_dir.glob("*.mp4")):
        extract_and_label(video, "train", extra_stride, model, args.out, rows, counters, args.conf)

    # --- Escribe data.yaml ---
    header = (
        "# Dataset de videos de Escaleras - generado por scripts/extract_escaleras_frames.py\n"
        "# NO editar a mano.\n"
    )
    data_yaml = (
        header
        + f"path: {args.out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    (args.out / "data.yaml").write_text(data_yaml, encoding="utf-8")

    # --- Reporte ---
    print("\n" + "=" * 64)
    print("RESUMEN")
    print("=" * 64)

    frame_counts: dict[str, int] = defaultdict(int)
    for split in ("train", "val", "test"):
        n = len(list((args.out / "images" / split).glob("*.jpg")))
        frame_counts[split] = n
        print(f"  {split:6}: {n:>5} imágenes")

    print(f"\n  Total: {sum(frame_counts.values())} imágenes")
    print(f"\nDetecciones por split y clase:")
    for (split, cls), n in sorted(counters.items()):
        print(f"    {split:6} {cls:24} {n:>6}")

    (args.out / "detections.csv").write_text(
        "", encoding="utf-8"
    )
    if rows:
        with (args.out / "detections.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["video", "split", "frame", "name", "class", "conf"])
            w.writeheader()
            w.writerows(rows)

    (args.out / "frames_config.json").write_text(
        json.dumps({
            "estandar_stride": estandar_stride,
            "extra_stride": extra_stride,
            "conf_threshold": args.conf,
            "video_split_estandar": VIDEO_SPLIT_ESTANDAR,
            "extra_split": "train",
            "classes": CLASS_NAMES,
            "n_frames": frame_counts,
             "n_detections": {f"{k[0]}|{k[1]}": v for k, v in counters.items()},
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nDataset: {args.out}")


if __name__ == "__main__":
    import torch
    main()
