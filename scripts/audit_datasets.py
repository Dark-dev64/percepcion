#!/usr/bin/env python3
"""
Auditoría de los datasets en `stairs_fallhuman` (Paso 1 del proyecto
"Prevención de Caídas y Riesgos en Escaleras").

No modifica nada: solo lee y reporta. Usa únicamente la librería estándar
para poder ejecutarlo antes de instalar torch/ultralytics.

Uso:
    python3 scripts/audit_datasets.py [--root stairs_fallhuman]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Utilidades de bajo nivel
# --------------------------------------------------------------------------- #
def image_size(path: Path) -> tuple[int, int] | None:
    """Lee ancho/alto de la cabecera del archivo sin depender de Pillow.

    Devuelve None si la cabecera no es reconocible (archivo corrupto).
    """
    try:
        with path.open("rb") as fh:
            head = fh.read(32)
            # JPEG: recorre los segmentos hasta SOFn
            if head[:2] == b"\xff\xd8":
                fh.seek(2)
                while True:
                    byte = fh.read(1)
                    if not byte:
                        return None
                    if byte != b"\xff":
                        continue
                    marker = fh.read(1)
                    while marker == b"\xff":  # bytes de relleno
                        marker = fh.read(1)
                    if not marker:
                        return None
                    code = marker[0]
                    # SOF0..SOF15, saltando los marcadores sin payload
                    if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
                        fh.read(3)  # longitud (2) + precisión (1)
                        h, w = struct.unpack(">HH", fh.read(4))
                        return w, h
                    length = struct.unpack(">H", fh.read(2))[0]
                    fh.seek(length - 2, 1)
            # PNG: IHDR siempre en los primeros bytes
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", head[16:24])
                return w, h
            # BMP
            if head[:2] == b"BM":
                fh.seek(18)
                w, h = struct.unpack("<ii", fh.read(8))
                return w, abs(h)
            # WEBP
            if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                chunk = head[12:16]
                if chunk == b"VP8X":
                    w = int.from_bytes(head[24:27], "little") + 1
                    h = int.from_bytes(head[27:30], "little") + 1
                    return w, h
                if chunk == b"VP8 ":
                    w = struct.unpack("<H", head[26:28])[0] & 0x3FFF
                    h = struct.unpack("<H", head[28:30])[0] & 0x3FFF
                    return w, h
    except Exception:
        return None
    return None


def file_md5(path: Path) -> str:
    """Hash del contenido completo para detectar duplicados exactos."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_yolo_label(path: Path) -> tuple[int, list[tuple[float, ...]]]:
    """Devuelve (n_clases_presentes, boxes) para un .txt en formato YOLO."""
    boxes: list[tuple[float, ...]] = []
    classes: set[int] = set()
    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        classes.add(cls)
        coords = tuple(float(p) for p in parts[1:])
        boxes.append((cls, *coords))
    return len(classes), boxes


def audit_yolo_dir(
    img_dir: Path, lbl_dir: Path, label: str, report: dict
) -> None:
    """Audita un par imágenes/<split> + labels/<split> en formato YOLO."""
    info = {
        "label": label,
        "img_dir": str(img_dir),
        "n_images": 0,
        "n_labels": 0,
        "missing_labels": [],
        "orphan_labels": [],
        "empty_labels": 0,
        "corrupt_images": [],
        "invalid_boxes": [],
        "class_counts": Counter(),
        "boxes_per_image": [],
        "sizes": Counter(),
        "n_boxes": 0,
    }

    images = sorted(
        p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ) if img_dir.is_dir() else []
    info["n_images"] = len(images)

    stem_map = defaultdict(list)
    for img in images:
        stem_map[img.stem].append(img)

    label_files = sorted(lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else []
    info["n_labels"] = len(label_files)

    for img in images:
        lbl = lbl_dir / f"{img.stem}.txt"
        dim = image_size(img)
        if dim is None:
            info["corrupt_images"].append(img.name)
        else:
            info["sizes"][dim] += 1
        if not lbl.exists():
            info["missing_labels"].append(img.name)
            continue
        _, boxes = parse_yolo_label(lbl)
        if not boxes:
            info["empty_labels"] += 1
        info["boxes_per_image"].append(len(boxes))
        info["n_boxes"] += len(boxes)
        for cls, *coords in boxes:
            info["class_counts"][cls] += 1
            # Formato YOLO det: cx cy w h (normalizados 0..1)
            if len(coords) < 4:
                info["invalid_boxes"].append((img.name, "len<5"))
                continue
            cx, cy, w, h = coords[:4]
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                info["invalid_boxes"].append(
                    (img.name, f"cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f}")
                )

    img_stems = {p.stem for p in images}
    for lbl in label_files:
        if lbl.stem not in img_stems:
            info["orphan_labels"].append(lbl.name)

    report[label] = info


def audit_tracking(root: Path, report: dict) -> None:
    """Audita el dataset MOTChallenge (img1/ + det/ + gt/ + seqinfo.ini)."""
    per_seq = []
    total_frames = 0
    total_dets = 0
    total_gts = 0
    class_counts: Counter = Counter()
    broken: list[str] = []
    corrupt: list[str] = []
    missing_det: list[str] = []
    missing_img: list[str] = []

    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for seq in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            img1, det, gt = seq / "img1", seq / "det", seq / "gt"
            info = seq / "seqinfo.ini"

            imgs = sorted(img1.glob("*")) if img1.is_dir() else []
            dets = sorted(det.glob("*.txt")) if det.is_dir() else []
            gts = sorted(gt.glob("*.txt")) if gt.is_dir() else []
            total_frames += len(imgs)
            total_dets += len(dets)
            total_gts += len(gts)

            seqinfo = info.read_text(errors="replace") if info.is_file() else ""
            m = re.search(r"seqLength=(\d+)", seqinfo)
            declared = int(m.group(1)) if m else None
            imwidth = re.search(r"imWidth=(\d+)", seqinfo)
            imheight = re.search(r"imHeight=(\d+)", seqinfo)

            if not dets:
                missing_det.append(f"{split_dir.name}/{seq.name}")
            if not imgs:
                missing_img.append(f"{split_dir.name}/{seq.name}")
            if declared is not None and declared != len(imgs):
                broken.append(
                    f"{split_dir.name}/{seq.name}: seqLength={declared} != "
                    f"{len(imgs)} frames en img1/"
                )

            # det/ en MOTChallenge det: frame,id,x,y,w,h,conf,-1,-1,-1
            for d in dets:
                for line in d.read_text(errors="replace").splitlines():
                    p = line.strip().split(",")
                    if len(p) < 6:
                        continue
                    try:
                        class_counts["cat=%s" % p[6].strip()] += 1
                        float(p[3]), float(p[4]), float(p[5])
                    except ValueError:
                        broken.append(f"{split_dir.name}/{seq.name}/{d.name}: linea malformada")

            if len(imgs) <= 3:  # muestra rápida de integridad
                for img in imgs:
                    if image_size(img) is None:
                        corrupt.append(f"{split_dir.name}/{seq.name}/{img.name}")

            per_seq.append(
                {
                    "seq": f"{split_dir.name}/{seq.name}",
                    "frames": len(imgs),
                    "declared": declared,
                    "det_files": len(dets),
                    "gt_files": len(gts),
                    "size": (
                        f"{imwidth.group(1)}x{imheight.group(1)}"
                        if imwidth and imheight
                        else "?"
                    ),
                }
            )

    report["tracking"] = {
        "sequences": len(per_seq),
        "total_frames": total_frames,
        "det_files": total_dets,
        "gt_files": total_gts,
        "det_categories": dict(class_counts),
        "per_seq": per_seq,
        "inconsistent": broken,
        "corrupt_sample": corrupt,
        "missing_det": missing_det,
        "missing_img1": missing_img,
    }


def find_duplicates(dirs: list[Path]) -> dict:
    """Detecta duplicados exactos por md5 entre varios directorios."""
    by_hash: dict[str, list[str]] = defaultdict(list)
    for d in dirs:
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                by_hash[file_md5(p)].append(str(p))
    dups = {h: paths for h, paths in by_hash.items() if len(paths) > 1}
    return dups


# --------------------------------------------------------------------------- #
# Reporte
# --------------------------------------------------------------------------- #
def print_yolo_report(info: dict) -> None:
    print(f"\n### {info['label']}")
    print(f"  imagenes        : {info['n_images']}  ({info['img_dir']})")
    print(f"  labels .txt     : {info['n_labels']}")
    print(f"  cajas totales   : {info['n_boxes']}")
    if info["boxes_per_image"]:
        bpi = info["boxes_per_image"]
        print(
            f"  cajas/img       : min={min(bpi)} max={max(bpi)} "
            f"media={sum(bpi)/len(bpi):.2f}"
        )
        empty = sum(1 for b in bpi if b == 0)
        if empty:
            print(f"  labels vacios   : {empty}")
    print(f"  clases (id:count): {dict(sorted(info['class_counts'].items()))}")
    if info["missing_labels"]:
        print(f"  !! imagenes SIN label : {len(info['missing_labels'])}")
        print(f"     p.ej. {info['missing_labels'][:5]}")
    if info["orphan_labels"]:
        print(f"  !! labels SIN imagen  : {len(info['orphan_labels'])}")
        print(f"     p.ej. {info['orphan_labels'][:5]}")
    if info["corrupt_images"]:
        print(f"  !! imagenes CORRUPTAS  : {len(info['corrupt_images'])}")
        print(f"     p.ej. {info['corrupt_images'][:5]}")
    if info["invalid_boxes"]:
        print(f"  !! cajas con formato/coord invalida : {len(info['invalid_boxes'])}")
        for name, why in info["invalid_boxes"][:5]:
            print(f"     {name}: {why}")
    if info["sizes"]:
        top = info["sizes"].most_common(5)
        print(f"  resoluciones-top : {top}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="stairs_fallhuman", type=Path)
    args = ap.parse_args()
    root: Path = args.root.resolve()
    print(f"# Auditoria de: {root}\n")

    report: dict = {}

    # ---- 1. fall/ (YOLO con splits) ------------------------------------- #
    fall = root / "fall"
    audit_yolo_dir(fall / "images" / "train", fall / "labels" / "train", "fall/train", report)
    audit_yolo_dir(fall / "images" / "val", fall / "labels" / "val", "fall/val", report)
    audit_yolo_dir(fall / "images" / "test", fall / "labels" / "test", "fall/test", report)

    # ---- 2. stairs/final_stairs/ (YOLO plano, sin splits) ---------------- #
    audit_yolo_dir(
        root / "stairs" / "final_stairs",
        root / "stairs" / "final_stairs" / "labels",
        "stairs/final_stairs",
        report,
    )

    for key in ("fall/train", "fall/val", "fall/test", "stairs/final_stairs"):
        if key in report:
            print_yolo_report(report[key])

    # ---- 3. tracking/ (MOTChallenge) ------------------------------------- #
    print("\n### tracking/ (MOTChallenge)")
    audit_tracking(root / "tracking", report)
    tr = report["tracking"]
    print(f"  secuencias      : {tr['sequences']}")
    print(f"  frames totales  : {tr['total_frames']}")
    print(f"  archivos det/   : {tr['det_files']}")
    print(f"  archivos gt/    : {tr['gt_files']}")
    print(f"  categoria det   : {tr['det_categories']}")
    if tr["inconsistent"]:
        print(f"  !! inconsistencias: {len(tr['inconsistent'])}")
        for b in tr["inconsistent"][:10]:
            print(f"     {b}")
    if tr["corrupt_sample"]:
        print(f"  !! corruptas (muestra): {len(tr['corrupt_sample'])}")
        for c in tr["corrupt_sample"][:10]:
            print(f"     {c}")
    if tr["missing_det"]:
        print(f"  !! secuencias sin det/: {tr['missing_det']}")
    if tr["missing_img1"]:
        print(f"  !! secuencias sin img1/: {tr['missing_img1']}")

    # ---- 4. Duplicados entre datasets ------------------------------------ #
    print("\n### Duplicados exactos (md5)")
    dirs = [
        root / "fall" / "images" / "train",
        root / "fall" / "images" / "val",
        root / "stairs" / "final_stairs",
        root / "tracking" / "train",
        root / "tracking" / "test",
    ]
    dups = find_duplicates(dirs)
    print(f"  grupos de duplicados exactos: {len(dups)}")
    for h, paths in list(dups.items())[:10]:
        print(f"  md5 {h[:10]}:")
        for p in paths:
            print(f"     {p}")

    # ---- 5. Fuga de datos train/val -------------------------------------- #
    print("\n### Posible fuga de datos (mismo md5 en train y val)")
    tr_hash, va_hash = {}, {}
    for name, store in (
        (root / "fall" / "images" / "train", tr_hash),
        (root / "fall" / "images" / "val", va_hash),
    ):
        if name.is_dir():
            for p in name.iterdir():
                if p.suffix.lower() in IMAGE_EXTS:
                    store.setdefault(file_md5(p), []).append(p.name)
    leak = set(tr_hash) & set(va_hash)
    print(f"  imagenes compartidas train<->val: {len(leak)}")
    for h in list(leak)[:10]:
        print(f"  md5 {h[:10]}: train={tr_hash[h]} val={va_hash[h]}")


if __name__ == "__main__":
    main()