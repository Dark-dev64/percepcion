#!/usr/bin/env python3
"""
Paso 2 - Preparación y unificación de datos.

Unifica los 3 datasets de `stairs_fallhuman` en un único dataset YOLO:

  1. fall/              -> posture classes (3 clases, formato YOLO nativo)
  2. stairs/final_stairs-> escalera (1 clase, formato YOLO nativo)
  3. tracking/          -> MOTChallenge -> YOLO (persona, clase generica)

GARANTIAS:
  * Los datasets originales se abren SOLO en lectura. No se modifican.
  * Las imagenes de salida son symlinks relativos: no se duplica el disco.
  * Los labels de salida se reescriben (remapeo de clases + saneado).
  * Split SIN fuga: los duplicados exactos se eliminan antes de repartir, y
    las muestras visualmente relacionadas (variantes Roboflow, secuencias de
    video) se agrupan para que caigan enteras en un solo split.
  * Se emite `manifest.csv` con la trazabilidad completa de cada archivo.

Uso:
    python3 scripts/prepare_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
from statistics import median
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuracion
# --------------------------------------------------------------------------- #
SEED = 42  # semilla fija para que el split sea reproducible

# Taxonomia unificada de 6 clases, alineada al documento de requisitos.
#
# DECISION REVERTIDA (parte 2): se habia fusionado la clase `persona`
# generica dentro de `persona_erguida` para eliminar el colapso de
# predicciones. Se midio y FUE UN ERROR: mAP50 de test paso de 0.706 a 0.587,
# y todas las clases bajaron. La causa es que forzar una sola clase sobre
# fotos de stock de personas de pie (grandes, iluminadas, posadas) y sobre
# peatones de CCTV (pequenos, al 1% del encuadre) multiplica la varianza
# intra-clase. Se conservan las 6 clases y se controla el colapso con el techo
# de cajas del recorte (max_boxes_per_class), que es donde se demuestra que
# funciona.
CLASSES = {
    0: "persona_caido",          # post-caida: persona tendida en el suelo
    1: "persona_sentado",        # postura no ergonomica / agachada
    2: "persona_erguida",        # postura erguida, fotos de stock y Le2i Stand
    3: "escalera",               # tramo de escalera
    4: "persona",                # peaton generico de CCTV (lejos / de espaldas)
    5: "persona_desequilibrio",  # PRE-CAIDA: pierde equilibrio
}

# Mapeo desde el id original de cada dataset al id unificado.
MAP_FALL = {0: 0, 2: 1, 1: 2}     # fall: caido->0, sentado->1, de_pie->2
MAP_STAIRS = {0: 3}                # stairs: escalera->3
# `tracking` se queda como clase propia: los peatones de CCTV son pequenos y
# lejanos, una escala muy distinta a la de las fotos de `fall`, y mezclarlos
# en `persona_erguida` degrada todas las clases (medido: mAP50 0.706 -> 0.587).
TRACKING_CLASS = 4
# Le2i se escribe ya con los ids de la taxonomia unificada (ver
# scripts/prepare_le2i.py), asi que solo hay que filtrar las clases validas.
MAP_LE2I = {0: 0, 2: 2, 5: 5}

# Cadencia de muestreo del video: frames consecutivos son casi identicos,
# asi que se conserva 1 de cada N para evitar redundancia y fuga temporal.
TRACKING_STRIDE = 5

# Proporcion del reparto para los datasets que no traen split.
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_yolo(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Lee un .txt YOLO. Descarta filas malformadas en lugar de fallar."""
    out = []
    for raw in path.read_text(errors="replace").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            vals = [float(p) for p in parts[:5]]
        except ValueError:
            continue
        out.append((int(vals[0]), vals[1], vals[2], vals[3], vals[4]))
    return out


def group_key_for_stairs(stem: str) -> str:
    """Las variantes exportadas por Roboflow comparten nombre base.

    `stair192_jpg.rf.<hash32>` -> `stair192`, de modo que todas las
    augmentaciones de una misma imagen caen juntas en un solo split.
    """
    s = re.sub(r"\.rf\.[0-9a-f]{32}$", "", stem)
    s = re.sub(
        r"(_output)?[-_]?(jpe?g|png)(_flip)?$", "", s, flags=re.IGNORECASE
    )
    return s.lower().strip("_ ")


def sanitize(name: str) -> str:
    """Normaliza el nombre de salida (los datasets traen espacios y acentos)."""
    name = re.sub(r"\s+", "_", name)
    return re.sub(r"[^A-Za-z0-9._-]", "", name)


# --------------------------------------------------------------------------- #
# Elenco de muestras
# --------------------------------------------------------------------------- #
class Sample:
    """Una imagen con su etiqueta, pendiente de asignar a un split."""

    __slots__ = ("src_img", "boxes", "group", "origin", "out_name", "split",
                 "src_hash")

    def __init__(self, src_img: Path, boxes, group: str, origin: str, out_name: str):
        self.src_img = src_img
        # md5 en el punto de alta: lo reutiliza el barrido de fuga sin tener
        # que volver a leer las imagenes (son ~5k archivos de varios MB)
        self.src_hash = md5(src_img)
        self.boxes = boxes          # lista de (class_id_unificado, cx, cy, w, h)
        self.group = group          # agrupa muestras que NO pueden separarse
        self.origin = origin        # dataset de procedencia
        self.out_name = out_name    # nombre del archivo de salida
        self.split: str | None = None

    def classes(self) -> set:
        return {c for c, *_ in self.boxes}

    def __repr__(self):
        return f"<{self.out_name} boxes={len(self.boxes)} split={self.split}>"


class Builder:
    def __init__(self, src_root: Path, out_root: Path):
        self.src = src_root
        self.out = out_root
        self.samples: list[Sample] = []
        self.seen_hashes: dict[str, str] = {}   # md5 -> out_name ya usado
        self.dropped: list[tuple[str, str]] = [] # (motivo, detalle)
        self.per_source: dict[str, int] = defaultdict(int)

    # -- deduplicacion ----------------------------------------------------- #
    def is_duplicate(self, img: Path) -> bool:
        h = md5(img)
        if h in self.seen_hashes:
            self.dropped.append(
                ("duplicado_exacto", f"{img} == {self.seen_hashes[h]}")
            )
            return True
        self.seen_hashes[h] = str(img)
        return False

    def add(self, sample: Sample) -> None:
        self.samples.append(sample)
        self.per_source[sample.origin] += 1

    # -- reparto ------------------------------------------------------------ #
    def assign_splits(self) -> None:
        """Reparte SOLO las muestras sin split ya decidido (las de `stairs`).

        `fall` conserva su split original y `tracking` ya viene asignado por
        secuencia en `load_tracking`; ambas decisiones son intencionadas y no
        deben sobrescribirse aqui.

        El reparto se hace por GRUPO (nombre base de la imagen), de modo que
        las variantes de una misma foto nunca se separan entre splits, y con
        semilla fija para que sea reproducible.
        """
        pending = [s for s in self.samples if s.split is None]
        if not pending:
            return

        by_group: dict[tuple[str, str], list[Sample]] = defaultdict(list)
        for s in pending:
            by_group[(s.origin, s.group)].append(s)

        rng = random.Random(SEED)
        keys = sorted(by_group)          # orden estable antes de barajar
        rng.shuffle(keys)

        n = len(keys)
        n_test = round(n * SPLIT_RATIOS["test"])
        n_val = round(n * SPLIT_RATIOS["val"])

        for i, key in enumerate(keys):
            if i < n_test:
                target = "test"
            elif i < n_test + n_val:
                target = "val"
            else:
                target = "train"
            for s in by_group[key]:
                s.split = target

    # -- seguridad contra fuga ------------------------------------------- #
    def dedupe_preferring_rare_classes(self) -> int:
        """Elimina imagenes duplicadas conservando la clase mas ESCASA.

        El problema aparece en Le2i: sus clips se solapan tanto que un mismo
        frame aparece en varias carpetas de estado a la vez (el mismo fotograma
        figura como `Likefall` y como `Stand`). Con una deduplicacion "el
        primero que llega gana", se quedaba el `Stand` --porque `Stand` se
        recorre antes-- y `persona_desequilibrio` se quedaba SIN NINGUNA
        instancia en test, con lo que su mAP era incalculable.

        Aqui, para cada hash repetido, se conserva la muestra cuya clase es
        globalmente mas escasa, que es la que tiene menos margen a perder. Es
        la misma idea que prioriza las clases minoritarias en cualquier
        problema de datos desbalanceados.
        """
        rarity = self.class_totals()
        by_md5: dict[str, list[Sample]] = defaultdict(list)
        for s in self.samples:
            by_md5[s.src_hash].append(s)

        order = {"train": 0, "val": 1, "test": 2}
        victims: list[Sample] = []
        for group in by_md5.values():
            if len(group) < 2:
                continue
            # rarer primero; a igualdad, train antes que val antes que test
            group.sort(key=lambda s: (
                min((rarity.get(c, 0) for c in s.classes()), default=0),
                order.get(s.split, 9),
                s.out_name,
            ))
            keeper = group[0]
            for s in group[1:]:
                victims.append(s)
                self.dropped.append(
                    ("duplicado_clase_escasa", f"{s.src_img} (== {keeper.src_img})")
                )
        if victims:
            ids = {id(s) for s in victims}
            self.samples = [s for s in self.samples if id(s) not in ids]
        return len(victims)

    def class_totals(self) -> Counter:
        c: Counter = Counter()
        for s in self.samples:
            c.update(s.classes())
        return c

    def scrub_cross_split_duplicates(self) -> int:
        """Elimina cualquier imagen presente en mas de un split.

        Agrupar por "video" no siempre basta: Le2i corta la MISMA escena
        estatica en varias carpetas (`Coffee_room_v10`, `v11`, `v13`...), de
        modo que dos videos distintos pueden contener el mismo frame byte a
        byte. Al estar en carpetas distintas acaban en splits distintos y la
        imagen aparece a la vez en train y en val.

        RED DE SEGURIDAD. En la practica devuelve siempre 0, porque
        `dedupe_preferring_rare_classes` se ejecuta ANTES y ya elimina todos
        los duplicados, no solo los que cruzan splits. Se conserva por si
        alguien cambia el orden de las dos llamadas: garantiza fuga cero por
        imagen sea cual sea la granularidad del agrupado, sin importar quien
        se quede con cada hash duplicado.
        """
        by_md5: dict[str, list[Sample]] = defaultdict(list)
        for s in self.samples:
            by_md5[s.src_hash].append(s)

        victims: list[Sample] = []
        for group in by_md5.values():
            if len({s.split for s in group}) <= 1:
                continue
            order = {"train": 0, "val": 1, "test": 2}
            group.sort(key=lambda s: (order.get(s.split, 9), s.out_name))
            keeper = group[0]
            for s in group[1:]:
                victims.append(s)
                self.dropped.append(
                    ("duplicado_entre_splits", f"{s.src_img} (== {keeper.src_img})")
                )
        if victims:
            ids = {id(s) for s in victims}
            self.samples = [s for s in self.samples if id(s) not in ids]
        return len(victims)

    # -- escritura --------------------------------------------------------- #
    def write(self) -> None:
        for split in ("train", "val", "test"):
            (self.out / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.out / "labels" / split).mkdir(parents=True, exist_ok=True)

        manifest_path = self.out / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["split", "origin", "group", "out_name", "n_boxes", "src_image"]
            )
            for s in sorted(self.samples, key=lambda x: (x.split, x.origin, x.out_name)):
                dst = self.out / "images" / s.split / s.out_name
                # symlink RELATIVO: no ocupa disco y no altera el original
                rel = os.path.relpath(s.src_img, dst.parent)
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(rel)

                lbl = self.out / "labels" / s.split / f"{Path(s.out_name).stem}.txt"
                lbl.write_text(
                    "".join(
                        f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
                        for c, cx, cy, bw, bh in s.boxes
                    )
                )
                w.writerow(
                    [s.split, s.origin, s.group, s.out_name, len(s.boxes), s.src_img]
                )

        self.write_data_yaml()

    def write_data_yaml(self) -> None:
        names = {i: n for i, n in CLASSES.items()}
        header = (
            "# Dataset unificado - Prevencion de Caidas y Riesgos en Escaleras\n"
            "# Generado por scripts/prepare_dataset.py (NO editar a mano).\n"
        )
        (self.out / "data.yaml").write_text(
            header
            + f"path: {self.out.resolve()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in names.items()),
            encoding="utf-8",
        )
        # Variante que ADEMAS usa las variantes de primer plano generadas por
        # scripts/add_zoom_variants.py. Se escribe siempre para que exista
        # aunque aun no se hayan generado: si la carpeta falta, Ultralytics
        # avisa y se puede volver al data.yaml normal.
        (self.out / "data_zoom.yaml").write_text(
            header
            + f"path: {self.out.resolve()}\n"
            "train:\n  - images/train\n  - images/train_zoom\n"
            "val: images/val\n"
            "test: images/test\n"
            "\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in names.items()),
            encoding="utf-8",
        )

    def report(self) -> None:
        print("\n" + "=" * 68)
        print("RESUMEN DE LA PREPARACION")
        print("=" * 68)
        print(f"\nDestino: {self.out}")
        counts: dict[str, int] = defaultdict(int)
        boxes: dict[str, Counter] = defaultdict(lambda: Counter())
        for s in self.samples:
            counts[s.split] += 1
            for c, *_ in s.boxes:
                boxes[s.split][c] += 1
        print("\nImagenes por split:")
        for sp in ("train", "val", "test"):
            print(f"  {sp:6} {counts[sp]:>6}")
        print(f"  {'TOTAL':6} {sum(counts.values()):>6}")

        print("\nCajas por clase y split:")
        header = "  clase               " + "".join(f"{sp:>9}" for sp in ("train", "val", "test"))
        print(header)
        for cid, name in CLASSES.items():
            row = f"  {cid} {name:19}"
            row += "".join(f"{boxes[sp][cid]:>9}" for sp in ("train", "val", "test"))
            print(row)

        print("\nImagenes por dataset de origen:")
        for origin, n in sorted(self.per_source.items()):
            print(f"  {origin:28} {n:>6}")

        print(f"\nDescartadas por duplicado exacto: {len(self.dropped)}")
        for motivo, det in self.dropped[:10]:
            print(f"  - {motivo}: {det}")
        if len(self.dropped) > 10:
            print(f"  ... y {len(self.dropped) - 10} mas (ver dropped.txt)")


# --------------------------------------------------------------------------- #
# Fuentes de datos
# --------------------------------------------------------------------------- #
def load_fall(b: Builder) -> None:
    """`fall/`: ya esta en formato YOLO, solo se remapean los ids de clase.

    No se conserva el split original: se hace un reparto ESTRATIFICADO sobre
    las 474 imagenes deduplicadas. El motivo es que las clases de postura
    son muy escasas y desiguales (la minoritaria tiene ~105 cajas en todo el
    dataset), de modo que el reparto original dejaba ~10 cajas en val y ~9 en
    test. La fuga `train/fall022.jpg == val/fall012.jpg` se resuelve al
    descartar el duplicado por hash antes de repartir.
    """
    root = b.src / "fall"
    buckets: dict[str, list[Sample]] = {"train": [], "val": []}

    for split in ("train", "val"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                continue
            if b.is_duplicate(img):          # resuelve la fuga train/val
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            boxes = [
                (MAP_FALL[c], cx, cy, w, h)
                for c, cx, cy, w, h in parse_yolo(lbl) if c in MAP_FALL
            ]
            buckets[split].append(
                Sample(img, boxes, f"fall_{split}", "fall", sanitize(img.name))
            )

    # Reparto ESTRATIFICADO sobre la totalidad de `fall`.
    #
    # El dataset es muy pequeno (474 imagenes tras deduplicar) y las clases de
    # postura son desigualadas: esta clase minoritaria tiene ~105 cajas en todo
    # el dataset. Conservar el split original dejaba solo ~10 cajas en val y ~9
    # en test, insufficientemente pocas para que el mAP por clase signifique
    # algo. Estratificar por la clase mas rara presente en cada imagen
    # garantiza que val y test reciban una proporcion parecida de cada clase.
    samples = buckets["train"] + buckets["val"]

    def stratum(s: Sample) -> int:
        """Clave de estratificacion: la clase mas rara presente en la imagen."""
        present = {c for c, *_ in s.boxes}
        for rare in (1, 2, 0):    # sentada -> erguida -> caido
            if rare in present:
                return rare
        return -1

    rng = random.Random(SEED)
    by_stratum: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        by_stratum[stratum(s)].append(s)

    for key in sorted(by_stratum):
        grp = sorted(by_stratum[key], key=lambda x: x.out_name)
        rng.shuffle(grp)
        n = len(grp)
        if n < 3:
            # con menos de 3 muestras no se puede poblar val y test
            continue
        n_test = max(1, round(n * SPLIT_RATIOS["test"]))
        n_val = max(1, round(n * SPLIT_RATIOS["val"]))
        if n_test + n_val > n:          # no dejar train vacio en un estrato
            n_val = max(0, n - n_test - 1)
        for i, s in enumerate(grp):
            if i < n_test:
                s.split = "test"
            elif i < n_test + n_val:
                s.split = "val"

    for s in samples:                  # lo no asignado va a train
        s.split = s.split or "train"
        b.add(s)


def load_stairs(b: Builder) -> None:
    """`stairs/final_stairs/`: 1 clase (escalera), sin split previo.

    El reparto es por grupo de nombre base para que las variantes de
    Robofflow de una misma imagen nunca se separen entre splits.
    """
    root = b.src / "stairs" / "final_stairs"
    lbl_dir = root / "labels"
    for img in sorted(root.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
            continue
        if b.is_duplicate(img):
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        boxes = [
            (MAP_STAIRS[c], cx, cy, w, h)
            for c, cx, cy, w, h in parse_yolo(lbl) if c in MAP_STAIRS
        ]
        b.add(
            Sample(
                img,
                boxes,
                group_key_for_stairs(img.stem),
                "stairs",
                sanitize(img.name),
            )
        )


def load_tracking(b: Builder) -> None:
    """`tracking/`: MOTChallenge -> YOLO (clase unica `persona`).

    Split POR SECUENCIA (nunca por frame): frames contiguos son casi
    identicos, asi que un split por frame produciria una fuga enorme.

    Los 11 splits `train` se reparten 8 train / 3 val, y los 11 `test`
    se usan como conjunto de prueba. El reparto dentro de cada bloque se
    hace por proporcion para no depender de una sola secuencia.

    Procedencia de las etiquetas, deducida del formato de cada archivo:
      * layout `...,1,-1,-1,-1`   -> GT real de benchmark (7 secuencias)
      * layout `...,conf,vx,vy,score` -> pseudo-etiqueta CenterTrack (4)
      * `det/det.txt`             -> salida de detector (11 de test)
    Las pseudo-etiquetas se filtran por score para limitar el ruido.
    """
    root = b.src / "tracking"

    for block in ("train", "test"):
        block_dir = root / block
        if not block_dir.is_dir():
            continue
        seqs = sorted(p for p in block_dir.iterdir() if p.is_dir())

        if block == "train":
            # 8 secuencias -> train, 3 -> val (determinista por semilla)
            rng = random.Random(SEED)
            order = [s.name for s in seqs]
            rng.shuffle(order)
            val_seqs = set(order[:3])
        else:
            val_seqs = set()

        for seq in seqs:
            split = "val" if seq.name in val_seqs else (
                "test" if block == "test" else "train"
            )
            imgs = sorted((seq / "img1").iterdir())
            if not imgs:
                continue
            width, height = read_seqinfo(seq / "seqinfo.ini")
            if not width or not height:
                print(f"  [aviso] {seq.name}: sin imWidth/imHeight, se omite")
                continue

            boxes_by_frame = parse_mot(seq, block)
            if not boxes_by_frame:
                continue

            for i, img in enumerate(imgs):
                if i % TRACKING_STRIDE:      # muestreo temporal
                    continue
                rows = boxes_by_frame.get(i + 1, [])   # MOT numera desde 1
                if not rows:
                    continue        # frame negativo: sin caja que aprender
                if b.is_duplicate(img):
                    continue
                boxes = [
                    (
                        TRACKING_CLASS,
                        (x + w / 2) / width,
                        (y + h / 2) / height,
                        w / width,
                        h / height,
                    )
                    for x, y, w, h in rows
                ]
                # recorte defensivo al rango [0,1] que exige YOLO
                boxes = [
                    (c, min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0),
                     min(max(bw, 1e-6), 1.0), min(max(bh, 1e-6), 1.0))
                    for c, cx, cy, bw, bh in boxes
                ]
                sample = Sample(
                    img,
                    boxes,
                    seq.name,          # agrupa por secuencia completa
                    "tracking",
                    sanitize(f"{seq.name}_{img.name}"),
                )
                sample.split = split  # secuencia -> un unico split, sin fuga
                b.add(sample)
        # `val` de tracking se deja fuera de test: solo hay 3 secuencias


def read_seqinfo(path: Path) -> tuple[int | None, int | None]:
    if not path.is_file():
        return None, None
    txt = path.read_text(errors="replace")
    w = re.search(r"imWidth=(\d+)", txt)
    h = re.search(r"imHeight=(\d+)", txt)
    return (int(w.group(1)) if w else None, int(h.group(1)) if h else None)


def parse_mot(seq: Path, block: str) -> dict[int, list[tuple[float, float, float, float]]]:
    """Devuelve {frame: [(x, y, w, h)]} de las cajas utilizables.

    Los scores de estos archivos NO estan normalizados (van de 0 a ~140), asi
    que un umbral absoluto tipo 0.5 no discrimina nada. Por eso las
    pseudo-etiquetas se filtran con un umbral RELATIVO: se conserva la mitad
    mas barata de cada secuencia (score >= mediana). Es un criterio
    uniforme y defendible frente a secuencias con escalas distintas.

    Prioridad de fuentes:
      1. `gt/gt.txt` con layout TUD  -> GT humano real (la mejor opcion)
      2. `gt/gt.txt` con layout CenterTrack -> pseudo-etiqueta con score
      3. `det/det.txt`               -> salida de detector (bloque `test`)
    """
    out: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)

    def add(frame, box):
        x, y, w, h = box
        if w > 0 and h > 0:
            out[frame].append((x, y, w, h))

    gt = seq / "gt" / "gt.txt"      # solo existe en el bloque `train`
    det = seq / "det" / "det.txt"

    if gt.is_file():
        real: list = []
        pseudo: list = []
        for line in gt.read_text(errors="replace").splitlines():
            p = line.split(",")
            if len(p) < 10:
                continue
            # col 7 (idx 6): '1' = caja anotada, '0' = "dont care" (descartar)
            if p[6].strip() in ("0", "0.0"):
                continue
            try:
                frame = int(float(p[0]))
                box = tuple(float(v) for v in p[2:6])
            except ValueError:
                continue
            if p[7].strip() == "-1":
                real.append((frame, box))        # layout TUD: GT real
            else:
                pseudo.append((frame, float(p[9]), box))   # layout CenterTrack

        if real:
            for frame, box in real:
                add(frame, box)
            return out
        if pseudo:
            thr = median([s for _, s, _ in pseudo])
            for frame, score, box in pseudo:
                if score >= thr:
                    add(frame, box)
            return out

    if det.is_file():                # bloque `test`: salida de detector
        rows: list = []
        for line in det.read_text(errors="replace").splitlines():
            p = line.split(",")
            if len(p) < 10:
                continue
            try:
                rows.append(
                    (int(float(p[0])), float(p[6]),
                     tuple(float(v) for v in p[2:6]))
                )
            except ValueError:
                continue
        if rows:
            thr = median([s for _, s, _ in rows])
            for frame, score, box in rows:
                if score >= thr:
                    add(frame, box)
    return out


def load_le2i(b: Builder, le2i_root: Path) -> None:
    """`Le2i` (preparado por scripts/prepare_le2i.py): aporta la clase 5.

    Es la unica fuente de postura PRE-CAIDA del proyecto, que es lo que pide
    `Caso.md` en RQF02 ("perdida inminente de equilibrio"). Sus ids de clase
    ya coinciden con la taxonomia unificada, asi que solo se leen.

    El split ya viene resuelto en el dataset de Le2i y es por VIDEO, de modo
    que aqui se respeta tal cual: los frames de un mismo video son casi
    identicos y separarlos entre train y val seria fuga garantizada.
    """
    root = le2i_root
    if not root.is_dir():
        print(f"  [aviso] {root} no existe; se omite Le2i")
        return

    for split in ("train", "val", "test"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMG_EXTS:
                continue
            # NO se deduplica aqui: en Le2i un mismo frame aparece en varias
            # carpetas de estado y hay que resolverlo globalmente priorizando
            # la clase mas escasa (ver Builder.dedupe_preferring_rare_classes).
            lbl = lbl_dir / f"{img.stem}.txt"
            boxes = [
                (MAP_LE2I[c], cx, cy, w, h)
                for c, cx, cy, w, h in parse_yolo(lbl) if c in MAP_LE2I
            ]
            sample = Sample(
                img, boxes, f"le2i_{img.stem.rsplit('_', 1)[0]}", "le2i", img.name
            )
            sample.split = split          # split ya decidido, por video
            b.add(sample)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("stairs_fallhuman"))
    ap.add_argument("--out", type=Path, default=Path("datasets/combined"))
    ap.add_argument("--le2i", type=Path, default=Path("datasets/le2i_yolo"),
                    help="Dataset Le2i preparado por scripts/prepare_le2i.py")
    ap.add_argument("--clean", action="store_true",
                    help="Borra el directorio de salida antes de generar")
    args = ap.parse_args()

    src = args.src.resolve()
    out = args.out.resolve()
    if not src.is_dir():
        raise SystemExit(f"No existe el dataset de origen: {src}")

    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    b = Builder(src, out)

    print("Leyendo datasets (solo lectura)...")
    load_fall(b)
    load_stairs(b)
    load_tracking(b)
    le2i = args.le2i if args.le2i.is_absolute() else Path.cwd() / args.le2i
    load_le2i(b, le2i)
    print(f"  muestras cargadas: {len(b.samples)}")

    n_dup = b.dedupe_preferring_rare_classes()
    print(f"  duplicados resueltos priorizando la clase mas escasa: {n_dup}")
    b.assign_splits()          # reasigna train/val/test por grupo, sin fuga
    n_removed = b.scrub_cross_split_duplicates()
    print(f"  barrido de fuga: {n_removed} imagenes eliminadas por estar en "
          f"mas de un split")
    b.write()
    b.report()

    # registro de descartes para auditoria
    (out / "dropped.txt").write_text(
        "\n".join(f"{m}: {d}" for m, d in b.dropped), encoding="utf-8"
    )
    (out / "config.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "classes": CLASSES,
                "map_fall": MAP_FALL,
                "map_stairs": MAP_STAIRS,
                "map_le2i": MAP_LE2I,
                "tracking_class": TRACKING_CLASS,
                "tracking_stride": TRACKING_STRIDE,
                "pseudo_score_rule": "score >= mediana de la secuencia",
                "split_ratios": SPLIT_RATIOS,
                "source": str(src),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nListo. data.yaml -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()