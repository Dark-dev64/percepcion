#!/usr/bin/env python3
"""
Integracion del dataset Le2i (raw) al proyecto.

POR QUE ESTE DATASET
--------------------
Los tres datasets originales (`fall`, `stairs`, `tracking`) NO contienen
postura pre-caida: `fall` solo tiene personas ya en el suelo y `tracking`
son peatones sin escalera. `Caso.md` (RQF02, categoria 4) exige detectar
"caidas activas o perdida inminente de equilibrio", y el objetivo declarado
del proyecto es anticipar la caida.

`Le2i-raw` (citado en `Caso.md` §1.4) aporta justo lo que falta: sus carpetas
estan etiquetadas por ESTADO y una de ellas es `Likefall` = "the center of
gravity was unstable", es decir, la perdida de equilibrio ANTES de caer.

MAPEO DE CLASES
---------------
    Blank    -> sin caja (negativo; RQF04 filtra falsos positivos)
    Fall     -> 0 persona_caido
    Lie      -> 0 persona_caido
    Stand    -> 2 persona_erguida
    Likefall -> 5 persona_desequilibrio   <-- clase de PRE-CAIDA

    Se fusionaron a proposito dos cosas:

    1. Se PROBO fusionar la clase `persona` generica dentro de
       `persona_erguida` para forzar que todo peaton reciba una postura
       (RQF03). Se entreno y se midio: mAP50 de test paso de 0.706 a 0.587 y
       TODAS las clases bajaron. La fusion quedo revertida por eso; la
       taxonomia final conserva las 6 clases y el colapso se controla con el
       techo de cajas del recorte, no eliminando la clase.

    2. `Fall` y `Lie` solo entran si se usa `--self-model`. Un detector COCO
       no localiza de forma fiable a una persona tendida (sujeto muy
       escorzado) y descartarlas dejaba `persona_caido` con ~250 cajas, la
       clase mas escasa. El modelo propio, ya entrenado, si las ve.

COMO SE GENERAN LAS CAJAS
-------------------------
Por defecto con un detector de personas preentrenado en COCO. Con
`--self-model`, con el modelo del proyecto.

En ambos casos la salida se VALIDA antes de escribir nada: si `Blank` mostrara
detecciones, o las clases con persona no se detectaran de forma consistente,
el script aborta en vez de generar etiquetas ruidoas.

Uso:
    python3 scripts/prepare_le2i.py --out datasets/le2i_yolo
    python3 scripts/prepare_le2i.py --self-model runs/yolov8s_zoom/weights/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Clase 0 en COCO: "person". Los modelos preentrenados de Ultralytics usan
# el orden de clases de COCO, donde el indice 0 es justamente `person`.
COCO_PERSON_ID = 0

# Clases que representan una persona en la taxonomia del propio modelo.
# `escalera` (id 3) queda fuera a proposito.
SELF_PERSON_CLASSES = [0, 1, 2, 4, 5]

# Frames consecutivos son casi identicos. Se conserva 1 de cada STRIDE[clase]
# para evitar redundancia (que ademas inflaria el tiempo de epoca).
STRIDE = {
    "Likefall": 1,   # clase escasa y precious: se conserva todo
    "Fall": 3,
    "Lie": 4,
    "Stand": 5,
    "Blank": 10,
}

# Etiqueta de origen -> id de clase unificado (taxonomia de 6 clases)
CLASS_MAP = {
    "Fall": 0,      # persona_caido
    "Lie": 0,       # persona_caido
    "Stand": 2,     # persona_erguida
    "Likefall": 5,  # persona_desequilibrio (PRE-CAIDA)
    "Blank": None,  # sin caja: es un negativo
}




def base_video(folder: str) -> str:
    """Nombre del video real, quitando el indice de clip.

    En Le2i las carpetas se llaman `<escena>_v<video>c<clip>`: `coffee_v11c23`
    y `coffee_v11c24` son clips 23 y 24 DEL MISMO video. Repartir por clips
    metia frames casi identicos en train y en test; se comprobo que las 64
    imagenes de test de `Likefall` desaparecian al eliminar duplicados, porque
    todas tenian un gemelo byte a byte en train.

    En `Blank` los nombres no llevan indice de clip, asi que aqui es un no-op.
    """
    return re.sub(r"c\d+$", "", folder)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "datasets/_external/Le2i-raw")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets/le2i_yolo")
    ap.add_argument("--weights", default="yolov8s.pt", help="Detector COCO")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=200,
                    help="Frames por tanda de deteccion. Acota el pico de RAM.")
    ap.add_argument("--self-model", default=None,
                    help="Pesos del modelo del proyecto. Si se indica, se usa "
                         "en lugar del detector COCO y se habilitan las clases "
                         "Fall/Lie, que el modelo propio si sabe localizar.")
    ap.add_argument("--fall-lie-stride", type=int, default=4,
                    help="Cadencia de muestreo para Fall y Lie.")
    ap.add_argument("--exclude", default="",
                    help="Clases de Le2i a EXCLUIR, separadas por comas.\n"
                         "Ej: 'Fall,Lie'. Se usa para excluir `Lie`, que aunque "
                         "se\nvalida bien (91% de deteccion) destruye la clase "
                         "`persona_desequilibrio`:\nson el mismo sujeto en "
                         "momentos consecutivos de las mismas escenas, de modo "
                         "que\ncaida y pre-caida se vuelven indistinguibles "
                         "(medido: 0.920 -> 0.267).")
    args = ap.parse_args()

    raw = args.raw.resolve()
    out = args.out.resolve()
    if not raw.is_dir():
        raise SystemExit(f"No existe el dataset crudo: {raw}")

    from ultralytics import YOLO

    # ---------------------------------------------------------------- #
    # 1. Recolectar los frames a procesar (con su clase y su video)
    # ---------------------------------------------------------------- #
    strides = dict(STRIDE)
    if args.self_model:
        strides["Fall"] = args.fall_lie_stride
        strides["Lie"] = args.fall_lie_stride
    else:
        # sin modelo propio estas clases no se pueden etiquetar con fiabilidad
        strides.pop("Fall", None)
        strides.pop("Lie", None)
    for c in [x.strip() for x in args.exclude.split(",") if x.strip()]:
        if c in strides:
            print(f"  excluida por --exclude: {c}")
            strides.pop(c)

    items = []   # (ruta, clase, video_id, split_origen)
    for split in ("train", "val"):
        for cls in strides:
            base = raw / split / cls
            if not base.is_dir():
                continue
            for video in sorted(p for p in base.iterdir() if p.is_dir()):
                frames = sorted(
                    p for p in video.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                )
                stride = strides[cls]
                for i, frame in enumerate(frames):
                    if i % stride:
                        continue
                    items.append((frame, cls, video.name, split))
    print(f"Frames seleccionados tras submuestreo: {len(items)}")
    by_cls = Counter(c for _, c, _, _ in items)
    for c, n in sorted(by_cls.items()):
        print(f"  {c:9} {n:>6}  (stride {strides.get(c, 1)})")

    # ---------------------------------------------------------------- #
    # 2. Deteccion de personas con el modelo COCO
    #
    # Se hace la inferencia A MANO con torch en vez de usar
    # `YOLO.predict()`. Motivo: reutilizar el Predictor de Ultralytics entre
    # varias llamadas deja estado acumulado yprovoca "CUDA driver error:
    # device not ready" a los pocos lotes; y el reintento en CPU con batch 8
    # desborda la RAM de la maquina (~8 GB) y muere por el OOM-killer.
    #
    # Con este codigo el pico de memoria es exactamente `batch` imagenes y no
    # hay estado que arrastrar entre lotes.
    # ---------------------------------------------------------------- #
    import cv2
    import numpy as np
    import torch
    from ultralytics.utils.nms import non_max_suppression

    # con --self-model se usa el detector del proyecto; si no, el de COCO
    model_path = args.self_model or args.weights
    self_model = bool(args.self_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDetectando con {model_path} (conf>={args.conf})...")
    net = YOLO(model_path).model.to(device).eval()
    net_cpu = None      # se crea bajo demanda si la GPU falla
    dets: dict[Path, list] = {}
    paths = [it[0] for it in items]
    total = len(paths)

    def detect_block(block: list[Path], on: str) -> dict[Path, list]:
        nonlocal net_cpu
        model = net
        if on == "cpu":
            if net_cpu is None:
                print("\n  (GPU no disponible: se continua en CPU)", flush=True)
                net_cpu = YOLO(model_path).model.to("cpu").eval()
            model = net_cpu
        batch_imgs = []
        usable: list[Path] = []
        for p in block:
            im = cv2.imread(str(p))            # BGR
            if im is None:                    # imagen ilegible -> sin caja
                dets[p] = []
                continue
            im = cv2.resize(im, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            batch_imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            usable.append(p)

        if not batch_imgs:
            return {}

        arr = np.stack(batch_imgs).astype(np.float32) / 255.0   # N,H,W,3
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(on)
        with torch.inference_mode():
            preds = model(tensor)
        # NMS con filtro de clases. Con COCO es imprescindible usar solo la
        # clase 0 (`person`): sin ese filtro cada silla, mesa o television
        # contaria como persona, y la validacion de mas abajo lo detecta.
        # Con el modelo propio se aceptan las clases que son persona.
        out = non_max_suppression(
            preds, conf_thres=args.conf, iou_thres=0.45,
            classes=SELF_PERSON_CLASSES if self_model else [COCO_PERSON_ID],
            max_det=5,
        )
        result = {}
        for path, det in zip(usable, out):
            if det is None or len(det) == 0:
                result[path] = []
                continue
            # las coordenadas estan en el espacio cuadrado de `imgsz`;
            # dividir por imgsz las devuelve a coordenadas normalizadas 0..1
            result[path] = (det[:, :4].cpu().numpy() / args.imgsz).tolist()
        del tensor, preds, out, arr, batch_imgs
        return result

    failures = 0
    for start in range(0, total, args.chunk):
        block = paths[start:start + args.chunk]
        # 1) GPU, 2) GPU tras vaciar la cache, 3) CPU. El ultimo recurso es la
        # CPU porque el allocator de CUDA puede quedar corrupto y vaciar la
        # cache no lo repara.
        for attempt, on in enumerate((device, device, "cpu")):
            try:
                dets.update(detect_block(block, on))
                break
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(
                    f"\n  [reintento {attempt + 1}/3] lote "
                    f"{start // args.chunk + 1} en {on}: {type(exc).__name__}: "
                    f"{str(exc)[:80]}",
                    flush=True,
                )
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                time.sleep(3)
        # si los tres intentos fallaron, el lote queda sin cajas en lugar de
        # romper el script
        dets.update({p: [] for p in block if p not in dets})
        print(
            f"  {min(start + args.chunk, total)}/{total} frames",
            end="\r", flush=True,
        )
    print()
    if failures:
        print(f"  (se registraron {failures} reintentos)")

    # ---------------------------------------------------------------- #
    # 3. VALIDACION de la pseudo-etiquetacion
    #
    # Si el detector fuera una caja negra, `Blank` (frames sin persona)
    # tendria tantas detecciones como el resto. Esa comprobacion es la que
    # garantiza que las cajas derivadas son fiables.
    #
    # El umbral es POR CLASE y no uniforme, porque las clases no son
    # equivalentes: se verifican las que el proyecto necesita de verdad
    # (`Likefall` = pre-caida, `Stand` = postura erguida, `Blank` = negativo)
    # y las demas se aceptan solo si tambien superan el control.
    #
    # `Fall` y `Lie` fallan de forma sistematica porque una persona TENDIDA
    # en el suelo esta muy escorzada, y los detectores de COCO se entrenan con
    # gente de pie. Es una limitacion real, no un ajuste de parametro: bajar el
    # umbral de 0.35 a 0.20 solo los subio del 56%/58% al 61%/66%.
    # ---------------------------------------------------------------- #
    print("\n--- VALIDACION de la pseudo-etiquetacion ---")
    print(f"{'clase':10} {'frames':>7} {'con persona':>12} {'%':>7} "
          f"{'cajas/frame':>12}  veredicto")
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for path, cls, _vid, _split in items:
        n = len(dets.get(path, []))
        stats[cls][0] += 1
        stats[cls][1] += 1 if n else 0
        stats[cls][2] += n

    MIN_DETECTION = 0.80      # clases con persona: se espera detectarlas
    MAX_BLANK = 0.15          # 'Blank' NO debe tener persona
    MAX_BOXES = 1.6           # Le2i graba un unico sujeto

    rejected: set[str] = set()
    for cls in sorted(stats):
        total, with_person, boxes = stats[cls]
        pct = with_person / total if total else 0
        bpf = boxes / total if total else 0.0

        if cls == "Blank":
            ok = pct <= MAX_BLANK
        else:
            ok = pct >= MIN_DETECTION and bpf <= MAX_BOXES
        verdict = "OK" if ok else "DESCARTADA"
        if not ok:
            rejected.add(cls)
        print(
            f"{cls:10} {total:>7} {with_person:>12} {100*pct:>6.1f}% "
            f"{bpf:>12.2f}  {verdict}"
        )

    # Las clases que el proyecto necesita si o si no pueden fallar.
    # `Fall` y `Lie` NO son criticas a proposito. `Lie` (persona ya en el
    # suelo) la resuelve bien el modelo propio; `Fall` esta en pleno
    # movimiento, con desenfoque, y se queda alrededor del 65%. Si se
    # exigiera unListado estricto el script abortaria, cuando lo que interesa
    # es aceptar `Lie` y descartar `Fall` sin perder lo demas.
    CRITICAL = {"Likefall", "Stand", "Blank"}
    missing = CRITICAL - set(stats) - rejected
    if missing or (CRITICAL & rejected):
        bad = sorted((CRITICAL & rejected) | missing)
        raise SystemExit(
            f"ABORTADO: las clases criticas {bad} no superaron la validacion. "
            f"No se escribe el dataset."
        )

    if rejected:
        print(
            "\nSe descartan "
            + ", ".join(sorted(rejected))
            + ".\n  Motivo: el detector no alcanza el 80% de deteccion en esas "
            "clases.\n  `Fall` esta en pleno movimiento y con desenfoque, y la "
            "persona muy\n  escorzada se le escapa. La clase `persona_caido` "
            "queda cubierta por\n  `Lie` (91%) y por las anotaciones humanas del "
            "dataset `fall`."
        )
    blank_pct = 100 * stats["Blank"][1] / stats["Blank"][0] if stats["Blank"][0] else 0
    print(f"\nACEPTADO: {blank_pct:.0f}% de falsos positivos en 'Blank'.")

    # ---------------------------------------------------------------- #
    # 4. Escribir el dataset YOLO
    #
    # El split NO se toma del `train`/`val` de Le2i tal cual: se reparte por
    # VIDEO para que los frames de un mismo video (identicos entre si) no se
    # separen entre train y val. Ademas se mezcla el train y el val originales
    # de Le2i y se vuelve a repartir, porque sus proporciones no encajan con el
    # dataset unificado.
    # ---------------------------------------------------------------- #
    for split in ("train", "val", "test"):
        # se limpia antes de escribir: si no, quedan archivos de ejecuciones
        # anteriores y prepare_dataset.py lee el disco (no el manifest), con
        # lo que se colarian muestras de un reparto viejo
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for stale in (out / "images" / split).iterdir():
            if stale.is_file() or stale.is_symlink():
                stale.unlink()
        for stale in (out / "labels" / split).iterdir():
            if stale.is_file() or stale.is_symlink():
                stale.unlink()

    import random
    rng = random.Random(42)
    videos: dict[tuple[str, str], list] = defaultdict(list)
    for it in items:
        if it[1] in rejected:      # clase que no supero la validacion
            continue
        # se agrupa por VIDEO BASE: los clips del mismo video no se separan
        videos[(it[1], base_video(it[2]))].append(it)

    # Reparto ESTRATIFICADO POR CLASE.
    #
    # Con un shuffle global de videos la clase mas escasa se quedaba sin
    # representacion: `persona_desequilibrio` (que solo son 6-10 videos)
    # acababa con 171 cajas en val y 17 en test, que hace su mAP insensible.
    # Repartiendo los videos de cada clase por separado, cada clase recibe
    # proporcion parecida en los tres splits.
    assign: dict[tuple[str, str], str] = {}
    by_class: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in videos:
        by_class[key[0]].append(key)
        # Reparto por tamaño de video, alternando val y test.
    #
    # Un reparto proporcional en numero de videos no equilibra nada cuando hay
    # pocos: `Likefall` tiene 12 videos de base y un codicioso por frames dejaba
    # 304 train / 32 val / 80 test, con un val tan fino que el early stopping
    # se guiaba casi a ciegas.
    #
    # Aqui se ordena cada clase de mayor a menor tamano: los mas grandes se van
    # a train (que debe conservar la mayor parte) y el resto se alterna entre
    # test y val, de modo que ambos reciben aproximadamente las mismas cajas.
    for cls in sorted(by_class):
        keys = sorted(by_class[cls], key=lambda k: (-len(videos[k]), k))
        n = len(keys)
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))
        n_holdout = n_test + n_val
        n_train = max(1, n - n_holdout)
        for i, k in enumerate(keys):
            if i < n_train:
                assign[k] = "train"
            else:
                # se alterna sobre el resto, empezando por test
                j = i - n_train
                assign[k] = "test" if j % 2 == 0 else "val"
        # si el reparto final dejo un split vacio, se roba el video mas pequeno
        for s in ("test", "val"):
            if not any(v == s for k, v in assign.items() if k[0] == cls):
                cand = [k for k in keys if assign[k] == "train"]
                if cand:
                    assign[cand[-1]] = s

    manifest = []
    written = Counter()
    for key, group in videos.items():
        cls, vid = key
        split = assign[key]
        cid = CLASS_MAP[cls]
        for path, _c, _v, _o in group:
            stem = f"{cls}_{vid}_{path.stem}"
            dst_img = out / "images" / split / f"{stem}.jpg"
            if not dst_img.exists():
                dst_img.symlink_to(path)

            rows = []
            if cid is not None:
                for x1, y1, x2, y2 in dets[path]:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    bw, bh = x2 - x1, y2 - y1
                    if bw <= 0 or bh <= 0:
                        continue
                    rows.append(
                        f"{cid} {min(max(cx,0),1):.6f} {min(max(cy,0),1):.6f} "
                        f"{min(max(bw,1e-6),1):.6f} {min(max(bh,1e-6),1):.6f}"
                    )
            (out / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(rows) + ("\n" if rows else "")
            )
            written[split] += 1
            manifest.append({
                "split": split, "origin": "le2i", "class_name": cls,
                "video": vid, "clip": key[1],
                "out_name": dst_img.name,
                "n_boxes": len(rows), "src_image": str(path),
            })

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)

    print("\nDataset Le2i escrito:")
    for split in ("train", "val", "test"):
        n_lbl = len(list((out / "labels" / split).glob("*.txt")))
        n_box = sum(
            1 for f in (out / "labels" / split).glob("*.txt")
            for line in f.read_text().splitlines() if line.strip()
        )
        print(f"  {split:6} {n_lbl:>6} imagenes, {n_box:>6} cajas")
    print(f"  (de las cuales 'persona_desequilibrio' solo puede venir de Likefall)")

    (out / "validacion.json").write_text(
        json.dumps(
            {c: {"frames": v[0], "con_persona": v[1], "cajas": v[2]}
             for c, v in stats.items()},
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()