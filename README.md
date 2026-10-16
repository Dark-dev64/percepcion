# Prevención de Caídas y Riesgos en Escaleras

Detector de riesgo de caídas en escaleras con **YOLOv8** (Ultralytics),
**MediaPipe Pose** y un segundo detector COCO para obstáculos, superresolución
4× con **Real-ESRGAN** (preprocesamiento opcional, `--esgran`), y una
plataforma web local para probarlo con imagen, vídeo o cámara en vivo.

El sistema está diseñado contra los requisitos del documento `Caso.md`
(IEEE 830 / ISO 25010): 8 funcionales (RQF01–RQF08) y 28 no funcionales
(RQNF01–RQNF26).

---

## Estado rápido

| Métrica | val | test |
|---|---|---|
| Precision | 0.7285 | 0.6516 |
| Recall | 0.6661 | 0.6807 |
| **mAP@0.5** | **0.7178** | **0.7025** |
| mAP@0.5:0.95 | 0.4774 | 0.4853 |

Modelo entregado: `runs/yolov8s_zoom/weights/best.pt` (yolov8s, 6 clases).

`RQNF10` pide mAP@0.5 > 75%: queda en **0.7178 val** y **0.7025 test**, por
debajo del umbral global. La clase que lo frena es `persona`, cuyo *ground
truth* de test es pseudo-etiquetado de detector y no anotación humana; sobre
las clases con etiqueta humana el test sube a **0.7536** (≥75%). Las clases de
riesgo clave superan ampliamente el umbral:

| Clase | mAP@0.5 val | mAP@0.5 test |
|---|---|---|
| `persona_caido` | 0.7962 | 0.7575 |
| `persona_desequilibrio` | 0.8565 | 0.8630 |
| `escalera` | 0.8304 | 0.7437 |

### Clases detectadas

| id | Clase | Riesgo | Referencia |
|----|-------|--------|------------|
| 0 | `persona_caido` | ALTO | RQF02 (4) · RQNF05 |
| 1 | `persona_sentado` | MEDIO | RQF03 |
| 2 | `persona_erguida` | BAJO | RQF03 |
| 3 | `escalera` | contexto | — |
| 4 | `persona` | neutro | entrada a MediaPipe |
| 5 | `persona_desequilibrio` | ALTO | RQF02 (4) — **pre-caída** |

Además, el motor de riesgo deriva tres categorías que **no son cajas** sino
condiciones de postura, con MediaPipe Pose y un detector COCO:

| Señal | Riesgo | Referencia |
|-------|--------|------------|
| `obstaculo_escalon` | ALTO | RQF02 (1) |
| `sin_pasamanos` | MEDIO | RQF02 (2) |
| `distraccion` (móvil / lectura) | MEDIO | RQF02 (3) |
| `tambaleo` | ALTO | RQF02 (4) |

---

## Puesta en marcha

```bash
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
```

> **Atención a la GPU.** La RTX 5050 es Blackwell (`sm_120`), que solo
> funciona con wheels de PyTorch compilados para CUDA 12.8:
>
> ```bash
> ./venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 \
>     torch==2.11.0+cu128 torchvision==0.26.0+cu128
> ./venv/bin/pip install -r requirements.txt
> ```
>
> Con las builds por defecto, torch cae a `sm < 120` y falla al ejecutar.
>
> **Funciona también sin GPU:** `setup.sh` detecta la ausencia de NVIDIA e
> instala PyTorch CPU automáticamente. El servidor funciona con
> `--device cpu` o `--device auto` (detecta CUDA si existe).

### Regenerar todo desde cero

```bash
./venv/bin/python scripts/prepare_le2i.py --exclude Fall,Lie
./venv/bin/python scripts/prepare_dataset.py --clean
./venv/bin/python scripts/add_zoom_variants.py --split train
./venv/bin/python scripts/train.py
./venv/bin/python scripts/evaluate.py --weights runs/yolov8s_zoom/weights/best.pt
```

### Plataforma web

```bash
./venv/bin/python app/server.py --host 0.0.0.0 --port 5000
```

Modos: **imagen**, **vídeo**, **cámara en vivo** y **clases/requisitos**.

Para exponerla en la red local (por ejemplo desde el móvil) hace falta
`--host 0.0.0.0`. **No hay cifrado**: úsala solo en redes de confianza.

Autenticación (`RQNF19`): se activa definiendo `PANEL_TOKEN`; sin esa variable
el panel arranca abierto y lo indica en `/api/health`.

```bash
PANEL_TOKEN="un-secreto-largo-y-aleatorio" ./venv/bin/python app/server.py
```

---

## Datasets utilizados y créditos

Todo lo que se ha descargado y usado, con su origen y licencia de referencia.

### Datasets del enunciado (facilitados)

| Dataset | Uso | Origen |
|---------|-----|--------|
| **Fall Detection Dataset** | clase `persona_caido` y `persona_sentado`; anota cajas | <https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset> |
| **Stairs** | clase `escalera`; 959 imágenes con caja de escalera | <https://www.kaggle.com/datasets/samuelayman/stairs> |
| **MOT15 / MOT17 Challenge** | clase `persona`: peatones de CCTV, lejanos y de espaldas | <https://www.kaggle.com/datasets/mdraselsarker/mot15-challenge-dataset> |

### Datasets externos añadidos (citados en `Caso.md` §1.4)

| Dataset | Uso | Origen |
|---------|-----|--------|
| **Le2i Fall Detection (raw)** | **clase `persona_desequilibrio`** (pre-caída), tomada de su carpeta `Likefall`; y refuerzo de `persona_erguida` | <https://github.com/YifeiYang210/Fall_Detection_dataset> |

> `Le2i` es el que aporta la clase que el enunciado exige y que ningún otro
> dataset tenía: personas con el **centro de gravedad inestable antes de caer**.
> Se descargó de los enlaces de Google Drive indicados en el README de ese
> repositorio (`Le2i-raw`, 310 MB).

### Datasets además citados en `Caso.md` §1.4

Constan como referencias del documento y quedaron **descartados** por no ser
accesibles sin registro o por no aportar cajas:

- **Le2i Fall Detection Dataset** (versión procesada, segmentación de cuerpo
  humano): el archivo descargado es un RAR y no hay `unrar` en el entorno.
- **SisFall**, **FDD** (NTU) y **URFD**: requieren solicitud o registro previo.

### Modelos preentrenados

| Modelo | Uso | Origen |
|--------|-----|--------|
| `yolov8s.pt` / `yolov8n.pt` (COCO) | pseudo-etiquetado inicial y **detección de obstáculos** (RQF02 1) | Ultralytics, AGPL-3.0 |
| MediaPipe Pose (`pose_landmark_lite`) | postura, sujeción del pasamanos, distracción, tambaleo | Google MediaPipe, Apache-2.0 |

### Créditos y atribución

- **Ultralytics YOLO** — licence AGPL-3.0 (<https://www.ultralytics.com/license>).
  Los pesos de `yolov8n/s` se distribuyen bajo los términos de Ultralytics y
  de sus fuentes (COCO).
- **MediaPipe** — Apache-2.0 (<https://github.com/google-ai-edge/mediapipe>).
- **Los datasets pertenecen a sus autores** y se usan aquí solo con fines de
  investigación y evaluación académica. Se cita cada origen porque es lo que
  pide el enunciado; esta redistribución **no** intenta sublicenciarlos. Para
  cualquier uso que no sea de investigación hay que revisar la licencia de
  cada dataset en su ficha original (Kaggle / GitHub).
- Proyecto en el contexto de la **Universidad Peruana de Aplicaciones y
  Ciencias Skylight (UPAO)**.

---

## Arquitectura

```
  frame (cámara / imagen / vídeo)
        │
        ▼
  ┌───────────────────────────────────────────────┐
  │  YOLOv8 propio  → persona / escalera          │  RQF02
  │  YOLOv8 COCO    → obstáculos sobre el escalón │  RQF02 (1)
  │  MediaPipe Pose → muñeca, cabeza, tronco      │  RQF02 (2)(3)
  └───────────────────────────────────────────────┘
        │
        ▼
  seguimiento con IoU + tolerancia a oclusión 1 s    RQF05
        │
        ▼
  filtrado CONF_PREFILTER (0.75) / CONF_CONFIRM (0.85)  RQF04
        │
        ▼
  UMBRAL DE ALERTA: CONF_ALERTA = 0.35   (desviación: no 0.75/0.85)
        │
        ▼
  persistencia según severidad — ALTO 1.2 s, MEDIO 0.6 s  RQF06
        │
        ▼
  alerta con metadatos + evidencia anonimizada     RQF07 · RQNF14
        │
        ▼
  panel web + registro auditable                   RQF08 · RQNF16
```

Todo el análisis se ejecuta en **un único hilo trabajador persistente**: Flask
atiende cada petición en un hilo distinto y en un hilo recién creado hay que
reenlazar el contexto CUDA, lo que costaba 85 ms por frame frente a 37 ms.

---

## Superresolución 4× con Real-ESRGAN (RQNF03-extra)

**Real-ESRGAN** es una técnica de **superresolución**: toma una imagen o video de
baja resolución y la reproduce a **4×** mayor resolución **reconstruyendo
detalles reales**, no estirando píxeles. Se entrena con una **GAN** (*Generative
Adversarial Network*): un *generador* produce la imagen ampliada y un
*discriminador* actúa como crítico que exige que el resultado "parezca una foto
verdadera", de ahí que gane nitidez de verdad (rango dinámico 0–255) y no el
grisáceo que producía la antigua normalización a [-1,1].

**Analogía (para no especialistas):** como si se le diera a un artista un boceto
borroso y se le pidiera que lo redibuje 4 veces más grande y nítido. El artista no
copia píxel por píxel; **averigua qué había** tras el borrón (texturas, sombras,
bordes) y los rellena. El discriminador es el "crítico" que exige realismo.

**Pipeline (`app/real_esrgan.py`):**

1. El frame bajo se reduce manteniendo la relación de aspecto: **112 px** en cámara
   en vivo (~90 ms) o **320 px** en imagen/vídeo (~463 ms). Antes se aplastaba a
   224×224 y distorsionaba 16:9 (Laplacian 56 → 103 al preservar aspecto).
2. El modelo (red residual ESRGAN) genera la salida 4× (112→448, 320→1280).
3. La salida se cierra a **1280 px** (`_MAX_OUTPUT`): una foto 4K a 4× → 16K
   agota VRAM y el codificador JPEG.
4. *Unsharp* leve restituye el contraste local. La SR **precede** a la detección:
   YOLOv8 trabaja sobre bordes reconstruidos (menos falsos negativos en
   obstáculos pequeños, RQF02) y el frontend dibuja la SR sobre el canvas en
   base64 (cámara en vivo) o hex (upload).

**¿Por qué no onnxruntime ni PyTorch JIT?** El ONNX público de Real-ESRGAN lleva el
nodo `Resize` con `scales` en el slot de `roi` (bug de exportación, opset ≥13),
lo que cuelga onnxruntime y `cv2.dnn`. Además `realesrgan`/`basicsr` importa
`torchvision.transforms.functional_tensor`, borrado en torchvision 0.26. Por eso
se convierte ONNX → **PyTorch FX** con `onnx2torch` (cacheado a nivel de clase) e
infiere en CUDA; el uso de `torch.jit` está descartado por `sm_120`.

**Alternativas de velocidad (modo cámara):** el flag `#fast`/async deja el live a
~5 fps. Si la fluidez importa más que la nitidez, se puede forzar `fast=True`
(112 px, ~90 ms) desde el frontend; por defecto se prioriza calidad.

**Créditos y atribución:**

- *Paper*: **Real-ESRGAN: Real-World Super-Resolution via Generative Adversarial
  Networks**, X. Wang et al., 2022.
- *Implementación*: **xinntao/Real-ESRGAN** — licencia **AGPL-3.0**.
- *Modelo ONNX*: `real_esrgan_x4.onnx` del espacio HuggingFace
  **SceneWorks/real-esrgan-onnx**, descargado por `app/build_esrgan.py`.

El flag **`--esgran`** (o `--esrgan`, aceptado por `argparse`) activa el paso; sin
él el pipeline omite la SR y trabaja a resolución nativa.

---

## Cumplimiento de requisitos

| Requisito | Descripción | Estado |
|-----------|-------------|--------|
| RQF01 | Captura continua de vídeo | **CUMPLE** — modo cámara |
| RQF02 | 4 categorías de riesgo | **CUMPLE** — 3 por pose + 1 por detector COCO |
| RQF03 | Discriminación seguro / riesgo | **CUMPLE** — veredicto por escena |
| RQF04 | Umbrales 75 % / 85 % | **PARCIAL** — implementados, pero la puerta de alerta usa 0.35 (ver nota) |
| RQF05 | Seguimiento con oclusión 1 s | **DESVIACIÓN** — IoU propio en vez de ByteTrack |
| RQF06 | Persistencia > 3 s | **DESVIACIÓN** — 1.2 s en rojo y 0.6 s en ámbar, por petición del usuario |
| RQF07 | Alertas con evidencia | **CUMPLE** — metadatos + fotograma anotado con las cajas |
| RQF08 | Panel de supervisión | **CUMPLE** — app web |
| RQNF01 | ≥ 30 FPS | **PARCIAL** — 26.5 fps con GPU (pose_every=2); en CPU más lento (~7 fps) |
| RQNF02 | Latencia < 500 ms | **CUMPLE** — ~42 ms por ciclo en cámara |
| RQNF03 | VRAM ≥ 6 GB | **CUMPLE** (GPU) — RTX 5050, 8.1 GB; en CPU se requiere RAM ≥ 6 GB |
| RQNF10 | mAP@0.5 > 75 % | **PARCIAL** — val 0.718, test 0.703 (por debajo del 75% global; clases de riesgo superan el umbral)
| RQNF12 | Resiliencia a iluminación | **CUMPLE** — augmentations HSV reforzadas |
| RQNF14 | Protección de datos personales | **CUMPLE** — rostros pixelados al guardar |
| RQNF16 | Registro auditable | **CUMPLE** — `runs/alerts/alertas.jsonl` |
| RQNF19 | Autenticación de usuario | **CUMPLE** — `PANEL_TOKEN` |
| RQNF21 | Configuración sin recompilar | **CUMPLE** — `configs/train.yaml`, flags CLI |
| RQNF22 | Verificación con vídeos | **CUMPLE** — modo vídeo |
| RQNF09 | Estado auto-descriptivo | **CUMPLE** — OPERATIVO / DEGRADADO / NO OPERATIVO real |
| RQNF13 | Reconexión automática de señal | **CUMPLE** — 5 reintentos con espera creciente |
| RQNF15 | Integridad de alertas | **CUMPLE** — imagen y registro se borran juntos |
| RQNF18 | No repudio de evidencia | **CUMPLE** — imagen pixelizada + línea de auditoría |
| RQNF20 | Arquitectura modular | **CUMPLE** — `risk_engine.py` separado de `server.py` |
| RQNF23 | Registro de diagnóstico | **CUMPLE** — detalle en `/api/health` |
| RQNF25 | Escalabilidad multicámara | **CUMPLE** — parámetro `?cam=` |
| RQNF26 | Estado no operativo | **CUMPLE** — el estado refleja fallos reales |

### Desviaciones conscientes y lo que no está cubierto

Tres requisitos se cumplen **de forma distinta** a lo que pide el enunciado, y
no por descuido:

| Requisito | Lo que pide `Caso.md` | Lo que hace el sistema | Por qué |
|---|---|---|---|
| **RQF04** | puerta de alerta al 75 % / 85 % | la puerta de alerta usa **0.35** | Medida la confianza real del detector sobre el test set: `persona_caido` tiene mediana **0.39** y máximo **0.79**. Con 0.75 no pasaba ninguna detección y **no se disparaba jamás ninguna alerta**. El 85 % de confirmación sigue sin activarse por lo mismo. |
| **RQF05** | tracker **IoU propio** (~30 líneas) | seguimiento IoU propio | Coincide; el commentario obsoleto en `risk_engine.py:12` que mencionaba "ByteTrack" fue corregido. |
| **RQF06** | persistencia **> 3 s** | **1.2 s** en rojo y **0.6 s** en ámbar | Una caída de pie al suelo dura ~1 s. Con 3 s el sistema solo alertaba cuando la persona llevaba rato en el suelo, que es justo lo que el requisito quiere evitar. Cambio pedido expresamente durante el desarrollo. |

Además:

- **RQNF10** (mAP@0.5 > 75%) queda en **0.703 en test (0.718 en val)**: 0.047 y 0.032 por debajo respectivamente.
  La clase que lo frena es `persona`, cuyo *ground truth* de test es
  pseudo-etiquetado de detector. Sobre clases con etiqueta humana el test sube a **0.754** (≥75%).
- **RQNF11** (disponibilidad en horario de campus) y **RQNF27** (restricción de
  actuación física) dependen del despliegue físico, no del código.
- **RQF02 (2)(3)** —no sujetar el pasamanos, distracción— funcionan e están
  implementados, pero **no se han medido**: ningún dataset anota esas
  conductas. Se resuelven con keypoints de MediaPipe, como prescribe el
  propio enunciado.
- Los umbrales de postura se calibraron con el **motor completo**
  (MediaPipe en modo *tracking*). Medidos con un script suelto dan 57 % de
  falsas alarmas en vez de 8 %: el seguimiento temporal es lo que hace
  estable la pose.

- **RQNF04** (flujo continuo de cámara fija) y **RQNF19** en su parte de
  permisos por rol: la autenticación es un secreto compartido, no un modelo de
  usuarios.
- **RQF02 (1)(2)(3) sin validar con datos propios**: las heurísticas de pasamanos
  y distracción funcionan y están implementadas, pero **no se han medido**
  porque no hay ground truth para ellas en ningún dataset disponible.
- **RQNF11** (disponibilidad en horario de campus) y **RQNF27** (restricción de
  actuación física) dependen del despliegue, no del código.

---

## Documentación de decisiones

- `Caso.md` — documento de requisitos (IEEE 830 / ISO 25010).
- `DECISIONES.md` — registro consolidado de decisiones de diseño: auditoría de
  datos, unificación de datasets, pseudo-etiquetado de Le2i, motor de riesgo,
  desviaciones conscientes (ByteTrack→IoU, 3s→1.2s persistencia, 75%→35% umbral
  de alerta), reducción de sesgo, estado operativo, borrado de alertas.

---

## Alertas

Una alerta salta cuando el riesgo **supera el tiempo de persistencia** exigido
para su severidad y se mantiene ese tiempo:

| Riesgo | Persistencia | Motivo |
|---|---|---|
| **Rojo** (caída, pérdida de equilibrio) | 1.2 s | filtra un tropiezo puntual sin perder la caída real |
| **Ámbar** (pre-caída, tambaleo) | 0.6 s | el tambaleo es más breve y hay que avisar antes |

Con tres capas de filtrado, como pide `RQF04`:

1. **0.35 de confianza** para pasar la puerta de la alerta
2. **tiempo de persistencia** según severidad, con **histéresis de 1 s**: si el
   riesgo parpadea un fotograma no reinicia la cuenta
3. **enfriamiento de 15 s** por persona y tipo, para que una condición
   sostenida no genere una alerta cada 15 segundos

### Qué se guarda

Cada alerta deja en `runs/alerts/<AAAA-MM-DD>/`:

- el **fotograma anotado** con las cajas y las señales de la postura
- una línea en `alertas.jsonl` con hora, clase, severidad, confianza,
  duración, identidad, referencia al requisito y ruta de imagen

Los rostros se **pixelizan** antes de escribir (`RQNF14`, Ley 29733). La limpieza
es automática: se conservan 7 días o las últimas 200 imágenes.

### Desde el panel

Cada pestaña tiene, bajo el panel de alertas:

- **Descargar todo (ZIP)** — imágenes + registro + un README
- **✕** en cada alerta — borrarla (imagen y línea juntas)
- **casillas** + **Borrar seleccionadas**
- **Borrar todas**, con confirmación

### Imágenes fijas

Una imagen subida **no genera alertas**: se analiza y se descarga con sus cajas,
pero no suena ninguna alerta. Una fotografía no es un suceso en el tiempo. Las
alertas son solo de vídeo y cámara en vivo.


---

## Estructura

```
├── Caso.md                  documento de requisitos del proyecto
├── DECISIONES*.md           registro de decisiones
├── configs/train.yaml       parámetros de entrenamiento
├── requirements.txt         entorno reproducible
├── scripts/
│   ├── audit_datasets.py    auditoría de los datasets de origen
│   ├── prepare_le2i.py      pseudo-etiquetado con puerta de validación
│   ├── prepare_dataset.py   unificación a YOLO sin fuga
│   ├── add_zoom_variants.py primeros planos por recorte exacto
│   ├── train.py             entrenamiento + evaluación
│   ├── evaluate.py          evaluación de un checkpoint
│   ├── eval_scale.py        evaluación por escala y recorte
│   ├── extract_frames.py    extracción de fotogramas de un clip (temporal)
│   ├── label_video_frames.py etiquetado de esos fotogramas (temporal)
│   └── build_kaggle_variant.py  variante con dataset externo (descartado)
└── app/
    ├── server.py            servidor Flask + hilo trabajador
    ├── risk_engine.py       motor de riesgo (RQF02-RQF07) y alertas
    ├── real_esrgan.py       superresolución 4× Real-ESRGAN (ONNX→PyTorch FX)
    ├── build_esrgan.py      descarga del modelo ONNX (HuggingFace)
    └── templates/           interfaz (index, login)
```

## Licencia

El código de este repositorio se distribuye bajo AGPL-3.0, alineado con
Ultralytics. Los datasets y los modelos preentrenados **no** se incluyen y
conservan sus propias licencias; ver [Créditos](#datasets-utilizados-y-créditos).