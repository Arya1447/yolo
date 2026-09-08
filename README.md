# Terpal-Orange YOLO Detector

Custom-trained YOLO11n object detector, single class `Terpal-Orange` (an
orange ground tarp/marker), running on an Orange Pi 4 Pro as the
object-detection component of a UAV pipeline (KRTI 2026 competition). No
training code here — this repo is the **inference side only**: a trained
model plus a hardware-tuned runner.

## TL;DR

```bash
pip install ultralytics ncnn opencv-python pyyaml
python3 detect_terpal.py --source 0                    # webcam, default (640) model
python3 detect_terpal.py --source photo.jpg             # single image
python3 detect_terpal.py --source 0 --record out.mp4    # + record annotated video
```
Full flags: `python3 detect_terpal.py --help`.

## Hardware context (why this isn't just `yolo predict`)

- **Board**: Orange Pi 4 Pro, Allwinner A733 — 2× Cortex-A76 @2.0GHz + 6×
  Cortex-A55 @1.79GHz, plus a 3 TOPS Vivante VIP9000 NPU.
- **The NPU is not used.** Its toolchain (ACUITY/Pegasus conversion,
  VIPLite runtime) needs a separate x86 host with Docker and relies on
  reverse-engineered community tooling, not an official SDK — out of scope
  here. Everything runs on **CPU via NCNN** instead (Ultralytics' own
  recommended path for ARM boards, ~2× faster than raw PyTorch on this
  class of hardware).
- Measured on this exact board: **~7.5–8.5 FPS at imgsz 640**, **~13 FPS at
  imgsz 480** (see the model-size table below).

## Repo contents

| Path | What it is |
|---|---|
| `detect_terpal.py` | The detector: threaded capture, NCNN inference, optional recording/save. |
| `terpal_orange.pt` | Trained weights (Ultralytics YOLO11n, 1 class), PyTorch format. |
| `terpal_orange_ncnn_model_640/` | NCNN export, fixed input 640×640 — matches training resolution, the accurate/default one. |
| `terpal_orange_ncnn_model_480/` | NCNN export, fixed input 480×480 — faster, measured less reliable (see below). |

## ⚠️ The trap that will burn you: NCNN exports are FIXED-SHAPE

`model.export(format="ncnn", imgsz=N)` bakes `N` into the traced graph
(PNNX). Run inference at any other `--imgsz` and it does **not error** — it
silently produces garbage: hundreds of fake boxes, every one at confidence
`1.00`. This is exactly how it presented in testing and it looks like a
real detection burst if you're not watching for it.

`detect_terpal.py` guards against this already: it reads the expected
`imgsz` from each model's `metadata.yaml` and refuses to run (loud error,
not silent garbage) if `--imgsz` doesn't match. If you add a new export at
a different size, point `--model` at that folder — don't just pass a
different `--imgsz` against an existing one.

## Model size tradeoff: 640 vs 480

Measured back-to-back on this board, same camera, same lighting:

| | `terpal_orange_ncnn_model_640` | `terpal_orange_ncnn_model_480` |
|---|---|---|
| Matches training resolution | yes | no (downscaled) |
| FPS | ~7.5–8.5 | ~13–13.5 |
| False positives observed | none | **yes** — a bright ceiling light panel was detected as `Terpal-Orange` at conf 0.65–0.78 |

`detect_terpal.py` defaults to the 640 model. The sample size behind this
table is small — no real orange-tarp ground truth was available during this
testing round, only sanity checks against non-target scenes. Treat 480's
speed advantage as **unproven, not disproven**, until it's validated
against the actual target in the field.

## Recording

`--record path.mp4` records every processed frame (with the detection
overlay burned in) from start until the process stops. The filename is
stamped with the run's **start time** automatically —
`path.mp4` → `path_20260904_163152.mp4` — so repeated runs never overwrite
each other. This matters on the deployed box specifically because the
`yolo.service` unit (see Deployment) has `Restart=always` and can start
fresh multiple times a day.

## Camera source

`--source` takes a V4L2 index/path (`0`, `/dev/video0`, …), an image file,
or a video file.

On the deployed board it's run as `--source /dev/video11`, a
`v4l2loopback` virtual device fed by a relay process that *also* feeds a
separate video-streaming pipeline (`/dev/video10`) — this lets both
consumers read the one physical USB camera concurrently without fighting
over exclusive V4L2 access. That relay lives in the streaming project's
repo (a different codebase), not here. If you're running this script
standalone against a real camera, none of that applies — just use
`--source 0`.

Format handling: `/dev/video0` is forced to `MJPG` (this camera's native
fast format — a plain USB2.0 UVC cam collapses in FPS above 480p on raw
YUYV). Any other source is left to auto-negotiate, since the loopback
device above carries raw video, not MJPEG, and forcing MJPG against it
breaks capture.

## Deployment (this board specifically)

Runs as systemd unit `yolo.service`: starts 5s after the video-streaming
service (`ExecStartPre=sleep 5`, ordered `After=`), pinned to the 2 A76
cores (`CPUAffinity=6 7`) so it doesn't contend with the streaming
pipeline's software H.264 encoder (pinned to the 6 A55 cores — there's no
hardware encoder on this SoC). Recordings land in `/var/lib/yolo/` and are
**not** pruned automatically — check disk usage periodically if the service
runs unattended for long stretches.

## Requirements

```
ultralytics
ncnn
opencv-python
pyyaml
```
No `requirements.txt` yet — installed ad hoc in a venv on the deployed
board (`torch`/`torchvision` come along as `ultralytics` dependencies, used
only if you re-export from `.pt`; inference itself only needs the NCNN
backend).
