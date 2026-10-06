# Driver Monitoring System — Hardened Build

Camera display, fullscreen dashboard, driver-side detection, face landmarks,
drowsiness/eye closure, yawning, head-turn distraction, phone detection, hand
visualization, night enhancement, and CSV event logging. Voice is the only
alert channel — there is no beep path.

One OpenCV display window. Press **Q** or **Esc** to exit.

## Accuracy note

No software-only rewrite can guarantee a 90–95% real-world accuracy rate
without measuring the system on labeled data from the actual camera,
mounting position, lighting, users, and failure cases. The included quality
gate and training tools exist so that target is measured, not assumed.

The default detector is **YOLO11n** (Ultralytics 8.3.0), with a published
COCO mAP50-95 of 39.5 at 640 px. That benchmark is not the same as a
90–95% driver-phone event accuracy number — see "Reaching 90–95%" below.

## Folder layout

```text
driver_monitoring_final/
  main.py
  app.py
  config.py
  yolo_detector.py
  face_analyzer.py
  hand_analyzer.py
  geometry.py
  temporal_state.py
  night_vision.py
  camera_utils.py
  audio_alert.py
  event_logger.py
  hud.py
  health_check.py
  train_yolo.py
  evaluate_yolo.py
  requirements.txt
  event_history_log.csv   (created at runtime)
  models/
    face_landmarker.task
    hand_landmarker.task
    yolo11n.pt             (downloaded on first run if not already present)
  dataset/
    data.yaml
    README.txt
    images/train/
    images/val/
    labels/train/
    labels/val/
  tests/
    test_geometry.py
    test_temporal.py
```

## Manual setup (Windows)

Use Python 3.12 for the most predictable install on this stack. Run everything
from the `venv/` in this folder (the system Python does not have cv2).

```
python -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python health_check.py
venv\Scripts\python main.py
```

The first run may download the YOLO11n weights. The face and hand MediaPipe
task files are already included in this package.

### GPU

`health_check.py` reports the inference device that will be used. The default
`YOLO_DEVICE = "auto"` uses the NVIDIA GPU when `torch.cuda.is_available()` is
true and falls back to CPU otherwise. The stock `torch` wheels are CPU-only; to
use an RTX-class GPU install the CUDA build yourself (nothing is installed
automatically):

```
venv\Scripts\python -m pip uninstall -y torch torchvision
venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
venv\Scripts\python health_check.py
```

Set `YOLO_DEVICE = "cpu"` or `YOLO_FORCE_CPU = True` in `config.py` to force CPU.

### Tests

```
venv\Scripts\python -m pip install pytest
venv\Scripts\python -m pytest tests -q
```

### Useful flags

```bat
:: Default camera (index 1, see DEFAULT_CAMERA_INDEX in config.py)
python main.py

:: Use a specific camera
python main.py --camera-index 0

:: Force a different Windows camera backend
python main.py --camera-index 1 --backend msmf

:: Normal resizable window instead of fullscreen
python main.py --windowed
```

## What was fixed

- **Night mode stays real-time.** `cv2.fastNlMeansDenoising` (180-350 ms per
  640x480 frame) is disabled by default (`DENOISE_H_NIGHT = 0`); CLAHE + gamma
  + unsharp costs about 3 ms. Night enter/exit luma thresholds were recalibrated
  from the measured cabin (dark cabin ~16-32, daylight ~120-160) so night mode
  does not latch on in a normal cabin.
- **The hand landmarker is gated.** It is not run at all when
  `ENABLE_HAND_ALERTS = False`, and runs at 5 Hz (`HAND_INTERVAL_SEC = 0.2`)
  when enabled. It costs 18-23 ms per frame.
- **Inference device is chosen at startup** (`YOLO_DEVICE = "auto"`): GPU when
  CUDA is available, CPU otherwise, with the reason printed by `health_check.py`
  and `load_yolo_model()`.
- **Voice alerts use one bounded worker and a bounded queue.** `voice_alert`
  returns immediately after a cooldown check; the per-key cooldowns in
  `VOICE_COOLDOWNS` are used (the old `VOICE_ALERT_COOLDOWN_SEC` reference never
  existed and silently forced every alert to 8 s); `shutdown_voice()` stops the
  worker so nothing speaks after exit.
- **Drowsiness state no longer blinks itself away.** A single blink does not
  clear `drowsy_active`; `sleep_active` clears only after `RECOVERY_DURATION_SEC`
  of continuously open eyes; and when the face has been missing for
  `FACE_GRACE_SEC` the sleep, drowsy, yawn and pose state plus all three
  smoothers are dropped, so "DRIVER SLEEP" can never fire with nobody in frame.
- **Timers are time-based, not frame-based** — changing camera FPS no longer
  silently changes the meaning of the drowsiness/yawn/distraction thresholds.
- **YOLO runs in a background worker** with a bounded "latest frame only"
  design. Old results expire, camera reconnection clears stale detections,
  and inference errors are recorded as a degraded state instead of a false
  "nothing detected" state.
- **Phone detection** uses phone-box containment inside the driver region
  rather than plain IoU, requires repeated fresh YOLO confirmations, and
  clears only after repeated misses — reducing one-frame false alarms.
- **Voice alerts** use one dedicated worker and a bounded queue: no
  one-thread-per-warning, and no beep implementation anywhere in the project.
- **Face model download** uses a timeout and atomic temp-file replacement,
  so an interrupted download can't leave a corrupt model file that looks
  valid.
- **CSV logging is non-fatal** — disk/full-file-lock errors are reported
  without taking down the camera loop.
- **`app.py`** is now only a compatibility wrapper around `main.py`, so
  there is one authoritative implementation.
- **Hand display** is retained, but no longer claims detected hands mean
  "hands off wheel" (that branch was logically reversed). The build shows
  driver/passenger hands and does not claim steering-wheel contact, since
  it doesn't detect the wheel itself.

## Reaching 90–95%

Fine-tune the detector on your own cabin-camera data, then validate on a
held-out test set plus a separate real-driving test set. `dataset/README.txt`
explains the expected YOLO dataset format.

```bat
python train_yolo.py --data dataset/data.yaml --model yolo11n.pt --device 0 --epochs 100
python evaluate_yolo.py --data dataset/data.yaml --model runs/detect/train/weights/best.pt --device 0 --min-map50 0.90
```

The `--min-map50 0.90` gate makes evaluation fail when mAP50 is below 90%.
It doesn't make the live system 90% by itself — it prevents the project from
being called "90% accurate" without a measured validation result.

## Licensing

This project uses the Ultralytics package and pretrained YOLO11 weights.
Ultralytics documents YOLO11 under AGPL-3.0 and Enterprise licensing —
review the applicable terms for your intended use or commercial deployment.