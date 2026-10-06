"""Driver-specific EAR/MAR calibration.

Collects a short baseline of eye and mouth measurements at startup and derives
reasonable thresholds from actual measurements. Falls back to safe defaults if
calibration fails or produces degenerate values.
"""
import math
import time

import config


class CalibrationResult:
    """Result of a calibration run."""

    __slots__ = (
        "ear_threshold", "mar_threshold", "ear_baseline", "mar_baseline",
        "ear_samples", "mar_samples", "calibrated", "reason",
    )

    def __init__(self, ear_threshold, mar_threshold, ear_baseline, mar_baseline,
                 ear_samples, mar_samples, calibrated, reason):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.ear_baseline = ear_baseline
        self.mar_baseline = mar_baseline
        self.ear_samples = ear_samples
        self.mar_samples = mar_samples
        self.calibrated = calibrated
        self.reason = reason


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _safe_mean(values):
    """Mean of non-None values, or None if the list is empty."""
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _safe_std(values, mean):
    """Population standard deviation of non-None values."""
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if len(clean) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in clean) / len(clean))


def collect_baseline(ear_fn, mar_fn, duration_sec=None, min_samples=None):
    """Collect baseline EAR/MAR samples for ``duration_sec`` seconds.

    ``ear_fn`` and ``mar_fn`` are callables returning the current EAR and MAR
    (or None when the face is unavailable). Returns (ear_samples, mar_samples).
    """
    duration = duration_sec if duration_sec is not None else config.CALIBRATION_DURATION_SEC
    min_samples = min_samples if min_samples is not None else config.CALIBRATION_MIN_SAMPLES
    ear_samples = []
    mar_samples = []
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        ear = ear_fn()
        mar = mar_fn()
        if ear is not None:
            ear_samples.append(float(ear))
        if mar is not None:
            mar_samples.append(float(mar))
        time.sleep(0.05)  # ~20 Hz sampling
        if len(ear_samples) >= min_samples and len(mar_samples) >= min_samples:
            # Keep collecting until the deadline so the baseline is time-stable.
            pass
    return ear_samples, mar_samples


def derive_thresholds(ear_samples, mar_samples):
    """Derive EAR/MAR thresholds from baseline samples.

    Returns a CalibrationResult. Always returns usable thresholds: if the
    samples are degenerate, the safe defaults from config are used.
    """
    ear_mean = _safe_mean(ear_samples)
    mar_mean = _safe_mean(mar_samples)

    if ear_mean is None or mar_mean is None:
        return CalibrationResult(
            config.EAR_THRESHOLD, config.MAR_THRESHOLD,
            None, None, ear_samples, mar_samples, False,
            "no face detected during calibration; using safe defaults",
        )

    ear_std = _safe_std(ear_samples, ear_mean)
    mar_std = _safe_std(mar_samples, mar_mean)

    # EAR: eyes-closed threshold sits below the baseline mean by N standard
    # deviations, clamped to a sane range. A blink or a small glance should not
    # trigger drowsiness; only sustained closure should.
    ear_offset = config.CALIBRATION_EAR_STDDEV_OFFSET * max(ear_std, 0.01)
    ear_threshold = _clamp(ear_mean - ear_offset,
                           config.EAR_CALIB_MIN, config.EAR_CALIB_MAX)

    # MAR: yawning threshold sits above the baseline mean by N standard
    # deviations, clamped to a sane range.
    mar_offset = config.CALIBRATION_MAR_STDDEV_OFFSET * max(mar_std, 0.01)
    mar_threshold = _clamp(mar_mean + mar_offset,
                           config.MAR_CALIB_MIN, config.MAR_CALIB_MAX)

    calibrated = ear_threshold != config.EAR_THRESHOLD or mar_threshold != config.MAR_THRESHOLD
    reason = "calibrated from {} EAR / {} MAR samples".format(
        len(ear_samples), len(mar_samples))

    return CalibrationResult(
        ear_threshold, mar_threshold,
        ear_mean, mar_mean,
        ear_samples, mar_samples, calibrated, reason,
    )


class ThresholdStore:
    """Holds the current calibrated thresholds and exposes them to the runtime.

    The runtime reads ``ear_threshold`` and ``mar_threshold`` on every frame,
    so updating these attributes is enough to change behaviour immediately.
    """

    __slots__ = ("ear_threshold", "mar_threshold", "ear_baseline",
                 "mar_baseline", "calibrated", "reason")

    def __init__(self):
        self.ear_threshold = config.EAR_THRESHOLD
        self.mar_threshold = config.MAR_THRESHOLD
        self.ear_baseline = None
        self.mar_baseline = None
        self.calibrated = False
        self.reason = "not calibrated"

    def apply(self, result):
        self.ear_threshold = result.ear_threshold
        self.mar_threshold = result.mar_threshold
        self.ear_baseline = result.ear_baseline
        self.mar_baseline = result.mar_baseline
        self.calibrated = result.calibrated
        self.reason = result.reason

    def reset(self):
        self.ear_threshold = config.EAR_THRESHOLD
        self.mar_threshold = config.MAR_THRESHOLD
        self.ear_baseline = None
        self.mar_baseline = None
        self.calibrated = False
        self.reason = "reset to defaults"


def run_calibration(face_landmarker, duration_sec=None, min_samples=None):
    """Collect a short baseline and derive EAR/MAR thresholds.

    Runs the real FaceLandmarker for ``duration_sec`` seconds, collecting EAR and
    MAR samples. Returns a CalibrationResult with the derived thresholds and the
    raw samples. Falls back to safe defaults if the face is never detected.
    """
    import time
    import cv2
    import face_analyzer as fa

    duration = duration_sec if duration_sec is not None else config.CALIBRATION_DURATION_SEC
    min_samples = min_samples if min_samples is not None else config.CALIBRATION_MIN_SAMPLES

    # Use the configured camera index for calibration.
    from camera_utils import open_camera
    cap = open_camera(preferred_index=config.DEFAULT_CAMERA_INDEX)
    if cap is None:
        return CalibrationResult(
            config.EAR_THRESHOLD, config.MAR_THRESHOLD,
            None, None, [], [], False,
            "no camera available during calibration; using safe defaults",
        )

    try:
        ear_samples = []
        mar_samples = []
        deadline = time.monotonic() + duration
        frame_id = 0
        while time.monotonic() < deadline:
            ret, raw = cap.read()
            if not ret or raw is None or raw.size == 0:
                time.sleep(0.02)
                continue
            frame = cv2.flip(raw, 1) if config.FLIP_FRAME else raw
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = fa.new_mp_image(rgb)
            ts_ms = int(time.monotonic_ns() // 1_000_000)
            res = fa.analyze_frame(face_landmarker, mp_image, ts_ms, w, h)
            if res.face_detected and res.avg_ear is not None:
                ear_samples.append(float(res.avg_ear))
                if res.mar is not None:
                    mar_samples.append(float(res.mar))
            frame_id += 1
            time.sleep(0.03)
    finally:
        cap.release()

    return derive_thresholds(ear_samples, mar_samples)