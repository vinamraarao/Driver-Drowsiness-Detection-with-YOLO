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
        "ear_valid_samples", "ear_rejection_rate", "calibration_quality"
    )

    def __init__(self, ear_threshold, mar_threshold, ear_baseline, mar_baseline,
                 ear_samples, mar_samples, calibrated, reason,
                 ear_valid_samples=None, ear_rejection_rate=0.0, calibration_quality="unknown"):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.ear_baseline = ear_baseline
        self.mar_baseline = mar_baseline
        self.ear_samples = ear_samples
        self.mar_samples = mar_samples
        self.calibrated = calibrated
        self.reason = reason
        self.ear_valid_samples = ear_valid_samples if ear_valid_samples is not None else len(ear_samples)
        self.ear_rejection_rate = ear_rejection_rate
        self.calibration_quality = calibration_quality


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


def _is_valid_ear_sample(ear_value):
    """Check if an EAR sample is valid for calibration.

    Reject samples that are:
    - Too low (< 0.08): indicates eyes mostly closed or landmark detection failure
    - Too high (> 0.45): indicates landmark detection failure or unusual eye geometry
    - Non-finite values
    """
    if not isinstance(ear_value, (int, float)) or not math.isfinite(ear_value):
        return False
    if ear_value < 0.08 or ear_value > 0.45:
        return False
    return True


def _is_valid_mar_sample(mar_value):
    """Check if a MAR sample is valid for calibration.

    Reject samples that are:
    - Too low (< 0.10): indicates mouth closed and landmarks stable, but if all
      samples are this low it may indicate landmark detection issues
    - Too high (> 2.0): indicates landmark detection failure
    - Non-finite values
    """
    if not isinstance(mar_value, (int, float)) or not math.isfinite(mar_value):
        return False
    if mar_value < 0.10 or mar_value > 2.0:
        return False
    return True


def derive_thresholds(ear_samples, mar_samples):
    """Derive EAR/MAR thresholds from baseline samples.

    Returns a CalibrationResult. Always returns usable thresholds: if the
    samples are degenerate, the safe defaults from config are used.

    EAR and MAR are calibrated independently. If one fails quality checks,
    its threshold falls back to the default while the other may still be calibrated.
    The overall ``calibrated`` flag is True only if at least one threshold was
    successfully calibrated.
    """
    # Filter EAR samples to reject invalid/abnormally low values
    valid_ear_samples = [ear for ear in ear_samples if _is_valid_ear_sample(ear)]
    ear_rejection_rate = 0.0
    if ear_samples:
        ear_rejection_rate = 1.0 - (len(valid_ear_samples) / len(ear_samples))

    # Filter MAR samples to reject invalid values
    valid_mar_samples = [mar for mar in mar_samples if _is_valid_mar_sample(mar)]
    mar_rejection_rate = 0.0
    if mar_samples:
        mar_rejection_rate = 1.0 - (len(valid_mar_samples) / len(mar_samples))

    # Calculate means for both EAR and MAR (using filtered samples)
    ear_mean = _safe_mean(valid_ear_samples)
    mar_mean = _safe_mean(valid_mar_samples)

    # Determine EAR calibration quality
    min_valid_ear_samples = max(config.CALIBRATION_MIN_SAMPLES // 2, 5)  # At least half or 5 samples
    sufficient_total_ears = len(ear_samples) >= config.CALIBRATION_MIN_SAMPLES  # Enough total samples collected
    sufficient_valid_ears = len(valid_ear_samples) >= min_valid_ear_samples  # Enough valid samples

    calibration_quality = "poor"
    if sufficient_total_ears and sufficient_valid_ears and ear_rejection_rate < 0.5:  # Less than 50% rejection rate
        calibration_quality = "good"
    elif sufficient_total_ears and sufficient_valid_ears:
        calibration_quality = "marginal"  # Enough total and valid samples but high rejection rate
    elif sufficient_total_ears and not sufficient_valid_ears:
        # Enough total samples collected but not enough valid ones = poor quality
        calibration_quality = "poor"
    # Note: if sufficient_total_ears is False, we'll handle it as "insufficient samples" below

    # --- EAR threshold calculation ---
    ear_threshold = config.EAR_THRESHOLD  # Default
    ear_was_calibrated = False
    if ear_mean is not None and sufficient_total_ears and calibration_quality in ("good", "marginal"):
        ear_std = _safe_std(ear_samples, ear_mean)
        ear_offset = config.CALIBRATION_EAR_STDDEV_OFFSET * max(ear_std, 0.01)
        ear_threshold = _clamp(ear_mean - ear_offset,
                               config.EAR_CALIB_MIN, config.EAR_CALIB_MAX)
        ear_was_calibrated = (ear_threshold != config.EAR_THRESHOLD)
        # If the calibrated EAR threshold is too low, fall back to default
        if ear_threshold < 0.10:
            ear_threshold = config.EAR_THRESHOLD
            ear_was_calibrated = False

    # --- MAR threshold calculation (independent of EAR) ---
    mar_threshold = config.MAR_THRESHOLD  # Default
    mar_was_calibrated = False
    if mar_mean is not None and len(valid_mar_samples) >= min_valid_ear_samples:
        mar_std = _safe_std(mar_samples, mar_mean)
        mar_offset = config.CALIBRATION_MAR_STDDEV_OFFSET * max(mar_std, 0.01)
        mar_threshold = _clamp(mar_mean + mar_offset,
                               config.MAR_CALIB_MIN, config.MAR_CALIB_MAX)
        mar_was_calibrated = (mar_threshold != config.MAR_THRESHOLD)

    # Determine overall calibration status
    calibrated = ear_was_calibrated or mar_was_calibrated

    # Build reason message
    reason_parts = []

    # EAR reasoning
    if len(ear_samples) == 0:
        reason_parts.append("no EAR samples detected")
    elif not sufficient_total_ears:
        # Not enough total samples collected during calibration period
        reason_parts.append(f"insufficient EAR samples collected ({len(ear_samples)}/{config.CALIBRATION_MIN_SAMPLES})")
    elif len(valid_ear_samples) == 0:
        # We have samples but none are valid = poor calibration quality
        reason_parts.append(f"poor EAR calibration quality (all samples rejected)")
    elif not sufficient_valid_ears:
        # Enough total samples but not enough valid ones = poor quality
        reason_parts.append(f"poor EAR calibration quality (insufficient valid samples: {len(valid_ear_samples)}/{len(ear_samples)})")
    elif calibration_quality == "poor":
        reason_parts.append(f"poor EAR calibration quality (rejection rate: {ear_rejection_rate:.1%})")
    elif ear_was_calibrated:
        reason_parts.append(f"EAR calibrated from {len(valid_ear_samples)} valid samples")
    else:
        reason_parts.append("EAR using default (calibration produced no improvement)")

    # MAR reasoning
    if len(mar_samples) == 0:
        reason_parts.append("no MAR samples detected")
    elif len(valid_mar_samples) == 0:
        # We have samples but none are valid = poor calibration quality
        reason_parts.append(f"poor MAR calibration quality (all samples rejected)")
    else:
        if mar_was_calibrated:
            reason_parts.append(f"MAR calibrated from {len(valid_mar_samples)} valid samples")
        else:
            reason_parts.append("MAR using default threshold")

    # Special case: if both EAR and MAR have no valid samples, indicate safe defaults
    if len(valid_ear_samples) == 0 and len(valid_mar_samples) == 0:
        reason_parts = ["using safe defaults"]

    reason = "; ".join(reason_parts)

    return CalibrationResult(
        ear_threshold, mar_threshold,
        ear_mean, mar_mean,
        ear_samples, mar_samples, calibrated,
        reason,
        ear_valid_samples=len(valid_ear_samples),
        ear_rejection_rate=ear_rejection_rate,
        calibration_quality=calibration_quality
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