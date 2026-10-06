"""Regression tests for the night-vision path: no NLM, cheap, correct hysteresis."""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import config
import night_vision as nv


def test_nlm_disabled_by_default():
    assert config.DENOISE_H_NIGHT == 0, "fastNlMeansDenoising costs 180-350 ms/frame; keep it off the real-time path"


def test_luma_thresholds_match_measured_cabin():
    # Measured native mean luma: dark cabin ~16 (index 1) and ~32 (index 0); daylight ~120-160.
    assert config.NIGHT_MODE_ENTER_LUMA > 32
    assert config.NIGHT_MODE_EXIT_LUMA > config.NIGHT_MODE_ENTER_LUMA


def test_enhance_is_under_20ms_at_640x480():
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    nv.enhance_for_low_light(frame)  # warm-up
    t = time.perf_counter()
    for _ in range(10):
        out = nv.enhance_for_low_light(frame)
    per_frame_ms = (time.perf_counter() - t) / 10 * 1000
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert per_frame_ms < 20.0, f"enhance_for_low_light took {per_frame_ms:.1f} ms"


def test_hysteresis_on_measured_values():
    state = nv.NightModeState()
    dark = np.full((480, 640, 3), 16, dtype=np.uint8)      # measured dark cabin
    bright = np.full((480, 640, 3), 130, dtype=np.uint8)   # typical daylight cabin
    assert state.update(dark) is True, "dark cabin must switch night mode on"
    assert state.update(bright) is False, "daylight cabin must switch night mode off again"
    assert state.update(bright) is False
    assert state.update(dark) is True


def test_enhanced_frame_is_brighter():
    dark = np.full((480, 640, 3), 16, dtype=np.uint8)
    out = nv.enhance_for_low_light(dark)
    assert nv.frame_mean_luma(out) > nv.frame_mean_luma(dark)