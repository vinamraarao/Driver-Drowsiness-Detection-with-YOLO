"""Regression tests for the alert priority chain and the no-face safety rule.

The chain itself lives inline in main.main(); this file re-declares the same
order and drives it with real DriverState objects so it runs without a camera.
No pytest fixtures are used.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from temporal_state import DriverState

CLOSED = config.EAR_THRESHOLD - 0.05
OPEN = config.EAR_THRESHOLD + 0.05


class _Flags:
    """Duck-typed stand-in for DriverState with controllable flags."""

    def __init__(self, sleep=False, phone=False, drowsy=False, yawn=False,
                 distraction=False, hands=False):
        self.sleep_active = sleep
        self.drowsy_active = drowsy
        self.yawn_active = yawn
        self.distraction_active = distraction
        self.hands_missing = hands


def choose_alert(state, phone_present):
    """Mirror of main.py alert selection (main.py lines 275-305)."""
    if state.sleep_active and phone_present:
        return "sleep_phone"
    if state.sleep_active:
        return "sleep"
    if phone_present:
        return "phone"
    if state.drowsy_active:
        return "drowsiness"
    if state.yawn_active:
        return "yawn"
    if state.distraction_active:
        return "distraction"
    if config.ENABLE_HAND_ALERTS and state.hands_missing:
        return "hands"
    return None


def test_nominal_produces_no_alert():
    assert choose_alert(_Flags(), False) is None


def test_sleep_beats_everything():
    assert choose_alert(_Flags(sleep=True), True) == "sleep_phone"
    assert choose_alert(_Flags(sleep=True, drowsy=True, yawn=True, distraction=True), False) == "sleep"


def test_phone_beats_drowsiness():
    assert choose_alert(_Flags(drowsy=True, yawn=True), True) == "phone"
    assert choose_alert(_Flags(drowsy=True), True) == "phone"


def test_drowsiness_beats_yawn_and_distraction():
    assert choose_alert(_Flags(drowsy=True, yawn=True, distraction=True), False) == "drowsiness"
    assert choose_alert(_Flags(yawn=True, distraction=True), False) == "yawn"
    assert choose_alert(_Flags(distraction=True), False) == "distraction"


def test_every_alert_kind_has_a_voice_cooldown():
    for k in ["sleep_phone", "sleep", "phone", "drowsiness", "yawn", "distraction", "hands"]:
        assert k in config.VOICE_COOLDOWNS


def test_hands_alert_disabled_by_default():
    assert config.ENABLE_HAND_ALERTS is False
    assert choose_alert(_Flags(hands=True), False) is None


def test_no_face_after_grace_yields_no_drowsiness_alert():
    """Regression for the 'Searching / DRIVER SLEEP DETECTED' CSV row."""
    s = DriverState()
    now = 100.0
    s.update_face(True, now)
    t = now
    for _ in range(6):
        s.update_eyes(CLOSED, t)
        t += 0.05
    t += config.SLEEP_DURATION_SEC
    s.update_eyes(CLOSED, t)
    assert choose_alert(s, False) == "sleep"

    # Face leaves the frame; the loop stops feeding eyes and passes False for distraction.
    for _ in range(40):
        t += 0.05
        s.update_face(False, t)
        s.update_distraction(False, t)
    assert s.face_lost is True
    assert choose_alert(s, False) is None


def test_face_return_with_open_eyes_clears_alert():
    s = DriverState()
    now = 200.0
    s.update_face(True, now)
    t = now
    for _ in range(6):
        s.update_eyes(CLOSED, t)
        t += 0.05
    t += config.SLEEP_DURATION_SEC + 0.05
    s.update_eyes(CLOSED, t)
    assert choose_alert(s, False) == "sleep"
    s.update_face(True, t)
    for _ in range(40):
        t += 0.05
        s.update_eyes(OPEN, t)
    assert choose_alert(s, False) is None


def test_single_blink_does_not_downgrade_or_clear_the_alert():
    s = DriverState()
    now = 300.0
    s.update_face(True, now)
    t = now
    for _ in range(6):
        s.update_eyes(CLOSED, t)
        t += 0.05
    t += config.SLEEP_DURATION_SEC + 0.05
    s.update_eyes(CLOSED, t)
    assert choose_alert(s, False) == "sleep"
    t += 0.05
    s.update_eyes(OPEN, t)  # blink
    assert choose_alert(s, False) == "sleep", "a blink must not clear or downgrade sleep"