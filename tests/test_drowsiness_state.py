"""Regression tests for the temporal drowsiness / sleep / face-loss state machine.

No pytest fixtures are used: tests drive ``DriverState`` with explicit
monotonic timestamps so results do not depend on wall-clock timing.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from temporal_state import DriverState

CLOSED = config.EAR_THRESHOLD - 0.05
OPEN = config.EAR_THRESHOLD + 0.05
YAWN_MAR = config.MAR_THRESHOLD + 0.2
NORMAL_MAR = config.MAR_THRESHOLD - 0.2
STEP = 0.05  # 20 Hz feed rate


def feed(s, value, seconds, key):
    """Feed ``value`` for ``seconds`` at STEP, advancing the clock on ``key``."""
    t = s._clock[key]
    n = int(round(seconds / STEP))
    for _ in range(n):
        t += STEP
        s._clock[key] = t
        s.update_eyes(value, t) if key == "eyes" else (
            s.update_yawn(value, t) if key == "yawn" else s.update_distraction(value, t)
        )
    return s._clock[key]


def new_state(now=100.0, frames=6):
    s = DriverState()
    s._clock = {"eyes": now, "yawn": now, "distraction": now}
    s.update_face(True, now)
    for _ in range(frames):
        s.update_eyes(CLOSED, now)
        s._clock["eyes"] += STEP
    return s


def test_smoothing_makes_eyes_stay_closed_under_occasional_blinks():
    # A real blink is 1-2 frames; with a 5-sample mean the smoothed EAR must stay
    # below the threshold so the closure timer is not reset.
    s = new_state()
    assert s.update_eyes(CLOSED, s._clock["eyes"]) is not None
    t = s._clock["eyes"] + STEP
    s.update_eyes(OPEN, t)
    smoothed = s.update_eyes(CLOSED, t + STEP)
    assert smoothed < config.EAR_THRESHOLD, "one blink must not look like open eyes"


def test_drowsy_triggers_after_duration():
    s = new_state()
    feed(s, CLOSED, config.DROWSY_DURATION_SEC, "eyes")
    assert s.drowsy_active is True
    assert s.sleep_active is False, "drowsy alone does not raise sleep"


def test_single_blink_does_not_clear_drowsy():
    # Regression: drowsy_active was cleared on any open frame.
    s = new_state()
    feed(s, CLOSED, config.DROWSY_DURATION_SEC, "eyes")
    assert s.drowsy_active is True
    feed(s, OPEN, STEP, "eyes")
    assert s.drowsy_active is True, "a single blink must not clear drowsiness"
    feed(s, CLOSED, STEP, "eyes")
    assert s.drowsy_active is True


def test_drowsy_clears_after_recovery_hold_of_open_eyes():
    s = new_state()
    feed(s, CLOSED, config.DROWSY_DURATION_SEC + 2 * STEP, "eyes")
    assert s.drowsy_active is True
    # Keep feeding OPEN until the smoothed EAR crosses the threshold, then hold.
    cleared = False
    for _ in range(40):
        t = s._clock["eyes"] + STEP
        s._clock["eyes"] = t
        s.update_eyes(OPEN, t)
        if not s.drowsy_active:
            cleared = True
            break
    assert cleared, "drowsy must clear once the smoothed eyes stay open for RECOVERY_DURATION_SEC"
    assert s.drowsy_active is False


def test_sleep_triggers_after_sleep_duration():
    s = new_state()
    feed(s, CLOSED, config.SLEEP_DURATION_SEC, "eyes")
    assert s.sleep_active is True


def test_sleep_clears_when_eyes_stay_open_past_recovery():
    # Regression: sleep stayed active (active = active or ...) after the eyes opened.
    s = new_state()
    feed(s, CLOSED, config.SLEEP_DURATION_SEC + 2 * STEP, "eyes")
    assert s.sleep_active is True
    cleared = False
    for _ in range(40):
        t = s._clock["eyes"] + STEP
        s._clock["eyes"] = t
        s.update_eyes(OPEN, t)
        if not s.sleep_active:
            cleared = True
            break
    assert cleared, "sleep must clear after RECOVERY_DURATION_SEC of continuously open eyes"
    assert s.sleep_active is False and s.drowsy_active is False


def test_sleep_survives_a_blink_mid_hold():
    # Stop 8 frames before the trigger so the smoothed EAR cannot cross the
    # threshold before the blink is over.
    s = new_state()
    feed(s, CLOSED, config.SLEEP_DURATION_SEC - 8 * STEP, "eyes")
    assert s.sleep_active is False
    feed(s, OPEN, STEP, "eyes")   # blink
    assert s.sleep_active is False
    feed(s, CLOSED, 8 * STEP, "eyes")
    assert s.sleep_active is True, "a blink must not restart the sleep timer"


def test_face_loss_within_grace_keeps_state():
    s = new_state()
    feed(s, CLOSED, config.SLEEP_DURATION_SEC, "eyes")
    # The face was last seen when it was primed; the sleep hold pushed the clock past
    # the grace period, so update_face must be fed a time inside the grace window.
    s.update_face(False, s.last_face_time + config.FACE_GRACE_SEC / 2)
    assert s.face_lost is False
    assert s.sleep_active is True, "state is held during the grace period"


def test_face_loss_beyond_grace_clears_all_face_alerts_and_smoothers():
    s = new_state()
    feed(s, CLOSED, config.SLEEP_DURATION_SEC + 2 * STEP, "eyes")
    feed(s, YAWN_MAR, config.YAWN_DURATION_SEC + 2 * STEP, "yawn")
    feed(s, True, config.DISTRACTION_DURATION_SEC + 2 * STEP, "distraction")
    assert s.sleep_active and s.yawn_active and s.distraction_active

    last_seen = s.last_face_time
    s.update_face(False, last_seen + config.FACE_GRACE_SEC + 0.01)
    assert s.face_lost is True
    assert s.sleep_active is False
    assert s.drowsy_active is False
    assert s.yawn_active is False
    assert s.distraction_active is False
    assert len(s.eye_smoother.values) == 0
    assert len(s.mar_smoother.values) == 0
    assert len(s.pose_smoother.values) == 0


def test_face_loss_reset_happens_once_not_every_frame():
    s = new_state()
    feed(s, CLOSED, config.SLEEP_DURATION_SEC + 2 * STEP, "eyes")
    last_seen = s.last_face_time
    s.update_face(False, last_seen + config.FACE_GRACE_SEC + 0.01)
    s.update_eyes(CLOSED, last_seen + config.FACE_GRACE_SEC + 0.02)   # old code path fed eyes
    assert s.sleep_active is False, "no eye update during face loss may re-raise sleep"
    s.update_face(False, last_seen + config.FACE_GRACE_SEC + 1.0)
    assert s.face_lost is True and s.sleep_active is False


def test_update_eyes_with_none_clears_face_state():
    s = new_state()
    feed(s, CLOSED, config.SLEEP_DURATION_SEC, "eyes")
    assert s.sleep_active is True
    t = s._clock["eyes"] + STEP
    assert s.update_eyes(None, t) is None
    assert s.sleep_active is False and s.drowsy_active is False


def test_face_reappearing_does_not_resurrect_alert():
    s = new_state()
    feed(s, CLOSED, config.SLEEP_DURATION_SEC, "eyes")
    last_seen = s.last_face_time
    s.update_face(False, last_seen + config.FACE_GRACE_SEC + 0.01)
    t = last_seen + config.FACE_GRACE_SEC + 0.5
    s.update_face(True, t)
    s.update_eyes(CLOSED, t)
    assert s.sleep_active is False, "old eye evidence must not carry over a face loss"


def test_no_alert_flags_without_ever_seeing_a_face():
    s = DriverState()
    for t in (0.0, config.FACE_GRACE_SEC + 1.0, config.FACE_GRACE_SEC + 5.0):
        s.update_face(False, t)
    assert not s.sleep_active and not s.drowsy_active and not s.yawn_active
    assert s.face_lost is False, "no face ever seen is not 'lost'"


def test_yawn_latches_and_recovers():
    s = DriverState()
    s.update_face(True, 50.0)
    t = 50.0
    for _ in range(6):
        s.update_yawn(YAWN_MAR, t)
        t += STEP
    s.update_yawn(YAWN_MAR, t + config.YAWN_DURATION_SEC)
    assert s.yawn_active is True
    t += config.YAWN_DURATION_SEC + STEP
    s.update_yawn(NORMAL_MAR, t)
    assert s.yawn_active is True, "one normal frame must not clear a yawn"
    cleared = False
    for _ in range(40):
        t += STEP
        s.update_yawn(NORMAL_MAR, t)
        if not s.yawn_active:
            cleared = True
            break
    assert cleared, "yawn must clear after RECOVERY_DURATION_SEC of a closed mouth"


def test_distraction_latches_and_recovers():
    s = DriverState()
    s.update_face(True, 50.0)
    s.update_distraction(True, 50.0)
    s.update_distraction(True, 50.0 + config.DISTRACTION_DURATION_SEC)
    assert s.distraction_active is True
    s.update_distraction(False, 50.0 + config.DISTRACTION_DURATION_SEC + 0.01)
    assert s.distraction_active is True
    s.update_distraction(False, 50.0 + config.DISTRACTION_DURATION_SEC + 0.01 + config.RECOVERY_DURATION_SEC)
    assert s.distraction_active is False


def test_reconnect_reset_clears_everything():
    s = new_state()
    feed(s, CLOSED, config.SLEEP_DURATION_SEC + 2 * STEP, "eyes")
    s.reset_for_camera_reconnect()
    assert not s.sleep_active and not s.drowsy_active and not s.yawn_active
    assert len(s.eye_smoother.values) == 0 and len(s.mar_smoother.values) == 0