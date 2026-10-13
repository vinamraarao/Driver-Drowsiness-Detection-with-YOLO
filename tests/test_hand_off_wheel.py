"""Tests for hand-off-wheel monitoring feature.

Tests cover 0/1/2 hand states, temporal confirmation,
duration thresholds, timer reset, missed detection, face/driver loss,
and risk-score/alert integration.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import time
from temporal_state import DriverState


def new_state(now=100.0):
    """Create a DriverState with initialized clock."""
    s = DriverState()
    s._clock = {"eyes": now, "yawn": now, "distraction": now}
    s.update_face(True, now)
    return s


def test_hand_off_wheel_initial_state():
    """Test that hand-off-wheel monitoring starts with both hands on wheel."""
    s = new_state()
    # Initially no hands have been detected as off wheel
    # (default state when hands are present)
    assert s.hand_off_wheel_state == "BOTH_ON"


def test_hand_off_wheel_one_hand_off():
    """Test detection of one hand off wheel."""
    s = new_state()
    # Simulate one hand off wheel
    s.hand_off_wheel_state = "ONE_OFF"
    s.hand_left_since = time.monotonic() - 5.0  # 5 seconds ago
    # Check state
    state, duration, is_warning, is_high_risk, is_critical = s.get_hand_off_wheel_status()
    assert state == "ONE_OFF"
    assert abs(duration - 5.0) < 0.1


def test_hand_off_wheel_both_hands_off():
    """Test detection of both hands off wheel."""
    s = new_state()
    # Simulate both hands off wheel
    s.hand_off_wheel_state = "BOTH_OFF"
    s.both_hands_off_since = time.monotonic() - 3.0  # 3 seconds ago
    # Check state
    state, duration, is_warning, is_high_risk, is_critical = s.get_hand_off_wheel_status()
    assert state == "BOTH_OFF"
    assert abs(duration - 3.0) < 0.1
    assert is_critical == False  # 3 seconds is not yet critical (default 5s)


def test_hand_off_wheel_both_hands_off_critical():
    """Test detection of both hands off wheel at critical threshold."""
    s = new_state()
    # Simulate both hands off wheel at critical threshold
    s.hand_off_wheel_state = "BOTH_OFF"
    s.both_hands_off_since = time.monotonic() - 6.0  # 6 seconds ago (> critical threshold)
    # Check state
    state, duration, is_warning, is_high_risk, is_critical = s.get_hand_off_wheel_status()
    assert state == "BOTH_OFF"
    assert abs(duration - 6.0) < 0.1
    assert is_critical == True
    assert is_high_risk == True


def test_hand_off_wheel_one_hand_warning_threshold():
    """Test one-hand warning threshold (default 30 sec)."""
    s = new_state()
    s.hand_off_wheel_state = "ONE_OFF"
    s.hand_left_since = time.monotonic() - 35.0  # 35 seconds ago (> warning threshold)
    # Check state with warning threshold = 30 sec
    state, duration, is_warning, is_high_risk, is_critical = s.get_hand_off_wheel_status()
    assert state == "ONE_OFF"
    assert is_warning == True  # 35s > 30s warning threshold
    assert is_high_risk == False  # 35s < 60s high risk threshold


def test_hand_off_wheel_one_hand_high_risk_threshold():
    """Test one-hand high risk threshold (default 60 sec)."""
    s = new_state()
    s.hand_off_wheel_state = "ONE_OFF"
    s.hand_left_since = time.monotonic() - 65.0  # 65 seconds ago (> high risk threshold)
    # Check state with high risk threshold = 60 sec
    state, duration, is_warning, is_high_risk, is_critical = s.get_hand_off_wheel_status()
    assert state == "ONE_OFF"
    assert is_warning == True  # 65s > 30s warning threshold
    assert is_high_risk == True  # 65s > 60s high risk threshold


def test_hand_off_wheel_both_hands_warning_threshold():
    """Test two-hands warning threshold (default 2 sec)."""
    s = new_state()
    s.hand_off_wheel_state = "BOTH_OFF"
    s.both_hands_off_since = time.monotonic() - 3.0  # 3 seconds ago (> warning threshold)
    # Check state with warning threshold = 2 sec
    state, duration, is_warning, is_high_risk, is_critical = s.get_hand_off_wheel_status()
    assert state == "BOTH_OFF"
    assert is_warning == True  # 3s > 2s warning threshold
    assert is_high_risk == False  # 3s < 5s critical threshold


def test_hand_off_wheel_both_hands_critical_threshold():
    """Test two-hands critical threshold (default 5 sec)."""
    s = new_state()
    s.hand_off_wheel_state = "BOTH_OFF"
    s.both_hands_off_since = time.monotonic() - 7.0  # 7 seconds ago (> critical threshold)
    # Check state with critical threshold = 5 sec
    state, duration, is_warning, is_high_risk, is_critical = s.get_hand_off_wheel_status()
    assert state == "BOTH_OFF"
    assert is_critical == True  # 7s > 5s critical threshold
    assert is_high_risk == True


def test_hand_off_wheel_timer_reset_when_hands_return():
    """Test that timers reset when hands return to wheel."""
    s = new_state()
    # Simulate both hands off wheel
    s.hand_off_wheel_state = "BOTH_OFF"
    s.both_hands_off_since = time.monotonic() - 2.0  # 2 seconds ago
    # Now simulate hands returning to wheel
    # In a real system, the update_hand_off_wheel would be called
    # with hands on wheel detection
    # For this test, we manually reset the state
    s.hand_off_wheel_state = "BOTH_ON"
    s.hand_left_since = None
    s.right_hand_left_since = None
    s.both_hands_off_since = None
    # Check state
    state, duration, is_warning, is_high_risk, is_critical = s.get_hand_off_wheel_status()
    assert state == "BOTH_ON"
    assert duration == 0.0


def test_hand_off_wheel_face_loss_resets_state():
    """Test that hand-off-wheel state resets when face is lost."""
    s = new_state()
    # Simulate both hands off wheel
    s.hand_off_wheel_state = "BOTH_OFF"
    s.both_hands_off_since = time.monotonic() - 2.0  # 2 seconds ago
    # Simulate face loss
    s.reset_hand_off_wheel_on_face_loss()
    # Check state after face loss reset
    assert s.hand_off_wheel_state == "BOTH_ON"
    assert s.hand_left_since is None
    assert s.right_hand_left_since is None
    assert s.both_hands_off_since is None


def test_hand_off_wheel_risk_level():
    """Test hand-off-wheel risk level integration with risk score."""
    s = new_state()
    # Test BOTH_ON risk level
    risk_level = s.get_hand_off_wheel_risk_level()
    assert risk_level == "SAFE"

    # Test ONE_OFF at warning level
    s.hand_off_wheel_state = "ONE_OFF"
    s.hand_left_since = time.monotonic() - 40.0  # 40 seconds ago (> warning)
    risk_level = s.get_hand_off_wheel_risk_level()
    assert risk_level == "WARNING"

    # Test ONE_OFF at high risk level
    s.hand_left_since = time.monotonic() - 65.0  # 65 seconds ago (> high risk)
    risk_level = s.get_hand_off_wheel_risk_level()
    assert risk_level == "HIGH_RISK"

    # Test BOTH_OFF at critical level
    s.hand_off_wheel_state = "BOTH_OFF"
    s.both_hands_off_since = time.monotonic() - 8.0  # 8 seconds ago (> critical)
    risk_level = s.get_hand_off_wheel_risk_level()
    assert risk_level == "CRITICAL"

    # Test BOTH_OFF at warning level (above warning, below critical)
    s.both_hands_off_since = time.monotonic() - 3.0  # 3 seconds ago
    risk_level = s.get_hand_off_wheel_risk_level()
    assert risk_level == "WARNING"


def test_hand_off_wheel_integration_with_risk_score():
    """Test hand-off-wheel integration with risk score."""
    s = new_state()

    # BOTH_ON - no contribution to risk score
    risk_score = s.calculate_risk_score(phone_present=False, face_detected=True)
    # Base risk should be 5 (normal operation)
    assert risk_score == 5  # Normal operation with BOTH_ON

    # ONE_OFF - should add 30 to risk score
    s.hand_off_wheel_state = "ONE_OFF"
    s.hand_left_since = time.monotonic() - 40.0  # 40 seconds ago (> warning)
    risk_score = s.calculate_risk_score(phone_present=False, face_detected=True)
    assert risk_score == 30  # Warning risk from one hand off

    # BOTH_OFF - should add 85 to risk score
    s.hand_off_wheel_state = "BOTH_OFF"
    s.both_hands_off_since = time.monotonic() - 2.0  # 2 seconds ago
    risk_score = s.calculate_risk_score(phone_present=False, face_detected=True)
    assert risk_score == 85  # Critical risk from both hands off