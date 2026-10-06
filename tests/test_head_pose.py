"""Head-pose tests: matrix -> Euler, side mapping, classification, and the main.py branch.

No pytest fixtures and no MediaPipe import: the matrix maths and the mapping are
exercised with hand-built rotations, and main.classify_head_pose() is called with a
duck-typed face object. Real-world yaw accuracy on a human face is NOT covered here.
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from head_pose import classify_yaw, euler_degrees_from_matrix, subject_side_for_image_yaw
from main import classify_head_pose

THRESH = config.HEAD_YAW_THRESHOLD_DEG

# 480 landmarks spread over the frame so face_bbox_from_landmarks yields a real box
LANDMARKS = [[type("L", (), {"x": (i % 24) / 24.0, "y": (i // 24) / 20.0})() for i in range(480)]]


# ---------------------------------------------------------------- matrix maths
def _rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    """Row-major 4x4 for R = Rx(pitch) * Ry(yaw) * Rz(roll) acting on column vectors.

    Rows are those of the product Rx(pitch) Ry(yaw) Rz(roll) (scipy's
    R.from_euler("XYZ", (yaw, pitch, roll))); the 3x3 block is emitted as a
    column-major flat list inside the 4x4, mirroring how MediaPipe hands over
    pose_transform_matrix: packed_data is column-major, and effect_renderer.cc
    (ExtractMatrixAsOpenGLArray) only transposes when layout == ROW_MAJOR.
    head_pose.euler_degrees_from_matrix() inverts this composition and negates
    yaw, so a matrix built with yaw y comes out as (-y, pitch, roll). Both are
    pinned against scipy in tests/test_rotation_reference.py.
    """
    ya, pa, ra = (math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg))
    cy, sy = math.cos(ya), math.sin(ya)
    cp, sp = math.cos(pa), math.sin(pa)
    cr, sr = math.cos(ra), math.sin(ra)
    rows = [
        [cy * cr, -cy * sr, sy],
        [cp * sr + sp * sy * cr, cp * cr - sp * sy * sr, -sp * cy],
        [sp * sr - cp * sy * cr, sp * cr + cp * sy * sr, cp * cy],
    ]
    flat = [rows[r][c] for c in range(3) for r in range(3)]
    return [flat[0], flat[1], flat[2], 0.0, flat[3], flat[4], flat[5], 0.0,
            flat[6], flat[7], flat[8], 0.0, 0.0, 0.0, 0.0, 1.0]


def _near(a, b, tol=0.5):
    return abs(a - b) <= tol


def test_neutral_matrix_gives_zero_angles():
    yaw, pitch, roll = euler_degrees_from_matrix(_rotation_matrix(0, 0, 0))
    assert _near(yaw, 0) and _near(pitch, 0) and _near(roll, 0)


def test_right_yaw_subject_gives_negative_yaw():
    # The driver turns to their own right -> yaw_deg negative by this convention.
    yaw, pitch, roll = euler_degrees_from_matrix(_rotation_matrix(-25, 0, 0))
    assert _near(yaw, 25), yaw
    assert _near(pitch, 0) and _near(roll, 0)


def test_left_yaw_subject_gives_positive_yaw():
    yaw, pitch, roll = euler_degrees_from_matrix(_rotation_matrix(25, 0, 0))
    assert _near(yaw, -25), yaw


def test_pitch_and_roll_signs():
    # pitch = atan2(-R12, R22) and roll = atan2(-R01, R00) of the Rx*Ry*Rz matrix
    yaw, pitch, roll = euler_degrees_from_matrix(_rotation_matrix(0, 15, -10))
    assert _near(yaw, 0), yaw
    assert _near(pitch, 15) and _near(roll, -10), (pitch, roll)


def test_round_trip_over_a_grid_of_angles():
    for yaw in (-40, -12, 0, 12, 40):
        for pitch in (-20, 0, 20):
            for roll in (-15, 0, 15):
                if abs(yaw) < 80:      # YXZ degenerates only at |yaw| -> 90 (gimbal lock)
                    y, p, r = euler_degrees_from_matrix(_rotation_matrix(yaw, pitch, roll))
                    assert _near(y, -yaw, 0.01) and _near(p, pitch, 0.01) and _near(r, roll, 0.01), (yaw, pitch, roll, y, p, r)


def test_gimbal_lock_returns_no_nan():
    yaw, pitch, roll = euler_degrees_from_matrix(_rotation_matrix(89.9, 10, 0))
    assert all(math.isfinite(v) for v in (yaw, pitch, roll))


def test_nested_row_major_matrix_is_accepted():
    flat = _rotation_matrix(-20, 0, 0)
    rows = [[flat[c * 4 + r] for c in range(4)] for r in range(4)]
    assert _near(euler_degrees_from_matrix(rows)[0], -20)


def test_missing_or_bad_matrix_returns_none():
    assert euler_degrees_from_matrix(None) is None
    assert euler_degrees_from_matrix([1, 2, 3]) is None


# ---------------------------------------------------------------- config
def test_config_has_documented_yaw_threshold():
    assert isinstance(config.HEAD_YAW_THRESHOLD_DEG, (int, float))
    assert 15.0 <= config.HEAD_YAW_THRESHOLD_DEG <= 40.0
    assert config.DRIVER_SIDE in ("left", "right")
    assert isinstance(config.FLIP_FRAME, bool)
    # The old nose-ratio thresholds are no longer used anywhere.
    assert not hasattr(config, "HEAD_LEFT_THRESHOLD")
    assert not hasattr(config, "HEAD_RIGHT_THRESHOLD")


# ---------------------------------------------------------------- classification
class _Face:
    def __init__(self, detected, yaw=None, source="matrix"):
        self.face_detected = detected
        self.bbox = (0, 0, 100, 100) if detected else None
        self.head_pose_yaw_deg = yaw
        self.head_pose_source = source
        self.avg_ear = 0.3
        self.mar = 0.2


def test_neutral_head_is_not_distracting():
    assert classify_head_pose(_Face(True, 0.0), "right", True) == ("Center Focus", False)
    assert classify_head_pose(_Face(True, 5.0), "right", True) == ("Center Focus", False)
    assert classify_head_pose(_Face(True, None, "none"), "right", True) == ("Center Focus", False)


def test_left_and_right_yaw_are_both_distracting():
    for yaw in (-30.0, 30.0):
        label, distracted = classify_head_pose(_Face(True, yaw), "right", True)
        assert distracted is True, yaw
        assert label in ("Looking Left", "Looking Right")


def test_rhd_mapping_right_yaw_is_road_side():
    # Driver's own right is the road side in an RHD car.
    label, distracted = classify_head_pose(_Face(True, -30.0), "right", True)
    assert distracted is True
    assert label == "Looking Right"


def test_rhd_mapping_left_yaw():
    label, distracted = classify_head_pose(_Face(True, 30.0), "right", True)
    assert distracted is True
    assert label == "Looking Left"


def test_lhd_mapping_is_mirrored():
    # LHD car: the road is on the driver's left.
    assert classify_head_pose(_Face(True, 30.0), "left", True) == ("Looking Right", True)
    assert classify_head_pose(_Face(True, -30.0), "left", True) == ("Looking Left", True)


def test_flip_frame_swaps_the_label_but_not_the_distraction():
    flipped = classify_head_pose(_Face(True, -30.0), "right", True)
    unflipped = classify_head_pose(_Face(True, -30.0), "right", False)
    assert flipped == ("Looking Right", True)
    assert unflipped == ("Looking Left", True)
    assert flipped[1] == unflipped[1] is True


def test_no_face_is_never_distracting():
    assert classify_head_pose(_Face(False, None), "right", True) == ("Center Focus", False)
    # Even if a stale yaw value were present, no face means no distraction.
    assert classify_head_pose(_Face(False, -40.0), "right", True) == ("Center Focus", False)


def test_threshold_boundary():
    just_below = classify_head_pose(_Face(True, THRESH - 0.5), "right", True)
    at = classify_head_pose(_Face(True, THRESH), "right", True)
    just_above = classify_head_pose(_Face(True, THRESH + 0.5), "right", True)
    assert just_below == ("Center Focus", False)
    assert at == ("Looking Left", True)
    assert just_above == ("Looking Left", True)
    # symmetric
    assert classify_head_pose(_Face(True, -THRESH), "right", True) == ("Looking Right", True)


def test_smoothing_does_not_hide_a_sustained_turn():
    # one-sample smoother: first sample has no history yet
    from temporal_state import FeatureSmoother
    sm = FeatureSmoother(1)
    assert sm.update(-30.0) == -30.0
    # five-sample smoother: five frames of a held turn converge to the held yaw
    sm5 = FeatureSmoother(5)
    vals = [sm5.update(-30.0) for _ in range(5)]
    assert _near(vals[-1], -30.0)
    assert classify_yaw(vals[-1], THRESH, "right", True) == ("Looking Right", True)


def test_subject_side_mapping_table():
    # subject right-hand yaw (negative) -> road side is right in an RHD car -> image right when mirrored
    assert subject_side_for_image_yaw(-30, "right", True) == "left"
    assert subject_side_for_image_yaw(30, "right", True) == "right"
    # without the mirror the same yaw reads on the other image side
    assert subject_side_for_image_yaw(-30, "right", False) == "left"
    assert subject_side_for_image_yaw(30, "right", False) == "right"
    # LHD car: road side is the driver's left
    assert subject_side_for_image_yaw(30, "left", False) == "left"
    assert subject_side_for_image_yaw(-30, "left", False) == "right"


def test_smoothing_a_real_mixed_sequence_still_crosses_threshold():
    from temporal_state import DriverState
    import time
    state = DriverState()
    now = time.monotonic()
    state.update_face(True, now)
    for _ in range(6):                     # neutral frames
        state.pose_smoother.update(0.0)
    for _ in range(6):                     # then a held left turn
        state.pose_smoother.update(30.0)
    yaw = state.pose_smoother.update(30.0)
    assert abs(yaw) >= THRESH
    label, distracted = classify_yaw(yaw, THRESH, "right", True)
    assert distracted and label == "Looking Left"


# ---------------------------------------------------------------- integration
def test_analyze_frame_without_transformation_matrices_is_safe():
    """When the landmarker returns no matrix, yaw stays None and nothing distracts."""
    import numpy as np
    import face_analyzer
    from geometry import landmarks_to_px

    class _FakeResult:
        def __init__(self):
            self.face_landmarks = LANDMARKS
            self.facial_transformation_matrixes = []

    class _FakeLandmarker:
        def detect_for_video(self, image, ts):
            return _FakeResult()

    res = face_analyzer.analyze_frame(_FakeLandmarker(), None, 1, 640, 480)
    assert res.face_detected is True
    assert res.head_pose_yaw_deg is None and res.head_pose_source == "none"
    assert res.avg_ear is not None and res.mar is not None
    label, distracted = classify_head_pose(res, "right", True)
    assert (label, distracted) == ("Center Focus", False)
    # smoothing a None pose yields None (main.py then holds the last label)
    from temporal_state import FeatureSmoother
    assert FeatureSmoother(5).update(res.head_pose_yaw_deg) is None
    assert np is not None and landmarks_to_px is not None


def test_analyze_frame_with_transformation_matrices_fills_yaw():
    import math
    import face_analyzer

    class _FakeResult:
        def __init__(self, matrix):
            self.face_landmarks = LANDMARKS
            self.facial_transformation_matrixes = [matrix]

    class _FakeLandmarker:
        def __init__(self, matrix):
            self.matrix = matrix

        def detect_for_video(self, image, ts):
            return _FakeResult(self.matrix)

    matrix = _rotation_matrix(-30, 0, 0)
    res = face_analyzer.analyze_frame(_FakeLandmarker(matrix), None, 1, 640, 480)
    assert res.face_detected is True
    assert res.head_pose_source == "matrix"
    assert res.head_pose_yaw_deg is not None and math.isfinite(res.head_pose_yaw_deg)
    assert _near(res.head_pose_yaw_deg, 30.0, 0.1), res.head_pose_yaw_deg   # subject yaw: left turn positive
    assert res.head_pose_pitch_deg is not None and res.head_pose_roll_deg is not None
    label, distracted = classify_head_pose(res, "right", True)
    assert distracted is True and label == "Looking Left"


def test_landmarker_requests_transformation_matrices():
    """create_face_landmarker() must enable the matrix output (option-level check)."""
    src = open(face_analyzer_path(), encoding="utf-8").read()
    assert "output_facial_transformation_matrixes=True" in src
    assert "head_pose_x_ratio" not in src


def face_analyzer_path():
    from pathlib import Path
    return Path(__file__).resolve().parents[1] / "face_analyzer.py"