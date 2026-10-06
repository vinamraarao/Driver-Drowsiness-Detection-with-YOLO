"""Head-pose helpers: transformation matrix -> Euler angles, and side mapping.

Pure math, no I/O, no MediaPipe import, so it can be unit tested directly.
"""
import math

import config


def euler_degrees_from_matrix(matrix):
    """MediaPipe facial transformation matrix -> (yaw, pitch, roll) degrees.

    Layout: the 3x3 rotation block is read as column-major flat data, M[r][c] =
    flat[c * 4 + r]. That is how MediaPipe hands the matrix over: the proto stores it
    column-major (framework/formats/matrix_data.proto default COLUMN_MAJOR), the Python
    ctypes layer transposes it into a row-major numpy array
    (mediapipe/tasks/python/components/containers/matrix_c.py), and the C++ renderer
    transposes it back to column-major for OpenGL (effect_renderer.cc,
    ExtractMatrixAsOpenGLArray). A nested sequence or a numpy 4x4 array is read row by
    row, which yields the same matrix for either layout.

    Convention: the matrix maps the canonical face model into MediaPipe's camera space.
    The canonical model and that camera share the OpenGL basis: +X to the face's own
    right, +Y up, +Z out of the face toward the viewer. The rotation is

        R = Rx(pitch) Ry(yaw) Rz(roll)          (= scipy from_euler("XYZ", (yaw, pitch, roll)))

    so the extraction is

        yaw   = asin(M[0][2])
        pitch = atan2(-M[1][2], M[2][2])
        roll  = atan2(-M[0][1], M[0][0])

    yaw is returned negated, so it is the driver's own yaw: positive means the driver
    turned toward their own right. Pitch and roll are camera angles: positive pitch is
    chin up, positive roll tilts the head toward the image's left.
    tests/test_rotation_reference.py pins all three against scipy over a grid of angles.
    """
    if matrix is None:
        return None
    try:
        r = _as_3x3_rows(matrix)
    except Exception:
        return None

    # Extract yaw, pitch, roll for R = Rx(pitch) Ry(yaw) Rz(roll)
    # Format: r[i][j] = element at row i, column j
    yaw = math.asin(r[0][2])
    pitch = math.atan2(-r[1][2], r[2][2])
    roll = math.atan2(-r[0][1], r[0][0])

    # Return yaw negated to get driver's own yaw: positive means the driver turned toward their own right
    return (-math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def _as_3x3_rows(matrix):
    """Return the 3x3 rotation block of a 4x4 matrix as rows, or raise ValueError."""
    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None and isinstance(matrix, np.ndarray):
        arr = np.asarray(matrix, dtype=float)
        if arr.shape != (4, 4):
            raise ValueError("expected a 4x4 matrix")
        return [[float(arr[r][c]) for c in range(3)] for r in range(3)]
    flat = list(matrix)
    if len(flat) == 16 and all(isinstance(v, (int, float)) for v in flat):
        return [[float(flat[c * 4 + r]) for c in range(3)] for r in range(3)]  # column-major flat
    if len(flat) == 4 and all(len(list(row)) == 4 for row in flat):
        # For nested rows input (row-major 4x4), extract 3x3 in column-major format
        return [[float(flat[r][c]) for r in range(3)] for c in range(3)]  # nested rows -> column-major
    raise ValueError("unsupported matrix shape")


def _as_row_major_4x4(matrix):
    """Return the 3x3 rotation block of a 4x4 matrix as rows, or raise ValueError."""
    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None and isinstance(matrix, np.ndarray):
        arr = np.asarray(matrix, dtype=float)
        if arr.shape != (4, 4):
            raise ValueError("expected a 4x4 matrix")
        return [[float(arr[r][c]) for c in range(3)] for r in range(3)]
    flat = list(matrix)
    if len(flat) == 16 and all(isinstance(v, (int, float)) for v in flat):
        return [[float(flat[r * 4 + c]) for c in range(3)] for r in range(3)]  # row-major flat
    if len(flat) == 4 and all(len(list(row)) == 4 for row in flat):
        return [[float(v) for v in list(row)[:3]] for row in flat]  # nested rows
    raise ValueError("unsupported matrix shape")


def subject_side_for_image_yaw(yaw_deg, driver_side="right", flip_frame=True):
    """Which way the driver is turned in their own frame, from a signed driver yaw.

    ``yaw_deg`` is the driver's own yaw: positive = turned toward their own right (the
    sign is chosen by ``config.HEAD_YAW_POSITIVE_IS_SUBJECT_RIGHT``). ``driver_side`` is
    the side the driver sits on, which for right-hand traffic is the side the road is on:
    looking toward that side means looking at the road. ``flip_frame`` is the display
    mirror applied in main.py; it is accepted for signature compatibility and does not
    change this subject-centric answer (classify_yaw applies the mirror to the wording).
    """
    if not getattr(config, "HEAD_YAW_POSITIVE_IS_SUBJECT_RIGHT", True):
        yaw_deg = -yaw_deg
    turned_to_own_right = yaw_deg > 0
    road_is_on_own_right = driver_side != "left"   # RHD car: road on the driver's right
    looking_at_road = turned_to_own_right == road_is_on_own_right
    return "right" if looking_at_road else "left"


def classify_yaw(yaw_deg, threshold_deg=None, driver_side="right", flip_frame=True):
    """Return (head_pose_label, is_distracted) for a signed driver yaw.

    The distraction decision uses the absolute yaw, so it is symmetric: more than
    ``threshold_deg`` of yaw in either direction is distraction. The label follows the
    cabin mapping above; ``flip_frame`` mirrors the wording only.
    """
    if threshold_deg is None:
        threshold_deg = config.HEAD_YAW_THRESHOLD_DEG
    if yaw_deg is None:
        return "Center Focus", False
    if abs(yaw_deg) < threshold_deg:
        return "Center Focus", False
    side = subject_side_for_image_yaw(yaw_deg, driver_side, flip_frame)
    if flip_frame:
        side = "left" if side == "right" else "right"
    return f"Looking {side.capitalize()}", True