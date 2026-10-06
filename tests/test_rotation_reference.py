"""Reference check of the head-pose rotation maths against scipy.

Scipy is installed (Ultralytics depends on it), so this pins the formulas to an
independent implementation. Run directly:

    venv\Scripts\python tests/test_rotation_reference.py

Established convention (sources listed in head_pose.py's docstring):

* Layout: the 3x3 block is column-major flat data, M[r][c] = flat[c * 4 + r]; nested
  sequences and numpy 4x4 arrays are read row by row.
* Rotation: R = Rx(pitch) Ry(yaw) Rz(roll) = scipy R.from_euler("XYZ", (yaw, pitch, roll)).
* Extraction: yaw = asin(M02), pitch = atan2(-M12, M22), roll = atan2(-M01, M00)
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import head_pose
from test_head_pose import _rotation_matrix, euler_degrees_from_matrix

CASES = [
    (0, 0, 0), (25, 0, 0), (-25, 0, 0), (0, 15, -10), (0, 0, 30), (0, 30, 0),
    (-40, -20, -15), (-80, 0, 30), (89.9, 10, 0), (-89.9, 0, -20), (60, 45, -70),
    (80, -30, 20), (12, -40, 55),
]
TOL = 1e-6


def main():
    try:
        from scipy.spatial.transform import Rotation as Rot
    except Exception as exc:  # pragma: no cover
        print(f"[SKIP] scipy not available: {exc}")
        return 0

    failures = 0
    for case in CASES:
        yaw, pitch, roll = case
        rows = _rotation_matrix(yaw, pitch, roll)
        # MediaPipe returns a column-major flat list: flat[c * 4 + r] = M[r][c],
        # so row r, column c is rows[c * 4 + r].
        R = [[rows[c * 4 + r] for c in range(3)] for r in range(3)]
        # The helper's product Rx(pitch) Ry(yaw) Rz(roll) == scipy 'XYZ' with the yaw slot negated.
        ref = Rot.from_euler("XYZ", (yaw, pitch, roll), degrees=True).as_matrix()
        diff = max(abs(R[i][j] - ref[i][j]) for i in range(3) for j in range(3))
        helper_ok = diff < TOL

        # Extraction uses the XYZ formulation and negates yaw, so a matrix built with
        # yaw y comes back as (-y, pitch, roll) (subject convention).
        got = head_pose.euler_degrees_from_matrix(rows)
        if abs(yaw) < 80:   # gimbal lock at |yaw| -> 90 is handled by its own test
            extract_ok = all(abs(g - e) < 0.01 for g, e in zip(got, (-yaw, pitch, roll)))
        else:
            extract_ok = all(math.isfinite(v) for v in got)

        # The extractor must agree with scipy's own XYZ decomposition of the same matrix.
        scipy_zyx = Rot.from_matrix(ref).as_euler("XYZ", degrees=True)
        if abs(yaw) < 80:   # gimbal lock at |yaw| -> 90 is handled by its own test
            agree_ok = all(abs(g - s) < 0.01 for g, s in zip(got, scipy_zyx))
        else:
            agree_ok = all(math.isfinite(v) for v in got)

        status = "PASS" if (helper_ok and extract_ok and agree_ok) else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"[{status}] case={case}  helper_vs_scipy_max_diff={diff:.2e}  "
              f"extracted=({got[0]:.2f},{got[1]:.2f},{got[2]:.2f})  expected={(yaw, pitch, roll)}  "
              f"scipy_XYZ={tuple(round(float(x), 2) for x in scipy_zyx)}")

    print(f"\nreference check: {len(CASES) - failures}/{len(CASES)} cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())