"""MediaPipe FaceLandmarker wrapper."""
import os
import tempfile
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config
from geometry import eye_aspect_ratio, face_bbox_from_landmarks, landmarks_to_px, mouth_aspect_ratio
from head_pose import euler_degrees_from_matrix


class FaceFrameResult:
    __slots__ = ("face_detected", "bbox", "head_pose_yaw_deg", "head_pose_pitch_deg",
                 "head_pose_roll_deg", "head_pose_source", "avg_ear", "mar", "error")

    def __init__(self):
        self.face_detected = False
        self.bbox = None
        self.head_pose_yaw_deg = None
        self.head_pose_pitch_deg = None
        self.head_pose_roll_deg = None
        # "matrix" when the facial transformation matrix was decoded, "none" otherwise.
        self.head_pose_source = "none"
        self.avg_ear = None
        self.mar = None
        self.error = None


def _atomic_download(url, destination):
    destination = os.path.abspath(destination)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".face_model_", suffix=".download", dir=os.path.dirname(destination))
    os.close(fd)
    try:
        with urllib.request.urlopen(url, timeout=30) as response, open(temp_path, "wb") as out:
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
            if total < 500_000:
                raise RuntimeError("Downloaded face model is unexpectedly small/corrupt")
        os.replace(temp_path, destination)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def ensure_model_downloaded():
    if os.path.exists(config.FACE_MODEL_PATH):
        return
    if not config.AUTO_DOWNLOAD_FACE_MODEL:
        raise FileNotFoundError(f"Missing face model: {config.FACE_MODEL_PATH}")
    print("[INFO] Downloading MediaPipe face_landmarker.task...")
    _atomic_download(config.FACE_MODEL_URL, config.FACE_MODEL_PATH)


def create_face_landmarker():
    ensure_model_downloaded()
    base_options = mp_python.BaseOptions(
        model_asset_path=str(config.FACE_MODEL_PATH),
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        # Required for real yaw/pitch/roll. Disable only if CPU cost becomes a problem.
        output_facial_transformation_matrixes=True,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def new_mp_image(rgb_frame):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)


def analyze_frame(face_landmarker, mp_image, timestamp_ms, w, h):
    result = FaceFrameResult()
    try:
        mesh = face_landmarker.detect_for_video(mp_image, timestamp_ms)
    except Exception as exc:
        result.error = str(exc)
        return result
    if not mesh.face_landmarks:
        return result

    pts = landmarks_to_px(mesh.face_landmarks[0], w, h)
    bbox = face_bbox_from_landmarks(pts, w, h)
    result.face_detected = bbox[2] > 0 and bbox[3] > 0
    result.bbox = bbox
    if not result.face_detected:
        return result

    # Head pose: prefer MediaPipe's facial transformation matrix (a 4x4 row-major numpy
    # array). If it is missing (older .task bundle, or a model without the geometry
    # pipeline) yaw stays None and the caller treats the pose as unknown rather than
    # guessing from the nose.
    matrices = getattr(mesh, "facial_transformation_matrixes", None)
    if matrices is not None and len(matrices) > 0:
        angles = euler_degrees_from_matrix(matrices[0])
        if angles is not None:
            result.head_pose_yaw_deg, result.head_pose_pitch_deg, result.head_pose_roll_deg = angles
            result.head_pose_source = "matrix"

    result.avg_ear = (
        eye_aspect_ratio(pts, config.LEFT_EYE_EAR_IDX)
        + eye_aspect_ratio(pts, config.RIGHT_EYE_EAR_IDX)
    ) / 2.0
    result.mar = mouth_aspect_ratio(pts)
    return result