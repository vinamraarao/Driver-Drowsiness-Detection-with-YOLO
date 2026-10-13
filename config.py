"""Central configuration for the driver monitoring system."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FACE_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
YOLO_MODEL_PATH = MODELS_DIR / "yolo11n.pt"
YOLO_MODEL_NAME = "yolo11n.pt"
LOG_FILE = BASE_DIR / "event_history_log.csv"

# ---------------- Camera ----------------
# Camera index used when --camera-index is not given (Windows). Override per machine here.
DEFAULT_CAMERA_INDEX = 1
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30
CAMERA_SCAN_MAX_INDEX = 4
CAMERA_BAD_FRAME_LIMIT = 12
CAMERA_RECONNECT_RETRIES = 4
CAMERA_OPEN_DELAY_SEC = 0.25

# ---------------- Display ----------------
WINDOW_NAME = "Driver Monitoring System"
FULLSCREEN = True
# Mirror the camera image so the driver appears on the expected side. Only affects
# the left/right label of the head-pose alert, never the distraction decision.
FLIP_FRAME = True
DRIVER_SIDE = "right"  # "right" = right-hand drive (India); "left" = left-hand drive

# ---------------- YOLO ----------------
YOLO_INFER_SCALE = 1.0
YOLO_INTERVAL_SEC = 0.08
YOLO_IMAGE_SIZE = 640
# "auto" -> GPU when torch reports CUDA, else CPU. Set to "cpu", "0", "1", ... to force.
YOLO_DEVICE = "auto"
YOLO_FORCE_CPU = False  # True disables GPU use even when CUDA is available
YOLO_PERSON_CONF = 0.45
YOLO_PHONE_CONF = 0.25
YOLO_PERSON_CONF_NIGHT = 0.40
YOLO_PHONE_CONF_NIGHT = 0.22
YOLO_STALE_AFTER_SEC = 0.90
YOLO_ERROR_LOG_INTERVAL_SEC = 5.0

# COCO labels used by the pretrained detector.
COCO_PERSON_CLASS = 0
COCO_CELL_PHONE_CLASS = 67

# Phone confirmation is based on NEW YOLO results, not camera frames.
PHONE_CONTAINMENT_THRESHOLD = 0.30
PHONE_CENTER_INSIDE_THRESHOLD = 0.10
PHONE_CONFIRMATION_LIMIT = 3
PHONE_CLEAR_CONFIRMATIONS = 2
PHONE_RESULT_TIMEOUT_SEC = 1.10
DRIVER_BOX_EXPANSION = 0.08

# ---------------- Face / drowsiness ----------------
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.55

# ---------------- Driver calibration ----------------
# Startup baseline collection (seconds). Kept short so the app does not block.
CALIBRATION_DURATION_SEC = 2.0
CALIBRATION_MIN_SAMPLES = 10
# Threshold offset from the measured baseline, in standard deviations.
CALIBRATION_EAR_STDDEV_OFFSET = 3.0   # closed = mean - 3*sigma
CALIBRATION_MAR_STDDEV_OFFSET = 3.0   # yawn  = mean + 3*sigma
# Hard bounds for calibrated thresholds (prevents degenerate values).
EAR_CALIB_MIN = 0.05
EAR_CALIB_MAX = 0.45
MAR_CALIB_MIN = 0.10
MAR_CALIB_MAX = 2.00
# Set False to skip the startup calibration and always use the static thresholds.
ENABLE_CALIBRATION = True

# ---------------- PERCLOS ----------------
# Rolling percentage-of-eyes-closed window (seconds). PERCLOS is the percentage
# of time the eyes were closed within this window; > threshold is drowsy.
PERCLOS_WINDOW_SEC = 3.0
PERCLOS_DROWSY_THRESHOLD = 80.0  # percent
# Head yaw from MediaPipe's facial transformation matrix, in driver-centric coordinates:
# positive = the driver turned toward their own right, negative = their own left (the
# sign is produced by head_pose.euler_degrees_from_matrix). Only |yaw| is compared with
# this threshold, so the sign only selects the Left/Right wording of the alert; the side
# mapping (DRIVER_SIDE x FLIP_FRAME) lives in head_pose.subject_side_for_image_yaw.
# Starting value 25 deg: a deliberate glance at the centre console or the passenger
# seat is usually 20-30 deg of yaw, while normal road monitoring stays under ~15 deg.
# NOT calibrated on this cabin or driver yet (no labelled face data); raise to 30 if
# false alerts appear at junctions, lower to 20 if slow side glances go unnoticed.
HEAD_YAW_THRESHOLD_DEG = 25.0
# Sign guard: head_pose.euler_degrees_from_matrix returns yaw > 0 for a turn toward the
# driver's own right. Set False only if classify_face.py shows the opposite on a real face.
HEAD_YAW_POSITIVE_IS_SUBJECT_RIGHT = True
FEATURE_SMOOTHING_SAMPLES = 5

# These preserve the old ~30-FPS behavior while making it FPS independent.
DROWSY_DURATION_SEC = 20.0 / CAM_FPS
SLEEP_DURATION_SEC = 45.0 / CAM_FPS
YAWN_DURATION_SEC = 20.0 / CAM_FPS
DISTRACTION_DURATION_SEC = 30.0 / CAM_FPS
# An eye must stay open for this long before drowsiness/sleep clear. 5 frames at ~50 ms,
# long enough to ride out a 100-150 ms blink.
RECOVERY_DURATION_SEC = 0.25
# How long the face may be missing before every face-derived alert is dropped.
FACE_GRACE_SEC = 0.20

# ---------------- Hands ----------------
HAND_MIN_DETECTION_CONF = 0.50
HAND_MIN_TRACKING_CONF = 0.50
HAND_MAX_COUNT = 2
ENABLE_HAND_ALERTS = False   # When False the hand landmarker is not run at all (18-23 ms/frame saved).
HAND_INTERVAL_SEC = 0.20    # Lower the hand rate to 5 Hz.
HANDS_NOT_VISIBLE_SEC = 2.0

# ---------------- Steering Wheel Hand Presence ----------------
# Steering wheel region as [x1, y1, x2, y2] normalized coordinates (0.0-1.0)
# Default: approximate center region where hands would be on wheel
STEERING_WHEEL_ROI = [0.3, 0.3, 0.7, 0.7]  # [x_min, y_min, x_max, y_max]
HAND_OFF_WHEEL_WARNING_1HAND = 30.0   # seconds for 1 hand off wheel warning
HAND_OFF_WHEEL_HIGH_RISK_1HAND = 60.0 # seconds for 1 hand off wheel high risk
HAND_OFF_WHEEL_WARNING_2HAND = 2.0    # seconds for 2 hands off wheel warning
HAND_OFF_WHEEL_CRITICAL_2HAND = 5.0   # seconds for 2 hands off wheel critical
HAND_OFF_WHEEL_CONFIRMATION_SEC = 0.5 # seconds of continuous state required to trigger
HAND_OFF_WHEEL_RECOVERY_SEC = 1.0     # seconds of correct state to reset timers

# ---------------- Night vision ----------------
ENABLE_NIGHT_VISION = True
# Recalibrated from measured native (un-enhanced) mean luma of this cabin:
# index 1 ~16, index 0 ~32 for the dark cabin; normal daylight cabin ~120-160.
# Hysteresis keeps night mode off in a normal cabin and on in the dark one.
NIGHT_MODE_ENTER_LUMA = 45.0
NIGHT_MODE_EXIT_LUMA = 60.0
BRIGHTNESS_SAMPLE_DOWNSCALE = 0.25
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = (8, 8)
GAMMA_NIGHT = 1.8
# 0 disables cv2.fastNlMeansDenoising, which costs 180-350 ms per 640x480 frame
# and cannot run on the real-time path. CLAHE + gamma + unsharp is ~3 ms.
DENOISE_H_NIGHT = 0
SHARPEN_NIGHT = True
NIGHT_VISION_DISPLAY_TINT = True
NIGHT_CAM_GAIN = None
NIGHT_CAM_EXPOSURE = None
DAY_CAM_GAIN = None
DAY_CAM_EXPOSURE = None

# ---------------- Voice ----------------
VOICE_ENABLED = True
VOICE_RATE = 165
VOICE_VOLUME = 1.0
VOICE_QUEUE_SIZE = 3
VOICE_DEFAULT_COOLDOWN_SEC = 4.0

VOICE_COOLDOWNS = {
    "sleep": 4.0,
    "sleep_phone": 4.0,
    "phone": 4.0,
    "drowsiness": 5.0,
    "yawn": 6.0,
    "distraction": 5.0,
    "hands": 6.0,
}

# ---------------- Model download ----------------
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
AUTO_DOWNLOAD_FACE_MODEL = True
AUTO_DOWNLOAD_YOLO_MODEL = True

# ---------------- Face landmark indices ----------------
LEFT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
MOUTH_TOP_IDX = [82, 13, 312]
MOUTH_BOTTOM_IDX = [87, 14, 317]
MOUTH_LEFT_IDX = 78
MOUTH_RIGHT_IDX = 308
NOSE_TIP_IDX = 1

# ---------------- HUD ----------------
HUD_HEIGHT = 58
HUD_X_OFFSET = 10
HUD_Y_OFFSET = 10
ALERT_BANNER_HEIGHT = 54

# ---------------- Evaluation target ----------------
TARGET_ACCURACY = 0.90
TARGET_ACCURACY_HIGH = 0.95
