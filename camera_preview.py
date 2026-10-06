"""Simple one-window camera diagnostic preview."""
import argparse
import cv2
import numpy as np
from camera_utils import open_camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--backend", choices=["dshow", "msmf", "default"], default=None)
    args = parser.parse_args()
    cap = open_camera(args.camera_index, args.backend)
    if cap is None:
        return 1
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if ret and frame is not None and frame.size and frame.ndim == 3:
                cv2.putText(frame, "Camera Preview", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Camera Preview", frame)
            else:
                canvas = blank.copy()
                cv2.putText(canvas, "CAMERA FRAME UNAVAILABLE", (60, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Camera Preview", canvas)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())