"""FPS-independent temporal state and smoothing."""
from collections import deque
import time

import config


class FeatureSmoother:
    def __init__(self, size):
        self.values = deque(maxlen=max(1, int(size)))

    def update(self, value):
        if value is None:
            return None
        self.values.append(float(value))
        return sum(self.values) / len(self.values)

    def clear(self):
        self.values.clear()


class DriverState:
    def __init__(self, thresholds=None):
        self.eye_closed_since = None
        self.eye_recovery_since = None
        self.yawn_since = None
        self.yawn_recovery_since = None
        self.distraction_since = None
        self.distraction_recovery_since = None
        self.last_face_time = None          # monotonic time of the last frame with a face
        self.face_lost = False              # True once no face has been seen for FACE_GRACE_SEC
        self._face_lost_reported = False    # guards the one-shot reset below
        self.hands_missing_since = None
        self.hands_recovery_since = None
        self.eye_smoother = FeatureSmoother(config.FEATURE_SMOOTHING_SAMPLES)
        self.mar_smoother = FeatureSmoother(config.FEATURE_SMOOTHING_SAMPLES)
        self.pose_smoother = FeatureSmoother(config.FEATURE_SMOOTHING_SAMPLES)

        # Threshold store: defaults to config values, can be replaced at runtime.
        if thresholds is None:
            from calibration import ThresholdStore
            thresholds = ThresholdStore()
        self.thresholds = thresholds

        self.sleep_active = False
        self.drowsy_active = False
        self.yawn_active = False
        self.distraction_active = False
        self.hands_missing = False

        # PERCLOS rolling window.
        self.perclos_window_sec = config.PERCLOS_WINDOW_SEC
        self._perclos_events = deque()  # (timestamp, is_closed)

    @staticmethod
    def _elapsed(start, now):
        return 0.0 if start is None else max(0.0, now - start)

    @staticmethod
    def _update_condition(flag, since, recovery_since, active, now, trigger_sec, recovery_sec):
        """Since/recovery hysteresis.

        ``active`` is only cleared once the flag has been false continuously for
        ``recovery_sec``; a single good frame therefore never ends an alert.
        """
        if flag:
            if since is None:
                since = now
            recovery_since = None
            if now - since >= trigger_sec:
                active = True
        else:
            since = None
            if active:
                if recovery_since is None:
                    recovery_since = now
                if now - recovery_since >= recovery_sec:
                    active = False
                    recovery_since = None
            else:
                recovery_since = None
        return since, recovery_since, active

    def update_face(self, face_detected, now=None):
        now = time.monotonic() if now is None else now
        if face_detected:
            self.last_face_time = now
            self.face_lost = False
            self._face_lost_reported = False
            return
        if self.last_face_time is None:
            # No face has ever been seen: there is nothing to lose, and the HUD
            # must not claim the face was "lost".
            self.face_lost = False
            return
        self.face_lost = (now - self.last_face_time) >= config.FACE_GRACE_SEC
        if self.face_lost and not self._face_lost_reported:
            # No face -> no eye, yawn or head-pose evidence. Drop every face-derived
            # alert once, when the grace period expires, so a lost face cannot keep
            # "DRIVER SLEEP" active while the cabin shows nobody.
            self.reset_face_features()
            self._face_lost_reported = True

    def reset_face_features(self):
        self.eye_closed_since = None
        self.eye_recovery_since = None
        self.yawn_since = None
        self.yawn_recovery_since = None
        self.distraction_since = None
        self.distraction_recovery_since = None
        self.sleep_active = False
        self.drowsy_active = False
        self.yawn_active = False
        self.distraction_active = False
        self.eye_smoother.clear()
        self.mar_smoother.clear()
        self.pose_smoother.clear()
        self._perclos_events.clear()
        self.perclos = 0.0

    def update_eyes(self, avg_ear, now=None):
        now = time.monotonic() if now is None else now
        if avg_ear is None:
            self.reset_face_features()
            return None
        smoothed = self.eye_smoother.update(avg_ear)
        ear_threshold = self.thresholds.ear_threshold
        closed = smoothed is not None and smoothed < ear_threshold
        self.eye_closed_since, self.eye_recovery_since, self.sleep_active = self._update_condition(
            closed, self.eye_closed_since, self.eye_recovery_since, self.sleep_active, now,
            config.SLEEP_DURATION_SEC, config.RECOVERY_DURATION_SEC,
        )
        # Drowsiness is the shorter version of the same eye-closure condition and
        # shares its recovery hysteresis: a blink never clears it. It stays alive
        # until sleep clears (eyes open past recovery), so the two flags cannot
        # disagree about the same eye evidence.
        if closed:
            if self._elapsed(self.eye_closed_since, now) >= config.DROWSY_DURATION_SEC:
                self.drowsy_active = True
        elif not self.sleep_active:
            self.drowsy_active = False
            self.eye_recovery_since = None

        # PERCLOS: rolling percentage of time eyes were closed in the window.
        self._perclos_events.append((now, closed))
        cutoff = now - self.perclos_window_sec
        while self._perclos_events and self._perclos_events[0][0] < cutoff:
            self._perclos_events.popleft()
        self.perclos = self._compute_perclos()
        return smoothed

    def _compute_perclos(self):
        """Percentage of samples in the window where eyes were closed."""
        if not self._perclos_events:
            return 0.0
        total = len(self._perclos_events)
        closed_count = sum(1 for _, c in self._perclos_events if c)
        return 100.0 * closed_count / total

    def update_yawn(self, mar, now=None):
        now = time.monotonic() if now is None else now
        if mar is None:
            self.yawn_since = None
            self.yawn_recovery_since = None
            self.yawn_active = False
            self.mar_smoother.clear()
            return None
        smoothed = self.mar_smoother.update(mar)
        yawn = smoothed is not None and smoothed > self.thresholds.mar_threshold
        self.yawn_since, self.yawn_recovery_since, self.yawn_active = self._update_condition(
            yawn, self.yawn_since, self.yawn_recovery_since, self.yawn_active, now,
            config.YAWN_DURATION_SEC, config.RECOVERY_DURATION_SEC,
        )
        return smoothed

    def update_distraction(self, looking_away, now=None):
        now = time.monotonic() if now is None else now
        self.distraction_since, self.distraction_recovery_since, self.distraction_active = self._update_condition(
            looking_away, self.distraction_since, self.distraction_recovery_since, self.distraction_active, now,
            config.DISTRACTION_DURATION_SEC, config.RECOVERY_DURATION_SEC,
        )

    def update_hands(self, hand_count, now=None):
        now = time.monotonic() if now is None else now
        missing = hand_count <= 0
        self.hands_missing_since, self.hands_recovery_since, self.hands_missing = self._update_condition(
            missing, self.hands_missing_since, self.hands_recovery_since, self.hands_missing, now,
            config.HANDS_NOT_VISIBLE_SEC, config.RECOVERY_DURATION_SEC,
        )

    def reset_for_camera_reconnect(self):
        self.__init__(thresholds=self.thresholds)