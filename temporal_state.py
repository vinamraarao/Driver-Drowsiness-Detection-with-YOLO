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

        # Progressive alert states
        self.alert_level = "SAFE"  # SAFE, LOW_RISK, WARNING, HIGH_RISK, CRITICAL
        self.alert_level_since = None  # When current alert level started
        self.alert_level_recovery_since = None  # Recovery tracking for de-escalation

        # Hand-off-wheel monitoring
        self.hand_left_since = None      # Timestamp when left hand came off wheel
        self.right_hand_left_since = None # Timestamp when right hand came off wheel
        self.both_hands_off_since = None  # Timestamp when both hands came off wheel
        self.hand_off_wheel_state = "BOTH_ON"  # BOTH_ON, ONE_OFF, BOTH_OFF

        self.sleep_active = False
        self.drowsy_active = False
        self.yawn_active = False
        self.distraction_active = False
        self.hands_missing = False

        # PERCLOS rolling window.
        self.perclos_window_sec = config.PERCLOS_WINDOW_SEC
        self._perclos_events = deque()  # (timestamp, is_closed)

    def update_hand_off_wheel(self, hand_result, frame_w, frame_h, now=None):
        """Update hand-off-wheel state based on hand positions relative to steering wheel ROI.

        Args:
            hand_result: HandFrameResult from hand analysis
            frame_w, frame_h: frame dimensions
            now: current monotonic time
        """
        if now is None:
            now = time.monotonic()

        # Get steering wheel ROI from config (normalized coordinates)
        roi = getattr(config, 'STEERING_WHEEL_ROI', [0.3, 0.3, 0.7, 0.7])
        wheel_x1 = int(roi[0] * frame_w)
        wheel_y1 = int(roi[1] * frame_h)
        wheel_x2 = int(roi[2] * frame_w)
        wheel_y2 = int(roi[3] * frame_h)

        # Count hands on wheel
        hands_on_wheel = 0
        total_hands = hand_result.hands_detected if hand_result else 0

        if hand_result and hand_result.landmarks_list:
            for hand_landmarks in hand_result.landmarks_list:
                # Use wrist landmark (index 0) as approximate hand position
                wrist_x, wrist_y = hand_landmarks[0]
                if wheel_x1 <= wrist_x <= wheel_x2 and wheel_y1 <= wrist_y <= wheel_y2:
                    hands_on_wheel += 1

        hands_off_wheel = total_hands - hands_on_wheel

        # Determine current state
        if hands_off_wheel == 0:
            current_state = "BOTH_ON"
        elif hands_off_wheel == 1:
            current_state = "ONE_OFF"
        else:  # hands_off_wheel >= 2
            current_state = "BOTH_OFF"

        # Update timers based on state changes
        if current_state != self.hand_off_wheel_state:
            # State changed, reset timers for new state
            self.hand_off_wheel_state = current_state
            if current_state == "BOTH_ON":
                self.hand_left_since = None
                self.right_hand_left_since = None
                self.both_hands_off_since = None
            elif current_state == "ONE_OFF":
                # Start timer for one hand off
                self.hand_left_since = now
                self.right_hand_left_since = None
                self.both_hands_off_since = None
            else:  # BOTH_OFF
                # Start timer for both hands off
                self.hand_left_since = None
                self.right_hand_left_since = None
                self.both_hands_off_since = now
        else:
            # State unchanged, update existing timers
            if current_state == "ONE_OFF":
                # Keep the earliest timestamp for one hand off
                if self.hand_left_since is None:
                    self.hand_left_since = now
            elif current_state == "BOTH_OFF":
                # Keep the earliest timestamp for both hands off
                if self.both_hands_off_since is None:
                    self.both_hands_off_since = now

    def get_hand_off_wheel_status(self, now=None):
        """Get hand-off-wheel status and duration.

        Returns:
            tuple: (state, duration_seconds, is_warning, is_high_risk, is_critical)
        """
        if now is None:
            now = time.monotonic()

        if self.hand_off_wheel_state == "BOTH_ON":
            return "BOTH_ON", 0.0, False, False, False
        elif self.hand_off_wheel_state == "ONE_OFF":
            duration = now - self.hand_left_since if self.hand_left_since else 0.0
            warning_threshold = getattr(config, 'HAND_OFF_WHEEL_WARNING_1HAND', 30.0)
            high_risk_threshold = getattr(config, 'HAND_OFF_WHEEL_HIGH_RISK_1HAND', 60.0)
            is_warning = duration >= warning_threshold
            is_high_risk = duration >= high_risk_threshold
            is_critical = False  # One hand off is never critical
            return "ONE_OFF", duration, is_warning, is_high_risk, is_critical
        else:  # BOTH_OFF
            duration = now - self.both_hands_off_since if self.both_hands_off_since else 0.0
            warning_threshold = getattr(config, 'HAND_OFF_WHEEL_WARNING_2HAND', 2.0)
            critical_threshold = getattr(config, 'HAND_OFF_WHEEL_CRITICAL_2HAND', 5.0)
            is_warning = duration >= warning_threshold
            is_high_risk = duration >= critical_threshold  # Treat as high risk for alert system
            is_critical = duration >= critical_threshold
            return "BOTH_OFF", duration, is_warning, is_high_risk, is_critical

    def reset_for_camera_reconnect(self):
        self.__init__(thresholds=self.thresholds)

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
            self.reset_hand_off_wheel_on_face_loss()
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

    def reset_hand_off_wheel_on_face_loss(self):
        """Reset hand-off-wheel state when face/driver is lost.

        When the face is lost, we cannot reliably determine hand position
        relative to the steering wheel, so we reset the hand-off-wheel
        state to avoid false alerts.
        """
        self.hand_off_wheel_state = "BOTH_ON"
        self.hand_left_since = None
        self.right_hand_left_since = None
        self.both_hands_off_since = None

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

    def update_alert_level(self, phone_present=False, face_detected=True, now=None):
        """Update the progressive alert level based on current state.

        Alert levels: SAFE -> LOW_RISK -> WARNING -> HIGH_RISK -> CRITICAL
        Escalates based on severity/persistence, de-escalates when conditions improve.
        """
        if now is None:
            now = time.monotonic()

        # Determine the base alert level from current conditions
        if self.sleep_active and phone_present:
            base_level = "CRITICAL"
        elif self.sleep_active:
            base_level = "CRITICAL"
        elif phone_present:
            base_level = "HIGH_RISK"
        elif self.drowsy_active or getattr(self, "perclos", 0.0) >= config.PERCLOS_DROWSY_THRESHOLD:
            base_level = "WARNING"
        elif self.yawn_active:
            base_level = "WARNING"
        elif self.distraction_active:
            base_level = "WARNING"
        elif config.ENABLE_HAND_ALERTS and self.hands_missing:
            base_level = "LOW_RISK"
        elif not face_detected:
            base_level = "LOW_RISK"
        else:
            base_level = "SAFE"

        # Hand-off-wheel integration: both hands off wheel is critical
        hand_state, _, _, _, _ = self.get_hand_off_wheel_status(now)
        if hand_state == "BOTH_OFF":
            base_level = "CRITICAL"
        elif hand_state == "ONE_OFF":
            # One hand off wheel is warning risk
            base_level = "WARNING"

        # If we're already at the base level or higher, just update the timestamp
        if base_level == "CRITICAL" or (self.alert_level == "CRITICAL" and base_level in ["HIGH_RISK", "WARNING", "LOW_RISK", "SAFE"]):
            # Stay at CRITICAL until conditions improve significantly
            if self.alert_level != "CRITICAL":
                self.alert_level = "CRITICAL"
                self.alert_level_since = now
                self.alert_level_recovery_since = None
        elif base_level == "HIGH_RISK":
            if self.alert_level in ["HIGH_RISK", "CRITICAL"]:
                # Maintain or escalate to HIGH_RISK
                if self.alert_level != "HIGH_RISK":
                    self.alert_level = "HIGH_RISK"
                    self.alert_level_since = now
                    self.alert_level_recovery_since = None
            else:
                # Escalate to HIGH_RISK from lower levels
                self.alert_level = "HIGH_RISK"
                self.alert_level_since = now
                self.alert_level_recovery_since = None
        elif base_level == "WARNING":
            if self.alert_level in ["WARNING", "HIGH_RISK", "CRITICAL"]:
                # Maintain or escalate from WARNING level
                if self.alert_level != "WARNING":
                    self.alert_level = "WARNING"
                    self.alert_level_since = now
                    self.alert_level_recovery_since = None
            else:
                # Escalate to WARNING from LOW_RISK or SAFE
                self.alert_level = "WARNING"
                self.alert_level_since = now
                self.alert_level_recovery_since = None
        elif base_level == "LOW_RISK":
            if self.alert_level in ["LOW_RISK", "WARNING", "HIGH_RISK", "CRITICAL"]:
                # Maintain or escalate from LOW_RISK level
                if self.alert_level != "LOW_RISK":
                    self.alert_level = "LOW_RISK"
                    self.alert_level_since = now
                    self.alert_level_recovery_since = None
            else:
                # Escalate to LOW_RISK from SAFE
                self.alert_level = "LOW_RISK"
                self.alert_level_since = now
                self.alert_level_recovery_since = None
        else:  # base_level == "SAFE"
            # De-escalate toward SAFE - only if we've been in recovery state long enough
            if self.alert_level == "SAFE":
                # Already safe, keep the timestamp updated
                self.alert_level_since = now
            elif self.alert_level_recovery_since is None:
                # Start recovery tracking
                self.alert_level_recovery_since = now
            elif now - self.alert_level_recovery_since >= config.RECOVERY_DURATION_SEC * 2:  # Require sustained good state
                # De-escalate one level toward SAFE
                if self.alert_level == "LOW_RISK":
                    self.alert_level = "SAFE"
                    self.alert_level_since = now
                    self.alert_level_recovery_since = None
                elif self.alert_level == "WARNING":
                    self.alert_level = "LOW_RISK"
                    self.alert_level_since = now
                    self.alert_level_recovery_since = None
                elif self.alert_level == "HIGH_RISK":
                    self.alert_level = "WARNING"
                    self.alert_level_since = now
                    self.alert_level_recovery_since = None
                elif self.alert_level == "CRITICAL":
                    self.alert_level = "HIGH_RISK"
                    self.alert_level_since = now
                    self.alert_level_recovery_since = None

    def calculate_risk_score(self, phone_present=False, face_detected=True):
        """Calculate a 0-100 risk score based on current driver state.

        Higher scores indicate higher risk:
        - 0-20: SAFE (normal operation)
        - 21-40: LOW RISK (minor issues like no face detection)
        - 41-60: WARNING (drowsiness, yawning, distraction)
        - 61-80: HIGH RISK (phone usage)
        - 81-100: CRITICAL (sleep, sleep+phone)
        """
        # Critical risks (81-100)
        if self.sleep_active and phone_present:
            return 100  # Sleep + phone
        elif self.sleep_active:
            return 90   # Sleep alone

        # High risks (61-80)
        elif phone_present:
            return 70   # Phone usage

        # Hand-off-wheel integration: both hands off wheel is critical
        hand_state, _, _, _, _ = self.get_hand_off_wheel_status()
        if hand_state == "BOTH_OFF":
            return 85   # Both hands off wheel (critical risk)

        # Warning risks (41-60)
        elif self.drowsy_active or getattr(self, "perclos", 0.0) >= config.PERCLOS_DROWSY_THRESHOLD:
            return 50   # Drowsiness/PERCLOS
        elif self.yawn_active:
            return 45   # Yawning
        elif self.distraction_active:
            return 40   # Distraction (head pose)
        elif config.ENABLE_HAND_ALERTS and self.hands_missing:
            return 35   # Hands not visible

        # Hand-off-wheel integration: one hand off wheel is warning
        elif hand_state == "ONE_OFF":
            return 30   # One hand off wheel (warning risk)

        # Low risks (21-40)
        elif not face_detected:
            return 25   # No face detected
        else:
            return 5    # Normal/safe operation

    def get_hand_off_wheel_risk_level(self, now=None):
        """Get hand-off-wheel risk level for integration with risk score and alert systems.

        Returns:
            str: Risk level contribution from hand-off-wheel ("SAFE", "LOW_RISK", "WARNING", "HIGH_RISK", "CRITICAL")
        """
        if now is None:
            now = time.monotonic()

        state, duration, is_warning, is_high_risk, is_critical = self.get_hand_off_wheel_status(now)

        if state == "BOTH_ON":
            return "SAFE"
        elif state == "ONE_OFF":
            if is_high_risk:
                return "HIGH_RISK"
            elif is_warning:
                return "WARNING"
            else:
                return "LOW_RISK"
        else:  # BOTH_OFF
            if is_critical:
                return "CRITICAL"
            elif is_high_risk:
                return "HIGH_RISK"
            else:
                return "WARNING"

    def reset_for_camera_reconnect(self):
        self.__init__(thresholds=self.thresholds)