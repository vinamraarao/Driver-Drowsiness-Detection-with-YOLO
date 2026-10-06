"""Voice-only alert system for the driver monitoring system.

Design constraints:
  * The camera loop must never block on audio. Every ``voice_alert`` call
    returns immediately after a cheap cooldown check and a queue put.
  * At most one bounded worker thread owns the SAPI5 engine and speaks one
    message at a time.
  * ``shutdown_voice()`` stops the worker and waits briefly for the current
    utterance, so the process is silent on exit.
  * No beep path exists.
"""

import queue
import threading
import time

import config


class _VoiceService:
    def __init__(self):
        self._queue = queue.Queue(maxsize=max(1, int(getattr(config, "VOICE_QUEUE_SIZE", 3))))
        self._lock = threading.Lock()
        self._last_times = {}
        self._stop = threading.Event()
        self._current = None
        self._started = False
        self._thread = None

    # -- lifecycle ---------------------------------------------------
    def start(self):
        with self._lock:
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(target=self._worker, name="VoiceAlert", daemon=True)
            self._thread.start()

    def stop(self, timeout=3.0):
        thread = self._thread
        if thread is not None:
            self._stop.set()
            try:
                thread.join(timeout=timeout)
            except Exception:
                pass
        engine = self._current
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
            self._current = None

    # -- enqueue -----------------------------------------------------
    def submit(self, message, key, cooldown):
        now = time.monotonic()
        with self._lock:
            last = self._last_times.get(key)
            if last is not None and (now - last) < cooldown:
                return False
            self._last_times[key] = now
            if not self._started:
                self._started = True
                self._thread = threading.Thread(target=self._worker, name="VoiceAlert", daemon=True)
                self._thread.start()
        try:
            self._queue.put_nowait((now, message))
        except queue.Full:
            # Drop the oldest pending message so the newest alert wins.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait((now, message))
            except Exception:
                return False
        return True

    # -- worker ------------------------------------------------------
    def _worker(self):
        engine = None
        try:
            while not self._stop.is_set():
                try:
                    _, message = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if self._stop.is_set():
                    break
                try:
                    if engine is None:
                        engine = self._make_engine()
                        self._current = engine
                    if engine is not None:
                        engine.say(message)
                        engine.runAndWait()
                    print(f"[VOICE] Spoken: {message}")
                except Exception as exc:
                    print(f"[VOICE ERROR] Worker failed: {exc}")
                    # A dead SAPI5 engine must not poison later alerts.
                    if engine is not None:
                        try:
                            engine.stop()
                        except Exception:
                            pass
                        engine = None
                        self._current = None
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
            self._current = None

    @staticmethod
    def _make_engine():
        import pyttsx3
        engine = pyttsx3.init("sapi5")
        engine.setProperty("rate", getattr(config, "VOICE_RATE", 165))
        engine.setProperty("volume", getattr(config, "VOICE_VOLUME", 1.0))
        return engine


_service = _VoiceService()


def _cooldown_for(key, override):
    if override is not None:
        return float(override)
    table = getattr(config, "VOICE_COOLDOWNS", None)
    if isinstance(table, dict) and key in table:
        return float(table[key])
    return float(getattr(config, "VOICE_DEFAULT_COOLDOWN_SEC", 4.0))


def voice_alert(message, alert_key="general", cooldown=None):
    """Queue one alert. Returns True when queued, False when suppressed."""
    if not getattr(config, "VOICE_ENABLED", True):
        return False
    return _service.submit(message, alert_key, _cooldown_for(alert_key, cooldown))


def alert_beep(frequency=0, duration=0):
    """Compatibility no-op. The project is intentionally voice-only."""
    return False


def sleep_alert():
    return voice_alert(
        "Warning! Driver sleep detected. Please wake up and focus on the road.",
        alert_key="sleep",
    )


def phone_alert():
    return voice_alert(
        "Warning! Mobile phone usage detected. Please put your phone away and focus on driving.",
        alert_key="phone",
    )


def sleep_and_phone_alert():
    return voice_alert(
        "Critical warning! Driver sleep and mobile phone usage detected. "
        "Please put the phone away, wake up, and focus on the road.",
        alert_key="sleep_phone",
        cooldown=6.0,
    )


def drowsiness_alert():
    return voice_alert(
        "Warning! Driver drowsiness detected. Please stay alert and focus on the road.",
        alert_key="drowsiness",
    )


def distraction_alert():
    return voice_alert(
        "Warning! Please focus on the road.",
        alert_key="distraction",
    )


def yawn_alert():
    return voice_alert(
        "Warning! Driver yawning detected. Please stay alert.",
        alert_key="yawn",
    )


def hands_off_wheel_alert():
    return voice_alert(
        "Warning! Please keep your hands on the steering wheel.",
        alert_key="hands_off_wheel",
    )


def hands_not_visible_alert():
    return voice_alert(
        "Warning! Driver hands are not visible. Please keep your hands on the steering wheel.",
        alert_key="hands",
    )


def shutdown_voice():
    """Stop the worker; the queue is drained so nothing speaks after exit."""
    _service.stop()


def stop():
    """Compatibility alias for older project versions."""
    return shutdown_voice()