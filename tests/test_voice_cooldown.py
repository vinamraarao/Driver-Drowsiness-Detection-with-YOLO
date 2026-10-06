"""Regression tests for the voice alert cooldown table and the bounded worker.

No pytest fixtures are used: config is saved/restored with try/finally and the
module-level voice service is swapped with unittest.mock.patch.
"""
import contextlib
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audio_alert
import config


@contextlib.contextmanager
def silent_service():
    """Run against a real _VoiceService whose SAPI5 engine is mocked out."""
    service = audio_alert._VoiceService()
    service._make_engine = lambda: None
    try:
        with patch.object(audio_alert, "_service", service):
            yield service
    finally:
        service.stop(timeout=1.0)


def test_config_has_no_phantom_cooldown_attribute():
    # Regression: voice_alert read config.VOICE_ALERT_COOLDOWN_SEC, which never
    # existed, so every alert fell back to 8 s instead of its configured value.
    assert not hasattr(config, "VOICE_ALERT_COOLDOWN_SEC")
    assert config.VOICE_DEFAULT_COOLDOWN_SEC in config.VOICE_COOLDOWNS.values()
    for key in ("sleep", "sleep_phone", "phone", "drowsiness", "yawn", "distraction", "hands"):
        assert key in config.VOICE_COOLDOWNS, f"missing cooldown for {key}"


def test_per_alert_cooldown_table_is_used():
    with silent_service() as service:
        seen = []
        service.submit = lambda message, key, cd: seen.append((message, key, cd)) or True
        assert audio_alert.sleep_alert() is True
        assert audio_alert.yawn_alert() is True
        assert audio_alert.phone_alert() is True
        assert audio_alert.sleep_and_phone_alert() is True
    keys = {k for _, k, _ in seen}
    assert keys == {"sleep", "yawn", "phone", "sleep_phone"}
    assert all(abs(cd - config.VOICE_COOLDOWNS[k]) < 1e-9 for _, k, cd in seen if k != "sleep_phone")
    assert seen[-1][2] == 6.0, "sleep+phone keeps its explicit 6 s cooldown"


def test_unknown_key_uses_default_cooldown():
    with silent_service() as service:
        seen = []
        service.submit = lambda message, key, cd: seen.append((key, cd)) or True
        audio_alert.voice_alert("hello", alert_key="something_new")
    assert seen == [("something_new", config.VOICE_DEFAULT_COOLDOWN_SEC)]


def test_cooldown_suppresses_repeat_and_is_per_key():
    service = audio_alert._VoiceService()
    try:
        assert service.submit("a", "drowsiness", config.VOICE_COOLDOWNS["drowsiness"]) is True
        assert service.submit("a", "drowsiness", config.VOICE_COOLDOWNS["drowsiness"]) is False
        # A different key is not blocked by the first key's cooldown.
        assert service.submit("a", "yawn", config.VOICE_COOLDOWNS["yawn"]) is True
        assert service.submit("b", "drowsiness", 0.0) is True
    finally:
        service.stop(timeout=0.5)


def test_alert_helpers_use_matching_keys():
    with silent_service() as service:
        seen = []
        service.submit = lambda message, key, cd: seen.append(key) or True
        audio_alert.sleep_alert()
        audio_alert.sleep_and_phone_alert()
        audio_alert.drowsiness_alert()
        audio_alert.yawn_alert()
        audio_alert.distraction_alert()
        audio_alert.phone_alert()
        audio_alert.hands_not_visible_alert()
    assert seen == ["sleep", "sleep_phone", "drowsiness", "yawn", "distraction", "phone", "hands"]


def test_queue_is_bounded_and_drops_oldest():
    service = audio_alert._VoiceService()
    try:
        for i in range(50):
            service.submit(f"m{i}", f"k{i}", 0.0)
        assert service._queue.qsize() <= config.VOICE_QUEUE_SIZE
        queued = [m for _, m in list(service._queue.queue)]
        assert queued[-1] == "m49", "the newest alert must be kept"
        assert queued[0] == "m47", "the oldest pending messages are dropped"
    finally:
        service.stop(timeout=0.5)


def test_single_worker_thread_only():
    before = {t.name for t in threading.enumerate()}
    with silent_service() as service:
        for i in range(5):
            service.submit(f"m{i}", f"k{i}", 0.0)
        time.sleep(0.2)
        after = {t.name for t in threading.enumerate()}
        assert len(after - before) <= 1, "one worker thread, not one per alert"
        assert service._thread is not None and service._thread.is_alive()
    assert not any(t.name == "VoiceAlert" and t.is_alive() for t in threading.enumerate()), \
        "shutdown_voice must stop the worker"


def test_submit_returns_immediately_and_does_not_speak_inline():
    with silent_service() as service:
        t0 = time.perf_counter()
        for i in range(200):
            service.submit(f"m{i}", f"k{i}", 0.0)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.5, f"200 submits took {elapsed:.2f}s; the camera loop must not block"
        assert service._current is None, "no engine is created on the calling thread"


def test_shutdown_is_idempotent_and_silent():
    audio_alert.shutdown_voice()
    audio_alert.shutdown_voice()
    assert audio_alert.stop() is None


def test_voice_disabled_short_circuits():
    saved = config.VOICE_ENABLED
    try:
        config.VOICE_ENABLED = False
        with silent_service() as service:
            assert audio_alert.voice_alert("x", alert_key="sleep") is False
            assert service._queue.qsize() == 0
            assert service._thread is None, "no worker is started when voice is disabled"
    finally:
        config.VOICE_ENABLED = saved


def test_beep_is_removed():
    assert audio_alert.alert_beep() is False
    assert not hasattr(audio_alert, "beep")