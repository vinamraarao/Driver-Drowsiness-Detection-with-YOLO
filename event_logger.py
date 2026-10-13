"""Safe CSV event logger with snapshot support."""
import csv
import cv2
from datetime import datetime
from pathlib import Path
import threading
import os
import time


class EventLogger:
    def __init__(self, log_file, snapshot_dir=None):
        self.log_file = Path(log_file)
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        if self.snapshot_dir:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._last_event = ""
        self._lock = threading.Lock()
        self.last_error = None
        self._last_snapshot_event = ""  # Track last event that triggered snapshot
        self._last_snapshot_time = 0    # Timestamp of last snapshot

    def reset(self):
        with self._lock:
            self._last_event = ""

    def log(self, cabin, identity, event, snapshot_frame=None):
        """Log an event, optionally saving a snapshot frame for HIGH_RISK or CRITICAL events.

        Args:
            cabin: Driver status string
            identity: "Driver" or "N/A"
            event: Event description string
            snapshot_frame: Optional camera frame to save as snapshot

        Returns:
            bool: True if logging succeeded, False otherwise
        """
        if not event or event == "System Nominal":
            return True

        snapshot_path = ""
        # Save snapshot for high-risk events, avoiding duplicates
        if (snapshot_frame is not None and self.snapshot_dir is not None and
            ("HIGH_RISK" in event or "CRITICAL" in event)):
            # Avoid duplicate snapshots for the same ongoing event
            if event != self._last_snapshot_event or (time.monotonic() - self._last_snapshot_time) >= 5.0:
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snapshot_filename = f"incident_{timestamp}_{event.replace(' ', '_').replace(':', '')}.jpg"
                    snapshot_path = str(self.snapshot_dir / snapshot_filename)
                    cv2.imwrite(snapshot_path, snapshot_frame)
                    self._last_snapshot_event = event
                    self._last_snapshot_time = time.monotonic()
                except Exception as exc:
                    print(f"[WARN] Snapshot save failed: {exc}")
                    snapshot_path = ""  # Continue logging even if snapshot fails

        with self._lock:
            if event == self._last_event and not snapshot_path:
                return True
            try:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                new_file = not self.log_file.exists() or self.log_file.stat().st_size == 0
                with self.log_file.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    if new_file:
                        writer.writerow(["timestamp", "cabin", "identity", "event", "snapshot_path"])
                    writer.writerow([
                        datetime.now().astimezone().isoformat(timespec="seconds"),
                        cabin,
                        identity,
                        event,
                        snapshot_path,
                    ])
                    handle.flush()
                self._last_event = event
                self.last_error = None
                return True
            except OSError as exc:
                self.last_error = str(exc)
                print(f"[WARN] Event log write failed: {exc}")
                # Do not crash the monitoring system because a log file is locked/full.
                return False

    def reset(self):
        with self._lock:
            self._last_event = ""
            self._last_snapshot_event = ""
            self._last_snapshot_time = 0
