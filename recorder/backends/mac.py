"""macOS backend: mss screen capture + Quartz window targeting.

Notes for first-time Mac setup:
- Grant the terminal (or whatever runs python) **Screen Recording** AND
  **Accessibility** permissions in System Settings → Privacy & Security.
- mss is not thread-safe, so capture runs on its own dedicated thread.
- Quartz returns the OUTER window bounds (includes the title bar). The
  in-game UI bboxes in config.py must therefore be measured RELATIVE to
  the value get_client_bbox() returns on this OS — don't reuse Windows
  measurements verbatim.
"""
from __future__ import annotations

import threading
import time

import numpy as np


def _list_windows():
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGWindowListOptionOnScreenOnly,
        kCGNullWindowID,
    )

    return CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)


def find_window(title_substr: str) -> dict | None:
    needle = title_substr.lower()
    for w in _list_windows():
        name = (w.get("kCGWindowName") or "").lower()
        owner = (w.get("kCGWindowOwnerName") or "").lower()
        if needle in name or needle in owner:
            return dict(w)  # copy: original is a CFDictionary
    return None


def get_client_bbox(window_info: dict) -> tuple[int, int, int, int]:
    b = window_info["kCGWindowBounds"]
    x = int(b["X"])
    y = int(b["Y"])
    w = int(b["Width"])
    h = int(b["Height"])
    return (x, y, x + w, y + h)


def is_foreground(window_info: dict) -> bool:
    from AppKit import NSWorkspace

    front = NSWorkspace.sharedWorkspace().frontmostApplication()
    if front is None:
        return False
    owner = window_info.get("kCGWindowOwnerName") or ""
    return str(front.localizedName()) == owner


class Capture:
    """mss-based capture running on a dedicated thread.

    Matches dxcam's "get the latest frame, return None if no new one since last call"
    semantics so the capture loop in capture.py works unchanged.
    """

    def __init__(self) -> None:
        self._region: tuple[int, int, int, int] | None = None
        self._target_fps = 30
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, region: tuple[int, int, int, int], target_fps: int) -> None:
        self._region = region
        self._target_fps = max(1, int(target_fps))
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import mss  # lazy: only loaded on macOS

        period = 1.0 / self._target_fps
        assert self._region is not None
        l, t, r, b = self._region
        monitor = {"left": l, "top": t, "width": r - l, "height": b - t}
        with mss.mss() as sct:
            while not self._stop.is_set():
                t0 = time.monotonic()
                shot = sct.grab(monitor)
                # shot is BGRA uint8; slice to BGR then reverse to RGB
                arr = np.asarray(shot)
                rgb = np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])
                with self._lock:
                    self._frame = rgb
                dt = time.monotonic() - t0
                if dt < period:
                    time.sleep(period - dt)

    def get_latest_frame(self) -> np.ndarray | None:
        with self._lock:
            f = self._frame
            self._frame = None
        return f

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
