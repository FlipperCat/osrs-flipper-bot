"""Windows backend: dxcam screen capture + win32gui window targeting."""
from __future__ import annotations

import numpy as np
import win32gui


def find_window(title_substr: str) -> int | None:
    found: list[int] = []

    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            t = win32gui.GetWindowText(hwnd)
            if title_substr.lower() in t.lower():
                found.append(hwnd)

    win32gui.EnumWindows(cb, None)
    return found[0] if found else None


def get_client_bbox(hwnd: int) -> tuple[int, int, int, int]:
    rect = win32gui.GetClientRect(hwnd)
    w, h = rect[2], rect[3]
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    return (left, top, left + w, top + h)


def is_foreground(hwnd: int) -> bool:
    return win32gui.GetForegroundWindow() == hwnd


class Capture:
    """Thin wrapper around dxcam matching the cross-platform Capture interface."""

    def __init__(self) -> None:
        self._cam = None

    def start(self, region: tuple[int, int, int, int], target_fps: int) -> None:
        import dxcam  # lazy: only loaded on Windows

        self._cam = dxcam.create(output_color="RGB")
        self._cam.start(region=region, target_fps=target_fps)

    def get_latest_frame(self) -> np.ndarray | None:
        if self._cam is None:
            return None
        return self._cam.get_latest_frame()

    def stop(self) -> None:
        if self._cam is not None:
            self._cam.stop()
            self._cam = None
