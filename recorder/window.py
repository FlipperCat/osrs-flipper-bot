"""Locate the OSRS client window and resolve its client-area bbox.

The client area excludes the title bar and borders, which is what dxcam should
capture and what cursor coordinates should be relative to.
"""
from __future__ import annotations

import win32gui


def find_client_hwnd(title_substr: str) -> int | None:
    """Return the hwnd of the first visible window whose title contains title_substr (case-insensitive)."""
    found: list[int] = []

    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            t = win32gui.GetWindowText(hwnd)
            if title_substr.lower() in t.lower():
                found.append(hwnd)

    win32gui.EnumWindows(cb, None)
    return found[0] if found else None


def get_client_bbox(hwnd: int) -> tuple[int, int, int, int]:
    """Client-area bbox in screen coords: (left, top, right, bottom)."""
    rect = win32gui.GetClientRect(hwnd)  # (0, 0, w, h)
    w, h = rect[2], rect[3]
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    return (left, top, left + w, top + h)


def is_foreground(hwnd: int) -> bool:
    return win32gui.GetForegroundWindow() == hwnd
