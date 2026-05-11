"""macOS input emulation — pyautogui.

Requires the running process to have **Accessibility** permission in
System Settings → Privacy & Security → Accessibility.

pyautogui.FAILSAFE = True (default) means moving the cursor to a corner
aborts — we leave that on as a hardware kill switch.
"""
from __future__ import annotations

import pyautogui

pyautogui.PAUSE = 0.0  # we manage timing via the env's dwell head, not a global delay


_KEY_ALIASES = {
    "esc": "escape",
}


def move(x: int, y: int) -> None:
    pyautogui.moveTo(int(x), int(y), duration=0)


def click(x: int, y: int, button: str = "left") -> None:
    pyautogui.click(x=int(x), y=int(y), button=button)


def press_key(key: str) -> None:
    pyautogui.press(_KEY_ALIASES.get(key, key))
