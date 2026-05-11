"""Windows input emulation — pydirectinput.

pydirectinput sends DirectInput events that games take, unlike SendInput-only
emulators that some game clients silently ignore. Keys use pyautogui-style
names which match our KEY_VOCAB closely (we normalize a couple aliases).
"""
from __future__ import annotations

import pydirectinput

# pydirectinput key normalization: KEY_VOCAB uses "esc", pydirectinput accepts both,
# but be explicit just in case the underlying mapping changes.
_KEY_ALIASES = {
    "esc": "escape",
}


def move(x: int, y: int) -> None:
    pydirectinput.moveTo(int(x), int(y))


def click(x: int, y: int, button: str = "left") -> None:
    pydirectinput.click(x=int(x), y=int(y), button=button)


def press_key(key: str) -> None:
    pydirectinput.press(_KEY_ALIASES.get(key, key))
