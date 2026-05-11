"""Platform dispatch for input emulation backends.

Each backend exposes the same surface:
    move(x: int, y: int) -> None
    click(x: int, y: int, button: str) -> None     # button in {"left", "right"}
    press_key(key: str) -> None                     # key names match KEY_VOCAB
"""
from __future__ import annotations

import sys

if sys.platform == "win32":
    from . import win as emulator  # noqa: F401
elif sys.platform == "darwin":
    from . import mac as emulator  # noqa: F401
else:
    raise NotImplementedError(
        f"No input backend for platform {sys.platform!r}. Add one under env/input_backends/."
    )
