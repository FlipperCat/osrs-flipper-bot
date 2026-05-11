"""Platform dispatch for capture + window backends.

Each backend exposes the same surface:
    find_window(title_substr: str) -> Optional[Handle]
    get_client_bbox(handle: Handle) -> tuple[int, int, int, int]   # (l, t, r, b) screen coords
    is_foreground(handle: Handle) -> bool
    Capture                                                          # start/stop/get_latest_frame

`Handle` is opaque and backend-specific (hwnd int on Windows, dict on macOS).
Consumers must not introspect it.
"""
from __future__ import annotations

import sys

if sys.platform == "win32":
    from .win import find_window, get_client_bbox, is_foreground, Capture  # noqa: F401
elif sys.platform == "darwin":
    from .mac import find_window, get_client_bbox, is_foreground, Capture  # noqa: F401
else:
    raise NotImplementedError(
        f"No recorder backend for platform {sys.platform!r}. "
        "Add one under recorder/backends/."
    )
