"""Recorder configuration.

Bboxes are RELATIVE to the OSRS client's client-area top-left, in pixels at the
client's reference resolution. UI scale and window resolution must be locked
before recording, and remeasured if either changes.
"""
from dataclasses import dataclass, asdict, field
from pathlib import Path

# Reference client size (width, height) at which the bboxes below were measured.
# OSRS classic fixed mode is 765x503; resizable is anything. Set to whatever you
# lock the client at, then never change it without remeasuring.
REFERENCE_SIZE = (765, 503)

# Region bboxes RELATIVE to the client area, format (x1, y1, x2, y2).
# These are PLACEHOLDERS — measure against your private server's GE interface
# and inventory before any real recording session.
DEFAULT_GE_MENU_BBOX = (10, 30, 510, 320)
DEFAULT_INVENTORY_BBOX = (550, 200, 760, 470)


@dataclass
class CaptureConfig:
    window_title_substr: str = "RuneLite"
    fps: int = 30
    output_root: Path = field(default_factory=lambda: Path("data/demos"))
    ge_menu_bbox: tuple = DEFAULT_GE_MENU_BBOX
    inventory_bbox: tuple = DEFAULT_INVENTORY_BBOX
    kill_key: str = "f12"
    save_full_frame_every_n: int = 0  # 0 disables; else save the full client view every N frames for debugging
    log_mouse_move: bool = False  # high volume; off by default — clicks/keys are usually enough

    def to_dict(self) -> dict:
        d = asdict(self)
        d["output_root"] = str(self.output_root)
        return d
