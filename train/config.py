"""Training data configuration: action grid, vocab, frame-stack, mask specs."""
from __future__ import annotations

from dataclasses import dataclass, field


# Action label vocabularies. Index = class id. "none" must always be index 0
# so that idle frames map cleanly to a zeroed label.
REGION_VOCAB: tuple[str, ...] = ("none", "menu", "inventory")
BUTTON_VOCAB: tuple[str, ...] = ("none", "left", "right")
KEY_VOCAB: tuple[str, ...] = (
    "none", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "enter", "backspace", "esc", "space", "k",
)
DWELL_BUCKETS_MS: tuple[int, ...] = (0, 100, 250, 500)


@dataclass
class MaskSpec:
    """One HSV color threshold producing one binary mask channel.

    PIL's HSV uses 0-255 for all three channels (H is *not* in degrees, *not* in
    OpenCV's 0-179 range). If h_min > h_max, the range wraps around 0 — useful
    for red, which lives at both ends of the hue circle.
    """
    name: str
    h_min: int
    h_max: int
    s_min: int = 100
    s_max: int = 255
    v_min: int = 100
    v_max: int = 255


@dataclass
class DataConfig:
    # Action label space
    grid_x: int = 32                      # horizontal click bins per region
    grid_y: int = 24                      # vertical click bins per region
    lookahead_s: float = 0.1              # only label a frame with an action if the human acted within this window (else idle)

    # Image preprocessing
    ge_resize_hw: tuple[int, int] = (224, 256)    # (H, W) the GE tower sees
    inv_resize_hw: tuple[int, int] = (256, 256)
    frame_stack: int = 4

    # Color masks — PLACEHOLDERS. Tune to the plugin's actual overlay colors
    # by measuring against captured demo frames before training.
    mask_specs: list[MaskSpec] = field(default_factory=lambda: [
        MaskSpec(name="overlay_green", h_min=60, h_max=110),     # green ~85 in PIL HSV
        MaskSpec(name="overlay_red", h_min=240, h_max=15),       # wraps 0
        MaskSpec(name="overlay_cyan", h_min=115, h_max=140),
    ])

    # ImageNet normalization (RGB channels only — mask channels stay in [0,1])
    rgb_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    rgb_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
