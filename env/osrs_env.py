"""Gymnasium environment wrapping the live OSRS client.

Used by:
  - live BC rollout (deterministic argmax actions, see env/rollout.py)
  - future PPO fine-tuning (Stage 3 in project.md), where the env can be
    wrapped to expose a real reward sourced from the private server's DB.

Safety (enforced at the env layer, not trusted from the model):
  - All click coords are clamped to the OSRS client window bbox.
  - Actions only fire when the client is the foreground window.
  - Hard floor on action rate (`action_min_interval_s`) caps the bot's pace
    even if the dwell head outputs 0.
  - pyautogui.FAILSAFE on macOS gives a hardware corner-of-screen abort.

Observation space (Dict):
  - "ge":  Box(K*(3+N), H_ge, W_ge)   float32, normalized via train.transforms
  - "inv": Box(K*(3+N), H_inv, W_inv) float32

Action space:
  MultiDiscrete([
    len(REGION_VOCAB),   # 0=none, 1=menu, 2=inventory
    grid_x,
    grid_y,
    len(BUTTON_VOCAB),   # 0=none, 1=left, 2=right
    len(KEY_VOCAB),
    len(DWELL_BUCKETS_MS),
  ])

Click takes priority over key — matches BC label semantics where each frame
maps to one next action (click OR key OR idle, never both).
"""
from __future__ import annotations

import collections
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.input_backends import emulator
from recorder import backends as cap_backends
from recorder.config import CaptureConfig
from train.config import (
    BUTTON_VOCAB,
    DWELL_BUCKETS_MS,
    DataConfig,
    KEY_VOCAB,
    REGION_VOCAB,
)
from train.transforms import preprocess_stack


HEAD_ORDER: tuple[str, ...] = (
    "region", "click_x", "click_y", "button", "key", "dwell",
)


class OSRSFlipEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data_cfg: DataConfig | None = None,
        capture_cfg: CaptureConfig | None = None,
        capture_fps: int = 30,
        action_min_interval_s: float = 0.05,
    ):
        super().__init__()
        self.data_cfg = data_cfg or DataConfig()
        self.capture_cfg = capture_cfg or CaptureConfig()
        self.capture_fps = capture_fps
        self.action_min_interval_s = action_min_interval_s

        # Bboxes: client-relative, set by the user when they calibrate
        # recorder/config.py to their game window.
        self.ge_bbox: tuple[int, int, int, int] = tuple(self.capture_cfg.ge_menu_bbox)
        self.inv_bbox: tuple[int, int, int, int] = tuple(self.capture_cfg.inventory_bbox)

        # Runtime state
        self._handle: Any = None
        self._client_bbox: tuple[int, int, int, int] | None = None
        self._cam: Any = None
        K = self.data_cfg.frame_stack
        self._ge_buf: collections.deque[np.ndarray] = collections.deque(maxlen=K)
        self._inv_buf: collections.deque[np.ndarray] = collections.deque(maxlen=K)
        self._last_action_t: float = 0.0

        # Spaces
        n_mask = len(self.data_cfg.mask_specs)
        ch = K * (3 + n_mask)
        H_ge, W_ge = self.data_cfg.ge_resize_hw
        H_inv, W_inv = self.data_cfg.inv_resize_hw
        self.observation_space = spaces.Dict({
            "ge": spaces.Box(low=-10.0, high=10.0, shape=(ch, H_ge, W_ge), dtype=np.float32),
            "inv": spaces.Box(low=-10.0, high=10.0, shape=(ch, H_inv, W_inv), dtype=np.float32),
        })
        self.action_space = spaces.MultiDiscrete([
            len(REGION_VOCAB),
            self.data_cfg.grid_x,
            self.data_cfg.grid_y,
            len(BUTTON_VOCAB),
            len(KEY_VOCAB),
            len(DWELL_BUCKETS_MS),
        ])

    # ---- gym API --------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        handle = cap_backends.find_window(self.capture_cfg.window_title_substr)
        if handle is None:
            raise RuntimeError(
                f"Game window not found (title contains "
                f"{self.capture_cfg.window_title_substr!r})."
            )
        self._handle = handle
        self._client_bbox = cap_backends.get_client_bbox(handle)

        if self._cam is None:
            self._cam = cap_backends.Capture()
            self._cam.start(region=self._client_bbox, target_fps=self.capture_fps)

        # Prime the frame stack
        K = self.data_cfg.frame_stack
        self._ge_buf.clear()
        self._inv_buf.clear()
        period = 1.0 / self.capture_fps
        deadline = time.monotonic() + period * (K * 4)
        while len(self._ge_buf) < K and time.monotonic() < deadline:
            self._grab_once()
            if len(self._ge_buf) < K:
                time.sleep(period / 2)
        if len(self._ge_buf) < K:
            raise RuntimeError(
                "Timed out filling the initial frame stack — is the game window visible?"
            )

        return self._build_observation(), {}

    def step(self, action):
        region_i, cx, cy, btn_i, key_i, dwell_i = (int(a) for a in action)

        # Min action interval (hard cap regardless of dwell head)
        now = time.monotonic()
        gap = self.action_min_interval_s - (now - self._last_action_t)
        if gap > 0:
            time.sleep(gap)

        if cap_backends.is_foreground(self._handle):
            self._execute(region_i, cx, cy, btn_i, key_i)
        self._last_action_t = time.monotonic()

        dwell_s = DWELL_BUCKETS_MS[dwell_i] / 1000.0
        if dwell_s > 0:
            time.sleep(dwell_s)

        self._wait_for_next_frame()
        obs = self._build_observation()
        # BC rollout has no reward signal. Stage-3 PPO will wrap the env to inject one.
        return obs, 0.0, False, False, {}

    def close(self):
        if self._cam is not None:
            self._cam.stop()
            self._cam = None

    # ---- internals ------------------------------------------------------

    def _grab_once(self) -> bool:
        frame = self._cam.get_latest_frame()
        if frame is None:
            return False
        self._ge_buf.append(self._crop(frame, self.ge_bbox))
        self._inv_buf.append(self._crop(frame, self.inv_bbox))
        return True

    def _wait_for_next_frame(self) -> None:
        period = 1.0 / self.capture_fps
        deadline = time.monotonic() + period * 4
        while time.monotonic() < deadline:
            if self._grab_once():
                return
            time.sleep(period / 4)
        # If we time out we silently reuse the last frame in the buffer; the
        # frame stack is already filled from reset() so the obs is still valid.

    @staticmethod
    def _crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2].copy()

    def _build_observation(self) -> dict[str, np.ndarray]:
        ge_t = preprocess_stack(list(self._ge_buf), self.data_cfg.ge_resize_hw, self.data_cfg)
        inv_t = preprocess_stack(list(self._inv_buf), self.data_cfg.inv_resize_hw, self.data_cfg)
        return {
            "ge": ge_t.numpy().astype(np.float32),
            "inv": inv_t.numpy().astype(np.float32),
        }

    def _execute(self, region_i: int, cx_bin: int, cy_bin: int, btn_i: int, key_i: int) -> None:
        region = REGION_VOCAB[region_i]
        button = BUTTON_VOCAB[btn_i]
        key = KEY_VOCAB[key_i]

        # Click is the priority action; key only fires if no click.
        if region != "none" and button != "none":
            bbox = self.ge_bbox if region == "menu" else self.inv_bbox
            screen_x, screen_y = self._bin_to_screen(cx_bin, cy_bin, bbox)
            emulator.click(screen_x, screen_y, button=button)
            return

        if key != "none":
            emulator.press_key(key)

    def _bin_to_screen(
        self, bx: int, by: int, bbox_client_rel: tuple[int, int, int, int]
    ) -> tuple[int, int]:
        """Map (bin_x, bin_y) in a client-relative region back to clamped screen coords."""
        x1, y1, x2, y2 = bbox_client_rel
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        rel_x = x1 + (bx + 0.5) / self.data_cfg.grid_x * w
        rel_y = y1 + (by + 0.5) / self.data_cfg.grid_y * h

        assert self._client_bbox is not None
        cx_origin, cy_origin = self._client_bbox[0], self._client_bbox[1]
        abs_x = int(cx_origin + rel_x)
        abs_y = int(cy_origin + rel_y)

        # Hard clamp to the client window — model cannot click outside the game.
        abs_x = max(self._client_bbox[0], min(self._client_bbox[2] - 1, abs_x))
        abs_y = max(self._client_bbox[1], min(self._client_bbox[3] - 1, abs_y))
        return abs_x, abs_y
