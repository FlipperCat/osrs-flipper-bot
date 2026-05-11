"""Capture orchestrator. Cross-platform (Windows + macOS).

DO NOT RUN until:
  1. `pip install -r requirements.txt` has been done in a venv
  2. The OSRS client is open at a locked resolution and UI scale
  3. ge_menu_bbox + inventory_bbox in config.py have been REMEASURED
     against the actual GE interface and inventory positions, on the OS
     you are recording on (don't reuse Windows measurements on macOS)

macOS only: grant Screen Recording AND Accessibility permissions to the
terminal/python binary in System Settings → Privacy & Security.

Usage (later, after the above):
    python -m recorder.capture --duration 600
    python -m recorder.capture --dry-run         # validate window detection only

Hard rule: pixel-only training. This captures only on-screen pixels and human
input events. Plugin internals are not read here and must never be.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np

from recorder import backends
from recorder.config import CaptureConfig
from recorder.input_hook import InputHook
from recorder.writer import SessionWriter


def crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2].copy()


def run(cfg: CaptureConfig, duration_s: float, dry_run: bool = False) -> int:
    handle = backends.find_window(cfg.window_title_substr)
    if handle is None:
        print(f"Client window not found (title contains '{cfg.window_title_substr}'). Aborting.")
        return 1

    bbox = backends.get_client_bbox(handle)
    cw, ch = bbox[2] - bbox[0], bbox[3] - bbox[1]
    print(f"Platform: {sys.platform}  bbox={bbox}  client_size=({cw}, {ch})")
    print(f"GE menu bbox (relative): {cfg.ge_menu_bbox}")
    print(f"Inventory bbox (relative): {cfg.inventory_bbox}")

    if dry_run:
        print("[dry-run] window detection ok; exiting without capturing.")
        return 0

    cam = backends.Capture()
    cam.start(region=bbox, target_fps=cfg.fps)

    hook = InputHook(log_mouse_move=cfg.log_mouse_move)
    hook.start()

    writer = SessionWriter(
        root=Path(cfg.output_root),
        meta={
            "platform": sys.platform,
            "config": cfg.to_dict(),
            "client_bbox": bbox,
            "client_size": (cw, ch),
        },
    )
    print(f"Recording → {writer.session_dir}")
    print(f"Press {cfg.kill_key} to stop, or Ctrl+C.")

    stop_flag = {"v": False}

    def _signal_stop(*_):
        stop_flag["v"] = True

    signal.signal(signal.SIGINT, _signal_stop)

    from pynput import keyboard

    def _on_kill(key):
        name = getattr(key, "name", None) or getattr(key, "char", None)
        if name == cfg.kill_key:
            stop_flag["v"] = True
            return False

    kill_listener = keyboard.Listener(on_press=_on_kill)
    kill_listener.start()

    t_end = time.monotonic() + duration_s
    frames = 0
    try:
        while not stop_flag["v"] and time.monotonic() < t_end:
            if not backends.is_foreground(handle):
                time.sleep(0.05)
                continue

            frame = cam.get_latest_frame()
            if frame is None:
                continue

            ts = time.monotonic()
            ge = crop(frame, cfg.ge_menu_bbox)
            inv = crop(frame, cfg.inventory_bbox)
            full = (
                frame
                if cfg.save_full_frame_every_n and frames % cfg.save_full_frame_every_n == 0
                else None
            )
            idx = writer.write_frame(ts, ge, inv, full=full)

            events = hook.drain()
            if events:
                writer.write_actions(events, idx)

            frames += 1
    finally:
        cam.stop()
        hook.stop()
        kill_listener.stop()
        writer.close()
        print(f"Done. Wrote {frames} frames to {writer.session_dir}")
    return 0


def main():
    p = argparse.ArgumentParser(description="OSRS Flipper Bot demo recorder.")
    p.add_argument("--duration", type=float, default=600, help="Seconds to record.")
    p.add_argument("--fps", type=int, default=None, help="Override capture FPS.")
    p.add_argument("--window", type=str, default=None, help="Window title substring.")
    p.add_argument("--out", type=str, default=None, help="Output dir for sessions.")
    p.add_argument("--log-mouse-move", action="store_true", help="Also log mouse-move events (high volume).")
    p.add_argument("--dry-run", action="store_true", help="Detect the window and exit without capturing.")
    args = p.parse_args()

    cfg = CaptureConfig()
    if args.fps:
        cfg.fps = args.fps
    if args.window:
        cfg.window_title_substr = args.window
    if args.out:
        cfg.output_root = Path(args.out)
    if args.log_mouse_move:
        cfg.log_mouse_move = True

    sys.exit(run(cfg, args.duration, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
