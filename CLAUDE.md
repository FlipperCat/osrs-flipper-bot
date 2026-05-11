# OSRS Flipper Bot

**Goal:** Train a PyTorch vision policy to flip items on the in-game GE of a private OSRS server, learning from pixels alone using a private overlay plugin as visual cue.
**Tech:** Python 3.12, PyTorch, `pynput` (input capture, cross-platform), platform backends in `recorder/backends/` — Windows: `dxcam` + `pywin32` + `pydirectinput`; macOS: `mss` + `pyobjc` (Quartz/AppKit) + `pyautogui`. `gymnasium` + `stable-baselines3` for the RL stage. SQLite for demo logs.
**Status:** Pre-implementation. Architecture committed — see `project.md`. Next step: build `recorder/` and capture first 1h demo.

## Read first
- `project.md` — full architecture, action space, training stages, data plan. SOURCE OF TRUTH.
- `recorder/` — demo capture (frames + actions). Cross-platform via `recorder/backends/{win,mac}.py`.
- `train/dataset.py` — `OSRSFlipDataset`: torch glue. `python -m train.dataset <data_root>` to inspect.
- `train/labeling.py` — pure-python session indexing + multi-head label building (torch-free, unit-tested).
- `tests/` — `python -m unittest discover tests`. Pure-python tests run anywhere; torch-gated tests skip if torch missing. `tests/synth.py` writes synthetic recorder sessions for tests.
- `train/transforms.py` — HSV color masks + resize + frame-stack concat (per-tower input).
- `train/config.py` — `DataConfig`: action grid, vocab, mask specs (PLACEHOLDER colors — retune).
- `model/policy.py` — `FlipperPolicy`: two ConvNeXt-Tiny towers (timm) + trunk + six action heads.
- `train/bc.py` — BC training script. Device auto (CUDA/MPS/CPU), backbone-freeze warmup, per-session temporal val split, checkpoints to `checkpoints/{last,best}.pt`.
- `env/osrs_env.py` — `OSRSFlipEnv` gymnasium env. Live capture loop + bbox-clamped action exec + foreground/rate-limit safety. Used by rollout + (later) PPO.
- `env/rollout.py` — load a checkpoint, run argmax actions live. F12 kill switch. `--dry-run` validates env init without acting.
- `env/input_backends/{win,mac}.py` — input emulation. Windows: `pydirectinput`. macOS: `pyautogui` (needs Accessibility permission).

## Constraints (hard)
- **Pixel-only input.** Plugin internals are private and OFF-LIMITS as a teacher signal. Model learns from RGB + HSV color masks only.
- **Bounded action surface:** clicks inside the trade menu bbox + keystrokes for price/qty. No world clicks, no walking, no chat.
- **Two regions:** trade menu + inventory. Crop towers, no global-view tower.
- **Private server only.** Never point this at official Jagex worlds — ToS violation, account ban.

## Don't
- Don't read or import plugin source/state — pixel-only is a hard rule.
- Don't skip the bbox-clamp safety: clicks must be physically clamped to the OSRS window.
- Don't jump to PPO before BC beats a scripted baseline on simulated GP/hr.
- Don't train on raw RGB without the HSV color-mask aux channels — that's the whole pixel-side preprocess.
- Don't record demos at variable resolution/UI scale — lock both before capture.

## Gotchas (will fill as encountered)
- **Cross-platform**: `recorder/backends/{win,mac}.py` is the OS-isolated layer. Anything OS-specific belongs there, not in `capture.py`.
- On Windows: `pydirectinput` > `pyautogui` — DirectInput survives game-side input filtering. `dxcam` is faster than `mss`.
- On macOS: terminal/python needs **Screen Recording** + **Accessibility** permission. Bboxes must be remeasured per-OS — Quartz returns outer window bounds (incl. title bar), `win32gui` returns the client area.
- Server-DB GP delta is the cleanest RL reward — query directly, don't OCR the bank.
