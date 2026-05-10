# OSRS Flipper Bot

**Goal:** Train a PyTorch vision policy to flip items on the in-game GE of a private OSRS server, learning from pixels alone using a private overlay plugin as visual cue.
**Tech:** Python 3.12, PyTorch, `dxcam` (screen capture), `pydirectinput` (mouse/keys), `pywin32` (window mgmt), `gymnasium` + `stable-baselines3` (RL stage), SQLite (demo logs).
**Status:** Pre-implementation. Architecture committed — see `project.md`. Next step: build `recorder/` and capture first 1h demo.

## Read first
- `project.md` — full architecture, action space, training stages, data plan. SOURCE OF TRUTH.
- `recorder/` — demo capture (frames + actions) — START HERE before any training
- `model/policy.py` — two-tower CNN + multi-head action policy
- `train/bc.py` — behavioral cloning loop
- `env/osrs_env.py` — gymnasium wrapper for online play / RL

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
- `pydirectinput` over `pyautogui` — DirectInput survives game-side input filtering.
- `dxcam` is faster than `mss` on Windows but needs the OSRS window in the foreground.
- Server-DB GP delta is the cleanest RL reward — query directly, don't OCR the bank.
