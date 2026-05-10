"""Pynput-based mouse + keyboard event capture.

Events are pushed to a thread-safe queue and drained by the capture loop each
frame. Timestamps use time.monotonic() so they share a clock with frame stamps.
"""
from __future__ import annotations

import time
from queue import Queue, Empty
from typing import Callable

from pynput import mouse, keyboard


def _key_name(key) -> str:
    try:
        return key.char
    except AttributeError:
        return getattr(key, "name", str(key))


class InputHook:
    def __init__(self, log_mouse_move: bool = False):
        self.q: Queue = Queue()
        self.log_mouse_move = log_mouse_move
        self._mouse_listener: mouse.Listener | None = None
        self._kb_listener: keyboard.Listener | None = None

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def _on_click(self, x, y, button, pressed):
        self.q.put({
            "ts": self._now(), "type": "click", "x": x, "y": y,
            "button": button.name, "pressed": pressed,
        })

    def _on_move(self, x, y):
        if self.log_mouse_move:
            self.q.put({"ts": self._now(), "type": "move", "x": x, "y": y})

    def _on_scroll(self, x, y, dx, dy):
        self.q.put({
            "ts": self._now(), "type": "scroll", "x": x, "y": y, "dx": dx, "dy": dy,
        })

    def _on_press(self, key):
        self.q.put({"ts": self._now(), "type": "key_down", "key": _key_name(key)})

    def _on_release(self, key):
        self.q.put({"ts": self._now(), "type": "key_up", "key": _key_name(key)})

    def start(self):
        self._mouse_listener = mouse.Listener(
            on_click=self._on_click,
            on_move=self._on_move,
            on_scroll=self._on_scroll,
        )
        self._kb_listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._mouse_listener.start()
        self._kb_listener.start()

    def stop(self):
        if self._mouse_listener:
            self._mouse_listener.stop()
        if self._kb_listener:
            self._kb_listener.stop()

    def drain(self) -> list[dict]:
        out: list[dict] = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except Empty:
                break
        return out
