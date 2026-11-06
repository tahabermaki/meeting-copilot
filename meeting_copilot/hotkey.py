"""Raccourci clavier global Windows (fonctionne même quand Teams a le focus)."""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable

log = logging.getLogger(__name__)


def start_hotkey_listener(vk: int, on_press: Callable[[], None], on_error: Callable[[str], None]) -> None:
    if sys.platform != "win32":
        log.info("Raccourci global indisponible hors Windows")
        return

    def run():
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        MOD_NOREPEAT, WM_HOTKEY = 0x4000, 0x0312
        if not user32.RegisterHotKey(None, 1, MOD_NOREPEAT, vk):
            on_error("⚠️ Touche d'aide déjà utilisée par un autre logiciel : utilise le bouton 🆘")
            log.warning("RegisterHotKey a échoué pour vk=0x%X", vk)
            return
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            if msg.message == WM_HOTKEY:
                on_press()

    threading.Thread(target=run, daemon=True, name="hotkey").start()
