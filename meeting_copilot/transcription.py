"""Transcription en streaming avec Deepgram (WebSocket).

- Reconnexion automatique avec délai progressif (1 s, 2 s, 4 s… plafonné à 30 s).
- Arrêt net si la clé est refusée (inutile de réessayer en boucle).
- Compte les secondes d'audio envoyées, pour le calcul du coût.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from urllib.parse import urlencode

import websockets

from .config import TranscriptionConfig

log = logging.getLogger(__name__)

URL = "wss://api.deepgram.com/v1/listen"


class AuthError(RuntimeError):
    pass


def build_url(cfg: TranscriptionConfig, sample_rate: int) -> str:
    params = {
        "model": cfg.model,
        "language": cfg.language,
        "encoding": "linear16",
        "sample_rate": sample_rate,
        "channels": 1,
        "punctuate": "true",
        "smart_format": "true",
        "interim_results": "true",
        "endpointing": cfg.endpointing_ms,
        "utterance_end_ms": 1000,
    }
    return f"{URL}?{urlencode(params)}"


class UtteranceAssembler:
    """Regroupe les segments « finaux » de Deepgram en phrases complètes."""

    def __init__(self, on_utterance: Callable[[str], None]):
        self.on_utterance = on_utterance
        self.buffer: list[str] = []

    def feed(self, data: dict) -> None:
        kind = data.get("type")
        if kind == "Results":
            alts = data.get("channel", {}).get("alternatives") or [{}]
            text = (alts[0].get("transcript") or "").strip()
            if data.get("is_final") and text:
                self.buffer.append(text)
            if data.get("speech_final"):
                self.flush()
        elif kind == "UtteranceEnd":
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            text = " ".join(self.buffer).strip()
            self.buffer.clear()
            if text:
                self.on_utterance(text)


async def transcribe(
    label: str,
    queue: asyncio.Queue,
    sample_rate: int,
    cfg: TranscriptionConfig,
    api_key: str,
    on_utterance: Callable[[str, str], None],
    on_status: Callable[[str], None],
    on_audio: Callable[[float], None] = lambda s: None,
) -> None:
    url = build_url(cfg, sample_rate)
    headers = {"Authorization": f"Token {api_key}"}
    delay = 1.0
    while True:
        connected_at = None
        try:
            async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
                connected_at = time.monotonic()
                on_status(f"🎧 {label} : à l'écoute · F9 = aide")
                log.info("Deepgram connecté (%s, %d Hz)", label, sample_rate)
                assembler = UtteranceAssembler(lambda text: on_utterance(label, text))

                async def sender():
                    while True:
                        try:
                            chunk = await asyncio.wait_for(queue.get(), timeout=5)
                        except TimeoutError:
                            await ws.send(json.dumps({"type": "KeepAlive"}))  # silence total
                            continue
                        await ws.send(chunk)
                        on_audio(len(chunk) / 2 / sample_rate)  # PCM 16 bits mono

                async def receiver(assembler=assembler):
                    async for message in ws:
                        assembler.feed(json.loads(message))

                tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
                for t in done:
                    if t.exception():
                        raise t.exception()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403):
                on_status(f"⚠️ {label} : clé Deepgram refusée — vérifie .env")
                log.error("Deepgram a refusé la clé (HTTP %s)", status)
                raise AuthError("Clé Deepgram refusée") from e
            if connected_at and time.monotonic() - connected_at > 30:
                delay = 1.0  # la connexion a tenu : on repart d'un délai court
            on_status(f"⚠️ {label} : reconnexion dans {delay:.0f} s ({type(e).__name__})")
            log.warning("Deepgram (%s) déconnecté : %r — reconnexion dans %.0f s", label, e, delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
