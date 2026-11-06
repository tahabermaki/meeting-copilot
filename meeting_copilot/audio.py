"""Capture audio Windows (WASAPI) : son de la réunion (loopback) et micro.

Le « loopback » capte ce qui sort des haut-parleurs ou du casque : la voix des
interlocuteurs dans Teams, Zoom ou Meet, sans bot dans la réunion.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

log = logging.getLogger(__name__)


def _pyaudio():
    import pyaudiowpatch as pyaudio  # Windows uniquement : import tardif pour les tests

    return pyaudio


def find_loopback_device(pa) -> dict:
    pyaudio = _pyaudio()
    wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    speakers = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
    if speakers.get("isLoopbackDevice"):
        return speakers
    for loopback in pa.get_loopback_device_info_generator():
        if speakers["name"] in loopback["name"]:
            return loopback
    raise RuntimeError(
        "Aucun périphérique loopback pour la sortie audio par défaut "
        "(vérifie la sortie son sélectionnée dans Windows)."
    )


def to_mono_int16(raw: bytes, channels: int) -> bytes:
    samples = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return samples.tobytes()


class AudioSource:
    """Pousse des blocs PCM mono 16 bits de 100 ms dans une queue asyncio."""

    def __init__(self, pa, device: dict, label: str, loop: asyncio.AbstractEventLoop):
        pyaudio = _pyaudio()
        self.label = label
        self.rate = int(device["defaultSampleRate"])
        self.channels = max(1, min(int(device["maxInputChannels"]), 2))
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=600)  # ~60 s de tampon max
        channels = self.channels

        def callback(in_data, frame_count, time_info, status):
            chunk = to_mono_int16(in_data, channels)
            loop.call_soon_threadsafe(self._put, chunk)
            return (None, pyaudio.paContinue)

        self.stream = pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=device["index"],
            frames_per_buffer=int(self.rate * 0.1),
            stream_callback=callback,
            start=False,
        )
        log.info("Source audio « %s » : %s (%d Hz, %d canaux)", label, device["name"], self.rate, self.channels)

    def _put(self, chunk: bytes) -> None:
        if self.queue.full():  # réseau coupé : on jette le plus ancien plutôt que saturer la mémoire
            self.queue.get_nowait()
        self.queue.put_nowait(chunk)

    def start(self) -> None:
        self.stream.start_stream()

    def close(self) -> None:
        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception:
            pass


def open_sources(loop, listen_mic: bool, on_status) -> tuple[object, list[AudioSource]]:
    from .assistant import ME, OTHERS

    pyaudio = _pyaudio()
    pa = pyaudio.PyAudio()
    sources = [AudioSource(pa, find_loopback_device(pa), OTHERS, loop)]
    if listen_mic:
        try:
            sources.append(AudioSource(pa, pa.get_default_input_device_info(), ME, loop))
        except Exception as e:
            log.warning("Micro indisponible : %s", e)
            on_status(f"⚠️ Micro indisponible : seul le son de la réunion est écouté ({e})")
    return pa, sources
