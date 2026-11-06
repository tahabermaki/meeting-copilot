"""Point d'entrée : assemble configuration, connaissances, audio, transcription, IA et fenêtre.

Deux threads :
- le thread Qt (interface) ;
- un thread asyncio (audio → Deepgram → Claude), qui parle à l'interface par signaux.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

from .config import ROOT, ConfigError, Settings, load_settings
from .journal import Journal
from .knowledge import KnowledgeBase, load_documents, load_text
from .metrics import MeetingMetrics

log = logging.getLogger("meeting_copilot")


def setup_logging(data_dir: Path) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        (data_dir / "logs").mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(
            data_dir / "logs" / "copilot.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        ))
    except OSError:
        pass  # pas de fichier de log : la console suffit
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s : %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    for noisy in ("httpx", "httpcore", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def run_pipeline(settings: Settings, ui, kb: KnowledgeBase, context: str,
                       metrics: MeetingMetrics, journal: Journal) -> None:
    from anthropic import AsyncAnthropic

    from .assistant import Assistant
    from .audio import open_sources
    from .transcription import transcribe

    loop = asyncio.get_running_loop()
    assistant = Assistant(AsyncAnthropic(), settings, kb, context, ui, metrics, journal)
    ui.on_help = lambda: asyncio.run_coroutine_threadsafe(assistant.manual_help(), loop)
    ui.on_typed = lambda t: asyncio.run_coroutine_threadsafe(assistant.typed(t), loop)

    pa, sources = open_sources(loop, settings.transcription.listen_mic, ui.show_status)
    for s in sources:
        s.start()
    try:
        await asyncio.gather(*(
            transcribe(
                s.label, s.queue, s.rate, settings.transcription, os.environ["DEEPGRAM_API_KEY"],
                on_utterance=assistant.on_utterance,
                on_status=ui.show_status,
                on_audio=lambda sec, label=s.label: metrics.add_audio(label, sec),
            )
            for s in sources
        ))
    finally:
        for s in sources:
            s.close()
        pa.terminate()


def main() -> None:
    load_dotenv(ROOT / ".env")
    try:
        settings = load_settings()
    except (ConfigError, OSError, ValueError) as e:
        sys.exit(f"config.toml invalide : {e}")

    setup_logging(settings.data_dir)
    missing = [k for k in ("ANTHROPIC_API_KEY", "DEEPGRAM_API_KEY") if not os.environ.get(k)]
    if missing:
        sys.exit(f"Clés manquantes dans .env : {', '.join(missing)} (voir .env.example)")

    docs = load_documents(settings.docs_dir)
    kb = KnowledgeBase(docs, settings.retrieval.full_context_max_chars, settings.retrieval.top_k)
    context = load_text(settings.context_file)
    log.info("Contexte de réunion : %s", settings.context_file if context else "aucun")

    stamp = f"{datetime.now():%Y-%m-%d_%H%M}"
    journal = Journal(settings.data_dir / "transcripts" / f"reunion_{stamp}.txt")
    metrics = MeetingMetrics(settings.pricing, settings.llm.model)

    from PyQt6.QtWidgets import QApplication

    from .hotkey import start_hotkey_listener
    from .ui import Overlay

    app = QApplication(sys.argv)
    ui = Overlay(settings.ui)
    ui.show()
    ui.show_status(f"{len(docs)} doc(s) · connexion… · pense à prévenir les participants")

    def on_quit():
        report = metrics.write_report(settings.data_dir / "reports")
        if report:
            log.info("Rapport de réunion : %s", report)

    app.aboutToQuit.connect(on_quit)
    start_hotkey_listener(settings.ui.hotkey_vk, ui.help_signal.emit, ui.show_status)

    def worker():
        try:
            asyncio.run(run_pipeline(settings, ui, kb, context, metrics, journal))
        except Exception as e:
            log.exception("Arrêt du pipeline")
            ui.show_status(f"⚠️ {e}")

    threading.Thread(target=worker, daemon=True, name="pipeline").start()
    sys.exit(app.exec())
