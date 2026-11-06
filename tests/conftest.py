"""Outils de test : faux client Anthropic (aucun appel réseau) et faux affichage."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from meeting_copilot.config import Settings
from meeting_copilot.knowledge import Document, KnowledgeBase
from meeting_copilot.metrics import MeetingMetrics


@dataclass
class Script:
    """Réponse scriptée d'un appel : texte découpé en morceaux, délai entre morceaux."""

    text: str
    chunk: int = 6
    delay: float = 0.0
    error: Exception | None = None
    usage: dict = field(default_factory=lambda: {"input_tokens": 1000, "output_tokens": 50})


class FakeStream:
    def __init__(self, script: Script):
        self.script = script
        self.current_message_snapshot = SimpleNamespace(usage=SimpleNamespace(input_tokens=1000, output_tokens=0))

    async def __aenter__(self):
        if self.script.error:
            raise self.script.error
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        async def gen():
            t = self.script.text
            for i in range(0, len(t), self.script.chunk):
                await asyncio.sleep(self.script.delay)
                yield t[i : i + self.script.chunk]

        return gen()

    async def get_final_message(self):
        return SimpleNamespace(usage=SimpleNamespace(**self.script.usage))


class FakeClient:
    def __init__(self, *scripts: Script):
        self.scripts = list(scripts)
        self.calls: list[dict] = []
        self.messages = self

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return FakeStream(self.scripts.pop(0))


class FakeDisplay:
    def __init__(self):
        self.answers: list[tuple[str, str, str, str]] = []
        self.statuses: list[str] = []
        self.heard: list[str] = []

    def show_answer(self, title, body, source, footer):
        self.answers.append((title, body, source, footer))

    def show_status(self, message):
        self.statuses.append(message)

    def show_heard(self, text):
        self.heard.append(text)

    @property
    def final(self):
        """Dernière réponse complète affichée (celle qui porte les délais)."""
        done = [a for a in self.answers if a[3]]
        return done[-1] if done else None


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def kb():
    return KnowledgeBase([Document("faq.md", "Q : Prix de l'atelier ?\nR : 3 900 € HT.")], 60_000, 6)


@pytest.fixture
def metrics(settings):
    return MeetingMetrics(settings.pricing, settings.llm.model)
