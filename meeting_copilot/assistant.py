"""Le « cerveau » : reçoit les phrases transcrites, appelle le modèle, pilote l'affichage.

Aucune dépendance à l'interface ni à l'audio : l'affichage passe par l'interface
`Display`, le modèle par un client compatible `anthropic.AsyncAnthropic`.
C'est ce qui permet de tester toute la logique avec un faux client.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections import deque
from typing import Protocol

from .config import Settings
from .detection import (
    SYSTEM_PROMPT,
    Reply,
    build_auto_prompt,
    build_help_prompt,
    build_typed_prompt,
    parse_reply,
    should_analyze,
)
from .journal import Journal
from .knowledge import KnowledgeBase
from .metrics import CallRecord, MeetingMetrics, Usage

log = logging.getLogger(__name__)

ME = "Moi"            # étiquette du micro de l'utilisateur
OTHERS = "Interlocuteur"


class Display(Protocol):
    def show_answer(self, title: str, body: str, source: str, footer: str) -> None: ...
    def show_status(self, message: str) -> None: ...
    def show_heard(self, text: str) -> None: ...


def build_system(meeting_context: str, kb: KnowledgeBase) -> list[dict]:
    """Prompt système : consignes + (contexte de réunion et documents, mis en cache)."""
    blocks = [{"type": "text", "text": SYSTEM_PROMPT}]
    reference = ""
    if meeting_context:
        reference += "Contexte de la réunion et profil de l'utilisateur :\n\n" + meeting_context + "\n\n"
    docs = kb.static_text()
    if docs:
        reference += "Documents de référence :\n\n" + docs
    if reference:
        # Mis en cache côté API : relu à ~10 % du prix à chaque appel suivant.
        blocks.append({"type": "text", "text": reference, "cache_control": {"type": "ephemeral"}})
    return blocks


def _usage_from(message) -> Usage:
    u = getattr(message, "usage", None)
    if u is None:
        return Usage(estimated=True)
    return Usage(
        input_tokens=getattr(u, "input_tokens", 0) or 0,
        output_tokens=getattr(u, "output_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
    )


class Assistant:
    def __init__(
        self,
        client,
        settings: Settings,
        kb: KnowledgeBase,
        meeting_context: str,
        display: Display,
        metrics: MeetingMetrics,
        journal: Journal | None = None,
    ):
        self.client = client
        self.s = settings
        self.kb = kb
        self.display = display
        self.metrics = metrics
        self.journal = journal or Journal(None)
        self.system = build_system(meeting_context, kb)
        self.history: deque[str] = deque(
            maxlen=max(settings.llm.context_turns, settings.llm.help_context_turns)
        )
        self._seq = itertools.count(1)
        self._shown = 0  # n° de l'appel qui possède la fenêtre
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------ entrées
    def on_utterance(self, speaker: str, text: str) -> asyncio.Task | None:
        """Appelée (dans la boucle asyncio) à chaque phrase transcrite terminée."""
        line = f"{speaker} : {text}"
        self.journal.write(line)
        self.display.show_heard(line)
        context = list(self.history)[-self.s.llm.context_turns:]
        self.history.append(line)
        if speaker == ME:
            return None  # ta voix sert de contexte, elle ne déclenche pas de réponse
        analyzed = should_analyze(text, self.s.detection.min_words)
        self.metrics.count_utterance(analyzed)
        if not analyzed:
            return None
        prompt = build_auto_prompt(context, speaker, text, self.kb.passages_for(text))
        return self._spawn(self.respond("auto", text, prompt))

    async def manual_help(self) -> None:
        recent = list(self.history)[-self.s.llm.help_context_turns:]
        if not recent:
            self.display.show_answer("🆘 Aide", "Rien n'a encore été entendu.", "", "")
            return
        last = recent[-1].split(" : ", 1)[-1]
        prompt = build_help_prompt(recent, self.kb.passages_for(" ".join(recent[-3:])))
        await self.respond("help", last, prompt)

    async def typed(self, text: str) -> None:
        self.journal.write(f"(écrit) {text}")
        recent = list(self.history)[-self.s.llm.help_context_turns:]
        prompt = build_typed_prompt(text, recent, self.kb.passages_for(text))
        await self.respond("typed", text, prompt)

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # ------------------------------------------------------------ cœur
    @staticmethod
    def _title(mode: str, kind: str, trigger: str) -> str:
        if mode == "help":
            return f"🆘 Aide · « {trigger[:120]} »"
        if mode == "typed":
            return f"✍️ « {trigger[:120]} »"
        return f"{'🗣️ Demande' if kind == 'DEMANDE' else '❓'} {trigger}"

    async def respond(self, mode: str, trigger: str, prompt: str) -> CallRecord:
        seq = next(self._seq)
        t0 = time.perf_counter()
        forced = mode != "auto"
        header_s = first_s = None
        out, outcome, usage = "", "erreur", Usage(estimated=True)

        if forced:
            self._shown = seq
            self.display.show_answer(self._title(mode, "", trigger), "Réflexion…", "", "")

        try:
            async with self.client.messages.stream(
                model=self.s.llm.model,
                max_tokens=self.s.llm.max_tokens,
                system=self.system,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                superseded = False
                async for chunk in stream.text_stream:
                    out += chunk
                    reply = parse_reply(out)
                    if reply.pending:
                        continue
                    if header_s is None:
                        header_s = time.perf_counter() - t0
                        if reply.is_answer and seq > self._shown:
                            self._shown = seq  # la réponse la plus récente prend la fenêtre
                    if reply.kind == "NON":
                        continue  # « NON » fait 1 ou 2 tokens : on laisse finir pour compter l'usage
                    if self._shown != seq:
                        superseded = True
                        break
                    if reply.body:
                        if first_s is None:
                            first_s = time.perf_counter() - t0
                        self.display.show_answer(self._title(mode, reply.kind, trigger), reply.body, reply.source, "")
                if superseded:
                    outcome = "remplacée"
                    snapshot = getattr(stream, "current_message_snapshot", None)
                    usage = _usage_from(snapshot)
                    usage.output_tokens = max(usage.output_tokens, len(out) // 3)
                    usage.estimated = True
                else:
                    usage = _usage_from(await stream.get_final_message())
                    final = parse_reply(out, final=True)
                    outcome = final.kind or "NON"
                    self._finish(mode, trigger, final, seq, t0, first_s)
        except Exception as e:
            log.exception("Appel au modèle en échec")
            self.display.show_status(f"⚠️ Erreur IA : {type(e).__name__} — {str(e)[:120]}")

        return self.metrics.record(CallRecord(
            mode=mode, outcome=outcome, header_s=header_s, first_text_s=first_s,
            total_s=time.perf_counter() - t0, usage=usage,
        ))

    def _finish(self, mode: str, trigger: str, reply: Reply, seq: int, t0: float, first_s) -> None:
        if reply.kind == "NON":
            if mode != "auto" and self._shown == seq:
                self.display.show_answer(self._title(mode, "", trigger), "Rien de précis n'est attendu de toi pour l'instant.", "", "")
            return
        if seq > self._shown:
            self._shown = seq  # sortie sans en-tête (format non respecté) : décidée à la fin
        if self._shown != seq:
            return
        total = time.perf_counter() - t0
        footer = f"1er mot {first_s if first_s is not None else total:.1f} s · fini {total:.1f} s"
        self.display.show_answer(self._title(mode, reply.kind, trigger), reply.body, reply.source, footer)
        self.journal.write(f"→ {reply.kind} ({reply.source or 'général'}, {footer}) : {reply.body}")

    async def drain(self) -> None:
        """Attend les réponses en cours (utile en fin de réunion et dans les tests)."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
