"""Mesures : latence, tokens et coût par réunion, et rapport de fin de réunion.

Toutes les durées sont mesurées depuis la fin de la phrase transcrite
(le moment où Deepgram renvoie la phrase finale) — voir README, « Ce que mesure la latence ».
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .config import PricingConfig

log = logging.getLogger(__name__)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    estimated: bool = False  # True si l'appel a été interrompu (usage partiellement estimé)

    def cost(self, p: PricingConfig) -> float:
        return (
            self.input_tokens * p.llm_input_per_mtok
            + self.output_tokens * p.llm_output_per_mtok
            + self.cache_creation_input_tokens * p.llm_cache_write_per_mtok
            + self.cache_read_input_tokens * p.llm_cache_read_per_mtok
        ) / 1_000_000


@dataclass
class CallRecord:
    mode: str                  # "auto" | "help" | "typed"
    outcome: str               # "NON" | "QUESTION" | "DEMANDE" | "remplacée" | "erreur"
    header_s: float | None     # décision prise (1re ligne reçue)
    first_text_s: float | None # 1er mot de réponse affiché
    total_s: float             # réponse complète
    usage: Usage = field(default_factory=Usage)
    cost_usd: float = 0.0


def percentile(values: list[float], q: float) -> float | None:
    """Percentile par interpolation linéaire (q entre 0 et 100)."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


class MeetingMetrics:
    def __init__(self, pricing: PricingConfig, model: str = ""):
        self.pricing = pricing
        self.model = model
        self.started_at = datetime.now()
        self._t0 = time.perf_counter()
        self.utterances = 0
        self.skipped_by_prefilter = 0
        self.calls: list[CallRecord] = []
        self.audio_seconds: dict[str, float] = {}

    # --- collecte ---
    def count_utterance(self, analyzed: bool) -> None:
        self.utterances += 1
        if not analyzed:
            self.skipped_by_prefilter += 1

    def add_audio(self, stream: str, seconds: float) -> None:
        self.audio_seconds[stream] = self.audio_seconds.get(stream, 0.0) + seconds

    def record(self, rec: CallRecord) -> CallRecord:
        rec.cost_usd = rec.usage.cost(self.pricing)
        self.calls.append(rec)
        return rec

    # --- synthèse ---
    @property
    def llm_cost(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def stt_cost(self) -> float:
        return sum(self.audio_seconds.values()) / 60 * self.pricing.stt_per_minute

    def summary(self) -> dict:
        duration_s = time.perf_counter() - self._t0
        answers = [c for c in self.calls if c.outcome in ("QUESTION", "DEMANDE")]
        first = [c.first_text_s for c in answers if c.first_text_s is not None]
        total = [c.total_s for c in answers]
        cost = self.llm_cost + self.stt_cost
        hours = duration_s / 3600

        def r(x):
            return None if x is None else round(x, 2)

        return {
            "date": self.started_at.isoformat(timespec="seconds"),
            "modele": self.model,
            "duree_min": round(duration_s / 60, 1),
            "phrases_transcrites": self.utterances,
            "phrases_filtrees_localement": self.skipped_by_prefilter,
            "appels_llm": len(self.calls),
            "reponses_affichees": len(answers),
            "par_issue": {
                k: sum(1 for c in self.calls if c.outcome == k)
                for k in ("NON", "QUESTION", "DEMANDE", "remplacée", "erreur")
            },
            "latence_premier_mot_s": {"p50": r(percentile(first, 50)), "p90": r(percentile(first, 90))},
            "latence_reponse_complete_s": {"p50": r(percentile(total, 50)), "p90": r(percentile(total, 90))},
            "latence_moyenne_premier_mot_s": r(statistics.fmean(first)) if first else None,
            "tokens": {
                "entree": sum(c.usage.input_tokens for c in self.calls),
                "sortie": sum(c.usage.output_tokens for c in self.calls),
                "cache_ecriture": sum(c.usage.cache_creation_input_tokens for c in self.calls),
                "cache_lecture": sum(c.usage.cache_read_input_tokens for c in self.calls),
            },
            "audio_min": {k: round(v / 60, 1) for k, v in self.audio_seconds.items()},
            "cout_usd": {
                "llm": round(self.llm_cost, 4),
                "transcription": round(self.stt_cost, 4),
                "total": round(cost, 4),
                "par_heure": round(cost / hours, 3) if hours > 0.01 else None,
            },
        }

    def write_report(self, folder: Path) -> Path | None:
        """Écrit rapport JSON (détaillé) + Markdown (lisible). Ne lève jamais d'exception."""
        try:
            folder.mkdir(parents=True, exist_ok=True)
            stem = folder / f"rapport_{self.started_at:%Y-%m-%d_%H%M}"
            data = {"resume": self.summary(), "appels": [asdict(c) for c in self.calls]}
            stem.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            stem.with_suffix(".md").write_text(render_markdown(data["resume"]), encoding="utf-8")
            return stem.with_suffix(".md")
        except OSError as e:
            log.error("Rapport non écrit : %s", e)
            return None


def render_markdown(s: dict) -> str:
    lat1, lat2, cost = s["latence_premier_mot_s"], s["latence_reponse_complete_s"], s["cout_usd"]
    return "\n".join([
        f"# Rapport de réunion — {s['date']}",
        "",
        f"- Modèle : `{s['modele']}` · durée : {s['duree_min']} min",
        f"- Phrases transcrites : {s['phrases_transcrites']} "
        f"(dont {s['phrases_filtrees_localement']} filtrées sans appel API)",
        f"- Appels LLM : {s['appels_llm']} · réponses affichées : {s['reponses_affichees']}",
        f"- Issues : {s['par_issue']}",
        "",
        "| Latence (depuis la fin de la phrase) | p50 | p90 |",
        "|---|---|---|",
        f"| 1er mot affiché | {lat1['p50']} s | {lat1['p90']} s |",
        f"| Réponse complète | {lat2['p50']} s | {lat2['p90']} s |",
        "",
        f"Coût : LLM {cost['llm']} $ + transcription {cost['transcription']} $ "
        f"= **{cost['total']} $** (≈ {cost['par_heure']} $/h)",
        "",
    ])
