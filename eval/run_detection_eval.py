"""Évaluation de la détection sur un jeu de phrases annotées (vrais appels API).

Usage :  python -m eval.run_detection_eval [--model claude-haiku-5-5] [--limit 10]

Passe par exactement le même code que l'application (pré-filtre, prompts,
`Assistant.respond`, lecture de la réponse), avec les documents de démo.
Produit eval/results/<modèle>.md (+ .json) : matrice de confusion, précision /
rappel, latence et coût. Coût d'un passage complet : quelques centimes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from meeting_copilot.assistant import OTHERS, Assistant
from meeting_copilot.config import ROOT, load_settings
from meeting_copilot.detection import build_auto_prompt, should_analyze
from meeting_copilot.knowledge import KnowledgeBase, load_documents, load_text
from meeting_copilot.metrics import MeetingMetrics, percentile

LABELS = ("NON", "QUESTION", "DEMANDE")
HERE = Path(__file__).parent


class NullDisplay:
    def show_answer(self, *a): pass
    def show_status(self, m): print("  statut :", m, file=sys.stderr)
    def show_heard(self, t): pass


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(pairs: list[tuple[str, str]]) -> dict:
    """pairs = [(attendu, prédit)]. Classes + tâche binaire « faut-il répondre ? »."""
    confusion = Counter(pairs)
    per_class = {}
    for c in LABELS:
        tp = confusion[(c, c)]
        fp = sum(v for (g, p), v in confusion.items() if p == c and g != c)
        fn = sum(v for (g, p), v in confusion.items() if g == c and p != c)
        per_class[c] = dict(zip(("precision", "rappel", "f1"), prf(tp, fp, fn), strict=True))
    tp = sum(v for (g, p), v in confusion.items() if g != "NON" and p != "NON")
    fp = sum(v for (g, p), v in confusion.items() if g == "NON" and p != "NON")
    fn = sum(v for (g, p), v in confusion.items() if g != "NON" and p == "NON")
    binary = dict(zip(("precision", "rappel", "f1"), prf(tp, fp, fn), strict=True))
    accuracy = sum(v for (g, p), v in confusion.items() if g == p) / max(len(pairs), 1)
    return {
        "exactitude": accuracy,
        "repondre_ou_non": binary,
        "par_classe": per_class,
        "confusion": {f"{g}->{p}": v for (g, p), v in sorted(confusion.items())},
    }


async def run(model: str, limit: int | None) -> dict:
    load_dotenv(ROOT / ".env")
    from anthropic import AsyncAnthropic

    settings = load_settings(ROOT / "config.example.toml")
    settings.llm.model = model
    docs = load_documents(ROOT / "demo" / "docs")
    kb = KnowledgeBase(docs, settings.retrieval.full_context_max_chars, settings.retrieval.top_k)
    context = load_text(ROOT / "demo" / "contexte-reunion.example.txt")
    metrics = MeetingMetrics(settings.pricing, model)
    assistant = Assistant(AsyncAnthropic(), settings, kb, context, NullDisplay(), metrics)

    rows = [json.loads(line) for line in (HERE / "detection_dataset.jsonl").read_text(encoding="utf-8").splitlines() if line]
    rows = rows[:limit] if limit else rows
    pairs, details = [], []
    for i, row in enumerate(rows, 1):
        if not should_analyze(row["text"], settings.detection.min_words):
            pred, rec = "NON", None  # filtré localement, comme en production
        else:
            prompt = build_auto_prompt(row["context"], OTHERS, row["text"], kb.passages_for(row["text"]))
            rec = await assistant.respond("auto", row["text"], prompt)
            pred = rec.outcome if rec.outcome in LABELS else "ERREUR"
        pairs.append((row["label"], pred))
        mark = "✓" if pred == row["label"] else "✗"
        print(f"{i:>2}/{len(rows)} {mark} attendu={row['label']:<8} prédit={pred:<8} {row['text']}")
        details.append({**row, "predit": pred,
                        "premier_mot_s": rec.first_text_s if rec else None,
                        "decision_s": rec.header_s if rec else None})

    answered = [c for c in metrics.calls if c.first_text_s is not None]
    decisions = [c.header_s for c in metrics.calls if c.header_s is not None]
    result = {
        "modele": model,
        "date": datetime.now().isoformat(timespec="seconds"),
        "n": len(rows),
        **score(pairs),
        "latence_s": {
            "decision_p50": percentile(decisions, 50),
            "premier_mot_p50": percentile([c.first_text_s for c in answered], 50),
            "premier_mot_p90": percentile([c.first_text_s for c in answered], 90),
            "complet_p50": percentile([c.total_s for c in answered], 50),
        },
        "cout_usd_total": metrics.llm_cost,
        "cout_usd_par_appel": metrics.llm_cost / max(len(metrics.calls), 1),
        "details": details,
    }
    return result


def to_markdown(r: dict) -> str:
    def pct(x):
        return f"{x * 100:.0f} %"

    def sec(x):
        return "—" if x is None else f"{x:.2f} s"

    b, lat = r["repondre_ou_non"], r["latence_s"]
    errors = [f"- attendu {d['label']}, prédit {d['predit']} : « {d['text']} »"
              for d in r["details"] if d["label"] != d["predit"]] or ["- aucune"]
    lines = [
        f"# Évaluation de la détection — `{r['modele']}` ({r['date']})",
        "",
        f"Jeu : {r['n']} phrases annotées à la main (`eval/detection_dataset.jsonl`), documents de démo.",
        "",
        "| Mesure | Valeur |",
        "|---|---|",
        f"| Exactitude (3 classes) | {pct(r['exactitude'])} |",
        f"| « Faut-il répondre ? » — précision | {pct(b['precision'])} |",
        f"| « Faut-il répondre ? » — rappel | {pct(b['rappel'])} |",
        f"| Décision (1re ligne reçue), p50 | {sec(lat['decision_p50'])} |",
        f"| 1er mot affiché, p50 / p90 | {sec(lat['premier_mot_p50'])} / {sec(lat['premier_mot_p90'])} |",
        f"| Réponse complète, p50 | {sec(lat['complet_p50'])} |",
        f"| Coût moyen par appel | {r['cout_usd_par_appel']:.5f} $ |",
        "",
        "| Classe | Précision | Rappel | F1 |",
        "|---|---|---|---|",
        *[f"| {c} | {pct(v['precision'])} | {pct(v['rappel'])} | {pct(v['f1'])} |" for c, v in r["par_classe"].items()],
        "",
        "Erreurs :",
        "",
        *errors,
        "",
        "Latences mesurées depuis la réception de la phrase (hors transcription), "
        "réseau du poste de test inclus.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=load_settings().llm.model)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    result = asyncio.run(run(args.model, args.limit))
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"{args.model}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md = to_markdown(result)
    (out / f"{args.model}.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
