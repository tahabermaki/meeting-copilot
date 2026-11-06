"""Tests de bout en bout de la logique (faux client, aucun appel réseau)."""

import asyncio

from conftest import FakeClient, FakeDisplay, Script

from meeting_copilot.assistant import ME, OTHERS, Assistant


def make(settings, kb, metrics, *scripts):
    client, display = FakeClient(*scripts), FakeDisplay()
    return Assistant(client, settings, kb, "Contexte fictif", display, metrics), client, display


async def say(assistant, text, speaker=OTHERS):
    task = assistant.on_utterance(speaker, text)
    if task:
        await task
    return task


async def test_question_is_answered_with_source_and_latency(settings, kb, metrics):
    a, client, d = make(settings, kb, metrics, Script("QUESTION | faq.md\n**3 900 € HT.**\nC'est le prix."))
    await say(a, "Combien coûte l'atelier ?")
    title, body, source, footer = d.final
    assert title.startswith("❓") and source == "faq.md" and body.startswith("**3 900")
    assert "1er mot" in footer
    rec = metrics.calls[0]
    assert rec.outcome == "QUESTION" and rec.first_text_s is not None
    assert rec.usage.input_tokens == 1000 and rec.cost_usd > 0


async def test_non_displays_nothing(settings, kb, metrics):
    a, _, d = make(settings, kb, metrics, Script("NON"))
    await say(a, "Je partage mon écran.")
    assert d.answers == []
    assert metrics.calls[0].outcome == "NON"


async def test_answer_starting_with_non_is_displayed(settings, kb, metrics):
    """Régression du bug « Non, … » (le modèle oublie l'en-tête)."""
    a, _, d = make(settings, kb, metrics, Script("Non, nous ne travaillons pas le week-end."))
    await say(a, "Vous livrez le week-end ?")
    assert d.final and "Non, nous" in d.final[1]


async def test_fillers_and_own_voice_cost_nothing(settings, kb, metrics):
    a, client, d = make(settings, kb, metrics)
    assert await say(a, "D'accord, merci.") is None
    assert await say(a, "Je vous présente notre offre.", speaker=ME) is None
    assert client.calls == []
    assert metrics.skipped_by_prefilter == 1
    assert any("Moi :" in h for h in d.heard)  # la voix de l'utilisateur est bien gardée en contexte
    assert "Moi : Je vous présente" in a.history[-1]


async def test_documents_and_context_are_in_cached_system_prompt(settings, kb, metrics):
    a, client, _ = make(settings, kb, metrics, Script("NON"))
    await say(a, "Une question sur le prix ?")
    system = client.calls[0]["system"]
    assert system[-1]["cache_control"] == {"type": "ephemeral"}
    assert "Contexte fictif" in system[-1]["text"] and "faq.md" in system[-1]["text"]


async def test_newer_answer_supersedes_older(settings, kb, metrics):
    slow = Script("QUESTION | général\n" + "Réponse lente. " * 20, chunk=4, delay=0.01)
    fast = Script("QUESTION | général\nRéponse rapide.", delay=0.0)
    a, _, d = make(settings, kb, metrics, slow, fast)
    t1 = a.on_utterance(OTHERS, "Première question longue ?")
    await asyncio.sleep(0.05)  # la 1re réponse a commencé à s'afficher
    t2 = a.on_utterance(OTHERS, "Deuxième question ?")
    await asyncio.gather(t1, t2)
    assert "Réponse rapide" in d.final[1]
    outcomes = sorted(c.outcome for c in metrics.calls)
    assert outcomes == ["QUESTION", "remplacée"]


async def test_help_and_typed_modes(settings, kb, metrics):
    a, client, d = make(settings, kb, metrics,
                        Script("DEMANDE | contexte\nJe suis Alex."), Script("QUESTION | général\nPlus court."))
    await a.manual_help()  # historique vide : pas d'appel
    assert client.calls == [] and "Rien n'a encore été entendu" in d.answers[-1][1]
    a.history.append(f"{OTHERS} : Je vous laisse vous présenter.")
    await a.manual_help()
    assert d.final[0].startswith("🆘") and "Alex" in d.final[1]
    await a.typed("plus court")
    assert d.final[0].startswith("✍️")
    assert "plus court" in client.calls[-1]["messages"][0]["content"]


async def test_api_error_is_reported_not_raised(settings, kb, metrics):
    a, _, d = make(settings, kb, metrics, Script("", error=RuntimeError("panne réseau")))
    await say(a, "Quel est le délai ?")
    assert any("panne réseau" in s for s in d.statuses)
    assert metrics.calls[0].outcome == "erreur"
