import json

import pytest

from meeting_copilot.config import PricingConfig
from meeting_copilot.metrics import CallRecord, MeetingMetrics, Usage, percentile


def test_usage_cost():
    p = PricingConfig(llm_input_per_mtok=1, llm_output_per_mtok=5,
                      llm_cache_write_per_mtok=1.25, llm_cache_read_per_mtok=0.1)
    u = Usage(input_tokens=1_000_000, output_tokens=200_000,
              cache_creation_input_tokens=0, cache_read_input_tokens=1_000_000)
    assert u.cost(p) == pytest.approx(1 + 1 + 0.1)


def test_percentile():
    assert percentile([], 50) is None
    assert percentile([3.0], 90) == 3.0
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90) == pytest.approx(9.1)


def test_summary_and_report(tmp_path):
    m = MeetingMetrics(PricingConfig(stt_per_minute=0.01), "modele-test")
    m.count_utterance(True)
    m.count_utterance(False)
    m.add_audio("Interlocuteur", 600)  # 10 min
    m.record(CallRecord("auto", "QUESTION", 0.4, 1.0, 3.0, Usage(input_tokens=1000, output_tokens=100)))
    m.record(CallRecord("auto", "DEMANDE", 0.5, 2.0, 4.0, Usage(input_tokens=1000, output_tokens=200)))
    m.record(CallRecord("auto", "NON", 0.3, None, 0.4, Usage(input_tokens=1000, output_tokens=2)))

    s = m.summary()
    assert s["phrases_transcrites"] == 2 and s["phrases_filtrees_localement"] == 1
    assert s["reponses_affichees"] == 2
    assert s["par_issue"]["NON"] == 1
    assert s["latence_premier_mot_s"]["p50"] == pytest.approx(1.5)
    assert s["cout_usd"]["transcription"] == pytest.approx(0.1)

    md = m.write_report(tmp_path)
    assert md.exists() and "Rapport de réunion" in md.read_text(encoding="utf-8")
    data = json.loads(md.with_suffix(".json").read_text(encoding="utf-8"))
    assert len(data["appels"]) == 3


def test_report_never_raises(tmp_path):
    blocker = tmp_path / "fichier"
    blocker.write_text("x")
    assert MeetingMetrics(PricingConfig()).write_report(blocker / "sous-dossier") is None
