import json

import pytest

from eval.run_detection_eval import LABELS, score
from meeting_copilot.config import ROOT


def test_score_perfect_and_binary():
    pairs = [("NON", "NON"), ("QUESTION", "QUESTION"), ("DEMANDE", "DEMANDE")]
    s = score(pairs)
    assert s["exactitude"] == 1.0 and s["repondre_ou_non"]["f1"] == 1.0


def test_score_counts_false_positive_and_class_confusion():
    pairs = [("NON", "QUESTION"), ("QUESTION", "DEMANDE"), ("DEMANDE", "DEMANDE"), ("QUESTION", "NON")]
    s = score(pairs)
    b = s["repondre_ou_non"]
    # 2 réponses légitimes affichées sur 3 affichées ; 2 trouvées sur 3 attendues
    assert b["precision"] == pytest.approx(2 / 3) and b["rappel"] == pytest.approx(2 / 3)
    assert s["par_classe"]["DEMANDE"]["precision"] == pytest.approx(0.5)


def test_dataset_is_well_formed_and_balanced():
    rows = [json.loads(x) for x in (ROOT / "eval" / "detection_dataset.jsonl").read_text(encoding="utf-8").splitlines() if x]
    assert len(rows) >= 40
    assert {r["label"] for r in rows} == set(LABELS)
    for r in rows:
        assert r["text"].strip() and isinstance(r["context"], list)
    counts = {c: sum(r["label"] == c for r in rows) for c in LABELS}
    assert min(counts.values()) >= 10
