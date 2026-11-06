import pytest

from meeting_copilot.config import ROOT, ConfigError, load_settings
from meeting_copilot.transcription import UtteranceAssembler, build_url


def test_defaults_without_file(tmp_path):
    s = load_settings(tmp_path / "absent.toml")
    assert s.llm.model == "claude-haiku-5-5"
    assert s.docs_dir == ROOT / "demo" / "docs"


def test_overrides(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text('[llm]\nmodel = "autre"\nmax_tokens = 300\n[pricing]\nstt_per_minute = 1\n', encoding="utf-8")
    s = load_settings(f)
    assert (s.llm.model, s.llm.max_tokens, s.pricing.stt_per_minute) == ("autre", 300, 1.0)


@pytest.mark.parametrize("content", ['[llm]\ninconnu = 1\n', '[nope]\na = 1\n', '[llm]\nmax_tokens = "600"\n'])
def test_invalid_config(tmp_path, content):
    f = tmp_path / "config.toml"
    f.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(f)


def test_example_config_is_valid():
    load_settings(ROOT / "config.example.toml")


def test_assembler_joins_final_segments():
    out = []
    a = UtteranceAssembler(out.append)
    a.feed({"type": "Results", "is_final": False, "channel": {"alternatives": [{"transcript": "Combien"}]}})
    a.feed({"type": "Results", "is_final": True, "channel": {"alternatives": [{"transcript": "Combien coûte"}]}})
    a.feed({"type": "Results", "is_final": True, "speech_final": True,
            "channel": {"alternatives": [{"transcript": "l'atelier ?"}]}})
    a.feed({"type": "UtteranceEnd"})  # rien en attente : pas de doublon
    assert out == ["Combien coûte l'atelier ?"]


def test_url_contains_settings():
    s = load_settings(ROOT / "config.example.toml")
    url = build_url(s.transcription, 48000)
    assert "language=fr" in url and "sample_rate=48000" in url and "endpointing=350" in url
