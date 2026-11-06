import pytest

from meeting_copilot.detection import build_auto_prompt, parse_reply, should_analyze


# ------------------------------------------------------------- parse_reply
class TestParseReply:
    def test_pending_until_first_line_complete(self):
        assert parse_reply("QUESTI").pending
        assert parse_reply("QUESTION | faq.md").pending  # pas encore de retour à la ligne

    def test_non(self):
        assert parse_reply("NON\n").kind == "NON"
        assert parse_reply("NON", final=True).kind == "NON"
        assert parse_reply("Non.", final=True).kind == "NON"

    def test_answer_starting_with_non_is_not_dropped(self):
        """Régression : une réponse commençant par « Non, … » était jetée."""
        reply = parse_reply("Non, nous ne travaillons pas le week-end.\nMais…", final=True)
        assert reply.is_answer
        assert "Non, nous" in reply.body

    def test_question_with_source(self):
        r = parse_reply("QUESTION | faq-commerciale.md\n**3 900 € HT.**\nC'est le prix de l'atelier.")
        assert (r.kind, r.source) == ("QUESTION", "faq-commerciale.md")
        assert r.body.startswith("**3 900 €")

    def test_demande(self):
        r = parse_reply("DEMANDE | contexte\nJe suis Alex.\n- point")
        assert r.kind == "DEMANDE" and r.source == "contexte" and "- point" in r.body

    def test_header_without_pipe_or_with_markdown(self):
        assert parse_reply("QUESTION\nRéponse").kind == "QUESTION"
        assert parse_reply("**DEMANDE** | général\nTexte").kind == "DEMANDE"

    def test_malformed_output_is_shown_not_lost(self):
        r = parse_reply("Voici la réponse : 42 jours.\nDétail", final=True)
        assert r.kind == "QUESTION" and "42 jours" in r.body

    def test_leading_whitespace(self):
        assert parse_reply("\n  NON\n").kind == "NON"


# ------------------------------------------------------------- pré-filtre
@pytest.mark.parametrize("text", ["D'accord.", "D'accord, merci.", "Oui, très bien.", "Merci beaucoup !", "OK", "Euh…", "Tout à fait.", "  "])
def test_prefilter_skips_fillers(text):
    assert not should_analyze(text)


@pytest.mark.parametrize("text", [
    "Présentez-vous.",                 # 1 mot composé
    "Délais ?",                        # 1 mot mais vraie question
    "Combien coûte l'atelier ?",
    "Je vous laisse présenter votre démarche.",
])
def test_prefilter_keeps_requests(text):
    assert should_analyze(text)


def test_prefilter_min_words():
    assert not should_analyze("Intéressant.", min_words=2)
    assert should_analyze("Très intéressant votre approche", min_words=2)


def test_prompt_contains_context_and_passages():
    p = build_auto_prompt(["Interlocuteur : Bonjour"], "Interlocuteur", "Et le prix ?", "<passage>x</passage>")
    assert "Bonjour" in p and "« Et le prix ? »" in p and "<passage>" in p
    assert "(début de réunion)" in build_auto_prompt([], "Interlocuteur", "Et le prix ?")
