from pathlib import Path

from meeting_copilot.config import ROOT
from meeting_copilot.knowledge import (
    BM25,
    Document,
    KnowledgeBase,
    chunk_documents,
    load_documents,
    tokenize,
)


def test_load_documents_md_txt_docx_and_skips_bad(tmp_path: Path):
    from docx import Document as Docx

    (tmp_path / "a.md").write_text("# Offre\nAtelier de cadrage", encoding="utf-8")
    (tmp_path / "b.txt").write_text("FAQ", encoding="utf-8")
    d = Docx()
    d.add_paragraph("Contenu Word")
    d.add_table(rows=1, cols=2).rows[0].cells[0].text = "cellule"
    d.save(tmp_path / "c.docx")
    (tmp_path / "vide.txt").write_text("   ", encoding="utf-8")
    (tmp_path / "casse.docx").write_bytes(b"pas un docx")
    (tmp_path / "~$temp.docx").write_bytes(b"fichier verrou Word")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    docs = load_documents(tmp_path)
    assert [x.name for x in docs] == ["a.md", "b.txt", "c.docx"]
    assert "cellule" in docs[2].text


def test_missing_folder_returns_empty(tmp_path):
    assert load_documents(tmp_path / "absent") == []


def test_tokenize_removes_accents_and_stopwords():
    assert tokenize("Où sont hébergées les Données ?") == ["hebergees", "donnees"]


def test_chunking_groups_paragraphs_without_splitting():
    text = "\n\n".join(f"Paragraphe {i} " + "x" * 300 for i in range(5))
    chunks = chunk_documents([Document("d.md", text)], max_chars=800)
    assert len(chunks) == 3
    assert all(c.doc == "d.md" for c in chunks)
    assert sum(c.text.count("Paragraphe") for c in chunks) == 5


def test_bm25_ranks_relevant_passage_first():
    docs = [
        Document("prix.md", "Le prix de l'atelier de cadrage est de 3 900 euros."),
        Document("hebergement.md", "Les données sont hébergées dans l'Union européenne."),
        Document("equipe.md", "L'équipe compte six personnes dont quatre ingénieurs."),
    ]
    hits = BM25(chunk_documents(docs)).search("Où hébergez-vous les données ?", k=2)
    assert hits[0].doc == "hebergement.md"


def test_full_context_mode_for_small_base():
    kb = KnowledgeBase([Document("a.md", "court")], full_context_max_chars=1000, top_k=3)
    assert kb.full_context
    assert '<document nom="a.md">' in kb.static_text()
    assert kb.passages_for("question") == ""


def test_search_mode_for_large_base():
    big = [Document(f"d{i}.md", f"Sujet {i}. " + "texte " * 200) for i in range(10)]
    big.append(Document("cible.md", "La garantie de service est de 99,9 % de disponibilité."))
    kb = KnowledgeBase(big, full_context_max_chars=1000, top_k=2)
    assert not kb.full_context
    assert kb.static_text() == ""
    assert 'document="cible.md"' in kb.passages_for("Quelle garantie de disponibilité ?")


def test_demo_documents_load_and_answer_typical_questions():
    """Les documents de démo du dépôt sont lisibles et couvrent les questions de démo."""
    docs = load_documents(ROOT / "demo" / "docs")
    assert len(docs) >= 3
    index = BM25(chunk_documents(docs))
    top = index.search("Quel est le prix de l'atelier de cadrage ?", 2)
    assert any("3 900" in c.text for c in top)
    assert index.search("Où sont hébergées les données ?", 1)[0].doc == "faq-commerciale.md"
