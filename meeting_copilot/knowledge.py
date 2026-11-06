"""Base de connaissances : chargement des documents et sélection du contexte.

Stratégie (volontairement simple) :
- petite base (cas le plus courant : quelques documents de préparation) → tout est
  envoyé au modèle et mis en cache côté API : meilleure qualité, coût marginal faible ;
- grosse base → découpage en passages et sélection des plus pertinents par BM25
  (recherche lexicale, sans dépendance ni modèle d'embedding).
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

SUPPORTED = {".md", ".txt", ".docx", ".pdf"}


@dataclass(frozen=True)
class Document:
    name: str
    text: str


@dataclass(frozen=True)
class Chunk:
    doc: str
    text: str


# ----------------------------------------------------------------- chargement
def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".docx":
        from docx import Document as Docx

        doc = Docx(str(path))
        lines = [p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                lines.append(" | ".join(c.text for c in row.cells))
        return "\n".join(lines)
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    raise ValueError(f"Format non pris en charge : {path.name}")


def load_documents(folder: Path) -> list[Document]:
    """Charge les documents lisibles du dossier ; un fichier illisible est ignoré (et loggé)."""
    if not folder.is_dir():
        log.warning("Dossier de documents introuvable : %s", folder)
        return []
    docs = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in SUPPORTED or path.name.startswith("~$"):
            continue
        try:
            text = read_document(path).strip()
        except Exception as e:  # un document corrompu ne doit pas empêcher le démarrage
            log.warning("Impossible de lire %s : %s", path.name, e)
            continue
        if text:
            docs.append(Document(path.name, text))
        else:
            log.warning("%s est vide (PDF scanné ?) : ignoré", path.name)
    return docs


def load_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore").strip()


# ----------------------------------------------------------------- découpage
def chunk_documents(docs: list[Document], max_chars: int = 800) -> list[Chunk]:
    """Découpe par paragraphes, regroupés jusqu'à ~max_chars (on ne coupe pas une phrase)."""
    chunks: list[Chunk] = []
    for doc in docs:
        current = ""
        for para in re.split(r"\n\s*\n", doc.text):
            para = para.strip()
            if not para:
                continue
            if current and len(current) + len(para) > max_chars:
                chunks.append(Chunk(doc.name, current))
                current = ""
            current = f"{current}\n\n{para}" if current else para
        if current:
            chunks.append(Chunk(doc.name, current))
    return chunks


# ----------------------------------------------------------------- BM25
_STOPWORDS = set(
    "le la les un une des de du d l et ou a au aux en dans sur pour par avec sans ce ces "
    "cet cette est sont etre qui que quoi quel quelle quels quelles vous nous je tu il elle "
    "on ils elles mon ma mes ton ta tes son sa ses votre vos notre nos leur leurs ne pas "
    "plus se s y c qu est-ce comment combien".split()
)


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+", text) if t not in _STOPWORDS and len(t) > 1]


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tokens = [tokenize(c.text) for c in chunks]
        self.avgdl = sum(map(len, self.tokens)) / max(len(self.tokens), 1)
        df: Counter[str] = Counter()
        for toks in self.tokens:
            df.update(set(toks))
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list[Chunk]:
        q = tokenize(query)
        scored = []
        for i, toks in enumerate(self.tokens):
            tf = Counter(toks)
            norm = self.k1 * (1 - self.b + self.b * len(toks) / (self.avgdl or 1))
            score = sum(
                self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf
            )
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [self.chunks[i] for _, i in scored[:k]]


# ----------------------------------------------------------------- base
class KnowledgeBase:
    """Fournit le texte de référence à envoyer au modèle."""

    def __init__(self, docs: list[Document], full_context_max_chars: int, top_k: int):
        self.docs = docs
        self.top_k = top_k
        total = sum(len(d.text) for d in docs)
        self.full_context = total <= full_context_max_chars
        self._index = None if self.full_context else BM25(chunk_documents(docs))
        log.info(
            "%d document(s), %d caractères → mode %s",
            len(docs), total, "contexte complet" if self.full_context else "recherche BM25",
        )

    @property
    def names(self) -> list[str]:
        return [d.name for d in self.docs]

    def static_text(self) -> str:
        """Documents envoyés en entier (mis en cache) — vide en mode recherche."""
        if not self.full_context:
            return ""
        return "\n\n".join(f'<document nom="{d.name}">\n{d.text}\n</document>' for d in self.docs)

    def passages_for(self, query: str) -> str:
        """Passages pertinents pour la question — vide en mode contexte complet."""
        if self.full_context or self._index is None:
            return ""
        hits = self._index.search(query, self.top_k)
        return "\n\n".join(f'<passage document="{c.doc}">\n{c.text}\n</passage>' for c in hits)
