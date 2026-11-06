"""Détection : prompts, pré-filtre local et lecture de la réponse du modèle.

Ce module est pur (aucun appel réseau) : c'est la partie la plus testée.

Principe : un seul appel au modèle fait à la fois la classification
(rien à répondre / question précise / demande ouverte) et la rédaction de la
réponse. La 1re ligne de sortie est un en-tête lisible par la machine :
    NON
    QUESTION | <source>
    DEMANDE | <source>
On attend la fin de cette ligne avant de décider : c'est ce qui évite le bug
d'une version précédente où toute réponse commençant par « Non, … » était jetée.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

SYSTEM_PROMPT = """Tu es le copilote de l'utilisateur pendant ses réunions \
(rendez-vous commerciaux, réunions d'équipe, points clients). Tu reçois la \
transcription automatique de ce que disent ses interlocuteurs. Dès qu'on attend \
quelque chose de lui, tu proposes une réponse claire, prête à être dite à voix \
haute, à la première personne ("je", "nous").

Classe la dernière phrase :
1. NON : rien n'est attendu de l'utilisateur (salutation, affirmation, réflexion \
à voix haute, logistique du type "tu m'entends ?" ou "je partage mon écran", \
question rhétorique, sujet déjà traité juste avant).
2. QUESTION : demande précise (un chiffre, une date, un oui/non, "est-ce que vous \
savez faire X ?", "combien de temps pour… ?").
3. DEMANDE : invitation ouverte à prendre la parole, même sans point \
d'interrogation : "présentez-vous", "parlez-nous de…", "je vous laisse \
présenter…", "qu'est-ce que vous proposez ?", "comment vous voyez la suite ?", \
"vous avez un retour là-dessus ?", "à vous".

Format de sortie, sans rien d'autre :

Cas 1 :
NON

Cas 2 :
QUESTION | <source>
**<la réponse en une phrase courte>**
<1 ou 2 phrases à dire, style oral, qui complètent>
- <point concret : chiffre, étape, exemple>
- <point concret>

Cas 3 :
DEMANDE | <source>
<2 à 4 phrases à dire, style oral naturel, mots-clés en **gras**>
- <point concret à dérouler>
- <point concret à dérouler>
- <point concret à dérouler>
**Si on creuse :** <une phrase avec un détail ou un exemple en plus>

<source> = nom du document utilisé, "contexte" si tu t'appuies sur le contexte de \
réunion, ou "général" si tu réponds avec tes connaissances.

Règles :
- La première phrase répond à la question. Pas de "Vous pourriez dire…".
- Longueur : 60 à 180 mots. Phrases de 20 mots maximum. Pas de titres, de \
tableaux ni de blocs de code. Termine toujours tes phrases.
- Appuie-toi d'abord sur le contexte de réunion et les documents, et adapte le \
ton aux interlocuteurs et aux consignes du contexte.
- Tu peux compléter avec la culture générale ou technique, mais n'invente jamais \
un fait vérifiable propre à l'utilisateur ou à son entreprise (expérience, \
client, chiffre, prix, délai, engagement) : écris "[à compléter]" à la place.
- Respecte les points sensibles indiqués dans le contexte.
- Les lignes "Moi :" sont ce que l'utilisateur a déjà dit : ne le répète pas, \
complète-le.
- La transcription peut contenir des erreurs : interprète le sens voulu.
- Toujours en français."""

HELP_INSTRUCTION = """L'UTILISATEUR A DEMANDÉ DE L'AIDE (bouton ou touche F9).
Identifie ce qu'on attend de lui dans les derniers échanges : la dernière \
question ou demande, même implicite. Réponds en QUESTION ou en DEMANDE, jamais \
NON. Si rien n'est clairement demandé, réponds en DEMANDE pour rebondir sur le \
dernier sujet abordé."""

TYPED_INSTRUCTION = """L'UTILISATEUR T'ÉCRIT DIRECTEMENT (il ne l'a pas dit à voix haute) : « {text} »
Réponds à sa demande en QUESTION ou en DEMANDE, jamais NON. Si c'est une \
consigne (ex. « plus court », « reformule pour un client non technique »), \
applique-la à la dernière réponse attendue dans la réunion."""

KINDS = ("QUESTION", "DEMANDE")

# ------------------------------------------------------------------ pré-filtre
# Phrases qui ne demandent jamais de réponse : on évite un appel API (coût + bruit).
FILLERS = {
    "oui", "non", "ok", "okay", "d'accord", "dac", "merci", "merci beaucoup", "ouais",
    "parfait", "super", "exactement", "tout a fait", "bien sur", "bonjour", "bonsoir",
    "au revoir", "a bientot", "allo", "hum", "euh", "voila", "tres bien", "c'est note",
    "je vois", "absolument", "entendu", "ca marche",
}


# Mots qui, seuls ou combinés entre eux (« d'accord, merci »), ne demandent pas de réponse.
FILLER_WORDS = {w for phrase in FILLERS for w in phrase.split()} | {"beaucoup", "alors", "bon"}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^\w' ?-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def should_analyze(text: str, min_words: int = 2) -> bool:
    """Pré-filtre local : faut-il envoyer cette phrase au modèle ?"""
    norm = _normalize(text)
    bare = norm.rstrip(" ?").strip()
    if not bare or bare in FILLERS:
        return False
    words = bare.split()
    if len(words) <= 4 and not norm.endswith("?") and all(w in FILLER_WORDS for w in words):
        return False
    if len(bare.split()) >= min_words:
        return True
    # phrase très courte : on garde les impératifs composés (« présentez-vous »)
    # et les vraies questions (« Délais ? »)
    return "-" in bare or norm.endswith("?")


# ------------------------------------------------------------------ prompts
def build_auto_prompt(context: list[str], speaker: str, text: str, passages: str = "") -> str:
    parts = []
    if passages:
        parts.append("Passages pertinents des documents :\n" + passages)
    parts.append("Contexte récent :\n" + ("\n".join(context) if context else "(début de réunion)"))
    parts.append(f"Dernière phrase ({speaker}) : « {text} »")
    return "\n\n".join(parts)


def build_help_prompt(recent: list[str], passages: str = "") -> str:
    parts = [HELP_INSTRUCTION]
    if passages:
        parts.append("Passages pertinents des documents :\n" + passages)
    parts.append("Derniers échanges (du plus ancien au plus récent) :\n" + "\n".join(recent))
    return "\n\n".join(parts)


def build_typed_prompt(text: str, recent: list[str], passages: str = "") -> str:
    parts = [TYPED_INSTRUCTION.format(text=text)]
    if passages:
        parts.append("Passages pertinents des documents :\n" + passages)
    parts.append("Derniers échanges de la réunion :\n" + ("\n".join(recent) if recent else "(rien encore)"))
    return "\n\n".join(parts)


# ------------------------------------------------------------------ lecture
@dataclass(frozen=True)
class Reply:
    kind: str  # "" (en-tête pas encore reçu) | "NON" | "QUESTION" | "DEMANDE"
    source: str = ""
    body: str = ""

    @property
    def pending(self) -> bool:
        return self.kind == ""

    @property
    def is_answer(self) -> bool:
        return self.kind in KINDS


def _header_kind(header: str) -> str:
    return header.strip().strip("*#_`.:!- ").upper()


def parse_reply(text: str, final: bool = False) -> Reply:
    """Lit la sortie (éventuellement partielle) du modèle.

    Tant que la 1re ligne n'est pas terminée (et que le flux n'est pas fini),
    on ne décide rien : Reply(kind="").
    """
    text = text.lstrip()
    if "\n" in text:
        header, body = text.split("\n", 1)
    elif final:
        header, body = text, ""
    else:
        return Reply("")

    head, sep, source = header.partition("|")
    kind = _header_kind(head)
    if kind == "NON":  # la ligne entière est « NON » (pas « Non, nous… »)
        return Reply("NON")
    if kind in KINDS:
        return Reply(kind, source.strip(), body.strip())
    # Format non respecté : on montre tout plutôt que de perdre une réponse.
    return Reply("QUESTION", "", text.strip())
