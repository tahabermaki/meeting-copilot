"""Journal texte de la réunion (transcription + réponses), stocké localement.

Il contient les paroles d'autres personnes : il reste sur le poste, n'est jamais
versionné, et l'écriture est désactivable (journal = None).
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


class Journal:
    def __init__(self, path: Path | None):
        self.path = path
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                log.warning("Journal désactivé (%s) : %s", path.parent, e)
                self.path = None

    def write(self, line: str) -> None:
        if self.path is None:
            return
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(f"[{datetime.now():%H:%M:%S}] {line}\n")
        except OSError as e:
            log.warning("Journal désactivé : %s", e)
            self.path = None
