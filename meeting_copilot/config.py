"""Configuration : valeurs par défaut + surcharge optionnelle par config.toml.

Les secrets (clés API) ne sont jamais dans ce fichier : ils viennent des variables
d'environnement (fichier .env chargé au démarrage).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PathsConfig:
    docs_dir: str = "demo/docs"
    context_file: str = "demo/contexte-reunion.example.txt"
    data_dir: str = ""  # vide = %LOCALAPPDATA%\meeting-copilot (ou ./data hors Windows)


@dataclass
class LLMConfig:
    model: str = "claude-haiku-5-5"
    max_tokens: int = 600
    context_turns: int = 8
    help_context_turns: int = 15


@dataclass
class RetrievalConfig:
    full_context_max_chars: int = 60_000
    top_k: int = 6


@dataclass
class TranscriptionConfig:
    model: str = "nova-3"
    language: str = "fr"
    endpointing_ms: int = 350
    listen_mic: bool = True


@dataclass
class DetectionConfig:
    min_words: int = 2


@dataclass
class UIConfig:
    width: int = 560
    height: int = 520
    font_size: int = 15
    auto_grow: bool = True
    hotkey_vk: int = 0x78  # F9


@dataclass
class PricingConfig:
    """Tarifs en USD. À vérifier sur les pages de prix officielles."""

    llm_input_per_mtok: float = 0.10
    llm_output_per_mtok: float = 0.50
    llm_cache_write_per_mtok: float = 0.125
    llm_cache_read_per_mtok: float = 0.01
    stt_per_minute: float = 0.0077


@dataclass
class Settings:
    paths: PathsConfig = field(default_factory=PathsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)

    # --- chemins résolus ---
    def resolve(self, value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else ROOT / p

    @property
    def docs_dir(self) -> Path:
        return self.resolve(self.paths.docs_dir)

    @property
    def context_file(self) -> Path:
        return self.resolve(self.paths.context_file)

    @property
    def data_dir(self) -> Path:
        if self.paths.data_dir:
            return self.resolve(self.paths.data_dir)
        base = os.environ.get("LOCALAPPDATA")
        # Hors de Documents : Windows y bloque souvent l'écriture (accès contrôlé aux dossiers).
        return Path(base) / "meeting-copilot" if base else ROOT / "data"


class ConfigError(ValueError):
    pass


def _apply(section: object, values: dict, prefix: str) -> None:
    known = {f.name: f for f in fields(section)}
    for key, value in values.items():
        if key not in known:
            raise ConfigError(f"Clé inconnue dans config.toml : [{prefix}] {key}")
        current = getattr(section, key)
        if is_dataclass(current):
            raise ConfigError(f"[{prefix}] {key} doit être une section")
        expected = type(current)
        if expected is float and isinstance(value, int):
            value = float(value)
        if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
            raise ConfigError(
                f"[{prefix}] {key} : {expected.__name__} attendu, reçu {type(value).__name__}"
            )
        setattr(section, key, value)


def load_settings(path: Path | None = None) -> Settings:
    """Charge config.toml s'il existe (sinon les valeurs par défaut)."""
    settings = Settings()
    path = path or ROOT / "config.toml"
    if not path.is_file():
        return settings
    with path.open("rb") as f:
        data = tomllib.load(f)
    for name, values in data.items():
        section = getattr(settings, name, None)
        if not is_dataclass(section) or not isinstance(values, dict):
            raise ConfigError(f"Section inconnue dans config.toml : [{name}]")
        _apply(section, values, name)
    return settings
