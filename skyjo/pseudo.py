"""Pseudos of human players (web UI, console, record): kept apart from
skyjo.storage so that the game sessions do not need SQLite."""

from __future__ import annotations

import re
import unicodedata

PSEUDO_MIN, PSEUDO_MAX = 2, 20


def clean_pseudo(raw) -> str:
    """Pseudo kept: letters (accents included), digits, spaces and hyphens, spaces
    collapsed, 2 to 20 characters; other characters are dropped. ValueError if
    nothing valid remains. Nothing prevents two players from using the same one."""
    if not isinstance(raw, str):
        raise ValueError("Le pseudo doit être un texte.")
    text = unicodedata.normalize("NFC", raw)
    text = "".join(c for c in text if c.isalpha() or c.isdigit() or c in " -")
    text = re.sub(r"\s+", " ", text).strip(" ")
    if not PSEUDO_MIN <= len(text) <= PSEUDO_MAX:
        raise ValueError(f"Pseudo invalide : {PSEUDO_MIN} à {PSEUDO_MAX} caractères (lettres, chiffres, "
                         "espaces, tirets).")
    return text
