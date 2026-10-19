"""
Fact catalogue loader and get_brand_facts tool implementation.
Returns rows for a given brand_id from speaker-catalogue-v4.
"""
from __future__ import annotations

import csv
import pathlib
from typing import Any

_FACTS_PATH = pathlib.Path(__file__).parent.parent / "data" / "brand_facts.csv"
_FACT_VERSION = "speaker-catalogue-v4"

# Loaded once at import time
_catalogue: dict[str, list[dict[str, Any]]] = {}


def _load() -> None:
    with open(_FACTS_PATH, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            brand = row["brand_id"]
            _catalogue.setdefault(brand, []).append(dict(row))


_load()

KNOWN_BRANDS: frozenset[str] = frozenset(_catalogue.keys())


def get_brand_facts(brand_id: str) -> dict[str, Any]:
    """
    Tool implementation.  Returns the fact rows for brand_id.
    Raises ValueError for unknown brands (tool-level error).
    """
    if brand_id not in _catalogue:
        raise ValueError(f"Unknown brand_id: {brand_id!r}. Known: {sorted(KNOWN_BRANDS)}")
    return {
        "fact_version": _FACT_VERSION,
        "brand_id": brand_id,
        "facts": _catalogue[brand_id],
    }


# JSON schema for the tool (OpenAI / Anthropic function-calling format)
GET_BRAND_FACTS_TOOL = {
    "type": "function",
    "function": {
        "name": "get_brand_facts",
        "description": (
            "Return all catalogue fact rows for a given brand.  "
            "fact_version is speaker-catalogue-v4.  "
            "Call this for every brand whose products appear in the answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "brand_id": {
                    "type": "string",
                    "description": "Lowercase brand identifier, e.g. cleartrail, stonewave, cloudnote.",
                }
            },
            "required": ["brand_id"],
        },
    },
}
