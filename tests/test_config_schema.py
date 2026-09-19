"""Checks for the minimal installable config example and server defaults."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import default_config

_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLE_CONFIG = _ROOT / "config.example.json"


def test_example_is_valid_json_and_non_empty() -> None:
    example = json.loads(_EXAMPLE_CONFIG.read_text())
    assert isinstance(example, dict)
    assert example, "config.example.json must be non-empty"
    # setup_install.py fills paths inside these four sections.
    for required in ("llm", "storage", "memu", "listen"):
        assert required in example, f"config.example.json missing top-level '{required}'"
    assert example["llm"]["embedding"]["embed_model"]
    assert example["storage"]["metadata_store"]["dsn"]
    assert example["memorize"]["semantic_dedupe_enabled"] is False


def test_default_whatsapp_paths_point_at_channels_data() -> None:
    hermes = default_config()["hermes"]

    assert hermes["home"].endswith("/hermes-channels/data")
    assert hermes["state_db_path"].endswith("/hermes-channels/data/state.db")
    assert hermes["sessions_index_path"].endswith("/hermes-channels/data/sessions/sessions.json")
    assert hermes["whatsapp_web_source_db"].endswith("/hermes-channels/data/whatsapp/web_source.db")
    assert "whatsapp_history_source" not in hermes


def test_default_dynamic_category_cluster_size_matches_engine() -> None:
    assert default_config()["categories"]["dynamic_category_cluster_size"] == 10
