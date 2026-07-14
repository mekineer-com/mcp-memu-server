"""Config schema drift test — keep config.json ↔ config.example.json in sync.

`config.json` is the live runtime config (operator-owned, gitignored).
`config.example.json` is the documented shape (committed).

When an operator's live config grows a key that was never added to the
example, new operators can't reproduce the setup. When the example grows
a key the code no longer reads, operators set it in good faith and wonder
why it has no effect.

The test below catches drift between the two committed-representative
runtime configs. It's a soft-lock: a legitimate schema change is a
two-file edit, which is the right amount of friction.

If runtime-only keys are legitimately operator-local (e.g., secrets),
list them in `_ALLOWED_RUNTIME_ONLY` with a justification comment.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import default_config

_ROOT = Path(__file__).resolve().parent.parent
_RUNTIME_CONFIG = _ROOT / "config.json"
_EXAMPLE_CONFIG = _ROOT / "config.example.json"

# Keys that may exist in the runtime config but aren't part of the
# example's schema (operator-local tweaks, secrets). Keep this list tight.
_ALLOWED_RUNTIME_ONLY: set[str] = set()

# Keys that may exist in the example but not in the current runtime
# (operator chose the default, hasn't overridden). These are safe — the
# example is the documentation, not the runtime's mirror.
# (No allow-list needed here; we only assert example ⊇ runtime.)


def _dot_keys(obj: object, prefix: str = "") -> set[str]:
    """Return every dotted key path in a JSON dict (recursive)."""
    out: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            out.add(path)
            out.update(_dot_keys(value, path))
    return out


def _load(path: Path) -> set[str]:
    return _dot_keys(json.loads(path.read_text()))


def test_runtime_config_keys_are_documented_in_example() -> None:
    runtime = _load(_RUNTIME_CONFIG)
    example = _load(_EXAMPLE_CONFIG)

    undocumented = runtime - example - _ALLOWED_RUNTIME_ONLY
    assert not undocumented, (
        "config.json has keys missing from config.example.json. "
        "Either add them to the example (preferred — documents the schema) "
        "or justify + list them in _ALLOWED_RUNTIME_ONLY with a comment.\n"
        f"Undocumented: {sorted(undocumented)}"
    )


def test_example_is_valid_json_and_non_empty() -> None:
    example = json.loads(_EXAMPLE_CONFIG.read_text())
    assert isinstance(example, dict)
    assert example, "config.example.json must be non-empty"
    # Spot-check: a few top-level sections must exist
    for required in ("llm", "storage", "memu", "listen"):
        assert required in example, f"config.example.json missing top-level '{required}'"


def test_default_whatsapp_paths_point_at_channels_data() -> None:
    hermes = default_config()["hermes"]

    assert hermes["home"].endswith("/hermes-channels/data")
    assert hermes["state_db_path"].endswith("/hermes-channels/data/state.db")
    assert hermes["sessions_index_path"].endswith("/hermes-channels/data/sessions/sessions.json")
    assert hermes["whatsapp_web_source_db"].endswith("/hermes-channels/data/whatsapp/web_source.db")
    assert "whatsapp_history_source" not in hermes


def test_default_dynamic_category_cluster_size_matches_engine() -> None:
    assert default_config()["categories"]["dynamic_category_cluster_size"] == 10
