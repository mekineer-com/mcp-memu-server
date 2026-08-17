"""Contract test: /config's masked api_key must be ASCII-safe.

History: the mask used U+2026 (ellipsis "…"). If a client forwarded the
masked value to /memorize (expecting a working api_key), the HTTP client
crashed with UnicodeEncodeError when building headers — HTTP headers are
ASCII-only. The SillyTavern plugin reads config.json directly so this
never bit Marcos; only external smoke drivers hit it.

Fix: mask with three plain ASCII dots ("sk-xxx...yyy"). This test pins
the contract: the masked value only contains ASCII bytes.
"""
from __future__ import annotations

from app.main import _mask_config


def test_masked_api_key_is_ascii_safe() -> None:
    cfg = {"llm": {"api_key": "sk-abcdef1234567890xyz", "base_url": "x"}}
    masked = _mask_config(cfg)
    k = masked["llm"]["api_key"]
    assert k != cfg["llm"]["api_key"], "mask did not alter the key"
    # This is the invariant: must encode as ASCII for HTTP header use.
    k.encode("ascii")  # raises UnicodeEncodeError if not ASCII


def test_mask_preserves_first_and_last_four() -> None:
    cfg = {"llm": {"api_key": "sk-abcd1234XYZwxyz"}}
    masked = _mask_config(cfg)
    assert masked["llm"]["api_key"].startswith("sk-a")
    assert masked["llm"]["api_key"].endswith("wxyz")


def test_mask_noop_on_empty_key() -> None:
    cfg = {"llm": {"api_key": ""}}
    masked = _mask_config(cfg)
    assert masked["llm"]["api_key"] == ""


def test_mentra_secrets_are_always_redacted() -> None:
    cfg = {
        "llm": {"api_key": "llm-secret"},
        "mentra": {
            "gemini_api_key": "gemini-secret",
            "integration_bearer_token": "bearer-secret",
        },
    }

    masked = _mask_config(cfg, include_llm_secret=True)

    assert masked["llm"]["api_key"] == "llm-secret"
    assert masked["mentra"]["gemini_api_key"] == "***"
    assert masked["mentra"]["integration_bearer_token"] == "***"
