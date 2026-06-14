"""Message-log helpers for cross-conversation history rendering."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.services.turn_contract import format_relative_time_label
from memu.utils.conversation import format_grouped_chat_history, normalize_whatsapp_identifier


def derive_source_label(conversation_id: str) -> str:
    cid = str(conversation_id or "").strip()
    if cid.startswith("whatsapp:"):
        suffix = cid.split(":", 1)[1] if ":" in cid else ""
        if "@g.us" in suffix:
            return "whatsapp:group"
        return "whatsapp:dm"
    if cid.startswith(("sillytavern", "integrity:", "chat:")):
        return "sillytavern"
    if cid.startswith("cron:"):
        return "cron"
    return cid.split(":")[0] if ":" in cid else "unknown"


DEFAULT_CROSS_RECENT_FALLBACK_MESSAGES = 8
_normalize_whatsapp_identifier = normalize_whatsapp_identifier


def _load_whatsapp_directory_names() -> dict[str, str]:
    hermes_home = Path(os.getenv("HERMES_HOME") or "~/.hermes").expanduser().resolve()
    directory_path = hermes_home / "channel_directory.json"
    out: dict[str, str] = {}
    if directory_path.exists():
        try:
            raw = json.loads(directory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        platforms = raw.get("platforms") if isinstance(raw, dict) else None
        rows = platforms.get("whatsapp") if isinstance(platforms, dict) else None
        if not isinstance(rows, list):
            rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("id") or "").strip()
            rname = str(row.get("name") or "").strip()
            if not rid or not rname:
                continue
            out[rid] = rname
            normalized = normalize_whatsapp_identifier(rid)
            if normalized and normalized not in out:
                out[normalized] = rname

    # memU Stack can enrich raw group ids with bridge-resolved names and
    # persist them here. Use that cache so payload headings match launcher UI.
    cache_path = hermes_home / "whatsapp_group_names.json"
    if cache_path.exists():
        try:
            cache_raw = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cache_raw = {}
        if isinstance(cache_raw, dict):
            for key, value in cache_raw.items():
                chat_id = str(key or "").strip()
                group_name = str(value or "").strip()
                if not chat_id or not group_name:
                    continue
                out[chat_id] = group_name
                normalized = normalize_whatsapp_identifier(chat_id)
                if normalized:
                    out[normalized] = group_name
    return out


def format_merged_history(messages: list[dict[str, Any]]) -> str:
    """Format merged messages as grouped markdown for the soul's cross-chat context."""
    return format_grouped_chat_history(
        messages,
        whatsapp_names=_load_whatsapp_directory_names(),
        time_label_resolver=format_relative_time_label,
    )
