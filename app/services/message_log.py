"""Message-log helpers for cross-conversation history rendering."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from app.services.turn_contract import format_relative_time_label

_SHARED_GROUP_PREFIX_RE = re.compile(r"^\[([^\]]+)\]\s+(.+)$")


def _parse_shared_group_sender_prefix(content: str) -> tuple[str, str] | None:
    match = _SHARED_GROUP_PREFIX_RE.match(content)
    if not match:
        return None
    sender = str(match.group(1) or "").strip()
    message = str(match.group(2) or "").strip()
    if not sender or not message:
        return None
    return sender, message


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


def _normalize_whatsapp_identifier(value: str) -> str:
    normalized = (
        str(value or "")
        .strip()
        .replace("+", "", 1)
        .split(":", 1)[0]
        .split("@", 1)[0]
    )
    # Conversation IDs flow into lid-mapping file lookups. Reject path-like
    # values so alias expansion cannot traverse outside the session dir.
    if not normalized or "/" in normalized or "\\" in normalized:
        return ""
    if normalized in {".", ".."}:
        return ""
    return normalized


def format_merged_history(messages: list[dict[str, Any]]) -> str:
    """Format merged messages as grouped markdown for the soul's cross-chat context."""
    numeric_like_re = re.compile(r"^[0-9+\-() .]+$")

    def _conversation_kind_and_key(conversation_id: str) -> tuple[str, str]:
        cid = str(conversation_id or "").strip()
        if cid.startswith("whatsapp:group:"):
            return ("whatsapp_group", cid[len("whatsapp:group:"):].strip())
        if cid.startswith("whatsapp:dm:"):
            return ("whatsapp_dm", cid[len("whatsapp:dm:"):].strip())
        if cid.startswith("sillytavern:"):
            return ("sillytavern_dm", cid[len("sillytavern:"):].strip() or "sillytavern")
        if cid.startswith("integrity:"):
            return ("sillytavern_dm", cid)
        if cid.startswith("chat:"):
            return ("sillytavern_dm", cid)
        if cid == "sillytavern":
            return ("sillytavern_dm", "sillytavern")
        return ("sillytavern_dm", cid or "sillytavern")

    def _load_whatsapp_directory_names() -> dict[str, str]:
        hermes_home = Path(os.getenv("HERMES_HOME") or "~/.hermes").expanduser().resolve()
        directory_path = hermes_home / "channel_directory.json"
        out: dict[str, str] = {}
        if not directory_path.exists():
            return out
        try:
            raw = json.loads(directory_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return out
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
            normalized = _normalize_whatsapp_identifier(rid)
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
                    normalized = _normalize_whatsapp_identifier(chat_id)
                    if normalized:
                        out[normalized] = group_name
        return out

    def _lookup_whatsapp_name(key: str, names: dict[str, str]) -> str:
        key_norm = _normalize_whatsapp_identifier(key)
        candidates: list[str] = []
        seen: set[str] = set()

        def _push(candidate_key: str) -> None:
            value = str(names.get(candidate_key) or "").strip()
            if value and value not in seen:
                seen.add(value)
                candidates.append(value)

        if key:
            _push(key)
        if key_norm:
            _push(key_norm)
            _push(f"{key_norm}@s.whatsapp.net")
            _push(f"{key_norm}@lid")

        if not candidates:
            return ""

        def _score(name: str) -> tuple[int, int, int]:
            normalized_name = _normalize_whatsapp_identifier(name)
            same_as_key = int(bool(key_norm) and normalized_name == key_norm)
            numeric_like = int(bool(numeric_like_re.fullmatch(name)))
            return (same_as_key, numeric_like, len(name))

        return min(candidates, key=_score)

    def _conversation_heading(
        kind: str,
        key: str,
        names: dict[str, str],
        chat_name: str | None,
    ) -> str:
        if kind == "whatsapp_group":
            pretty = _lookup_whatsapp_name(key, names) or str(chat_name or "").strip() or key or "group"
            return f"[group][{pretty}]"
        if kind == "whatsapp_dm":
            pretty = _lookup_whatsapp_name(key, names) or str(chat_name or "").strip() or key or "contact"
            return f"[dm][{pretty}]"
        if kind == "sillytavern_dm":
            pretty = (chat_name or "").strip() or key or "sillytavern"
            return f"[dm][{pretty}]"
        return f"[dm][{key or 'sillytavern'}]"

    def _section_title(kind: str) -> str:
        if kind.startswith("sillytavern_"):
            return "My SillyTavern Conversations:"
        if kind.startswith("whatsapp_"):
            return "My WhatsApp Conversations:"
        return "My SillyTavern Conversations:"

    by_conversation: dict[str, list[dict[str, Any]]] = {}
    for msg in messages:
        cid = str(msg.get("conversation_id") or "").strip() or "unknown"
        by_conversation.setdefault(cid, []).append(msg)

    dir_names = _load_whatsapp_directory_names()

    sections: dict[str, list[tuple[str, str]]] = {}
    for cid, rows in by_conversation.items():
        kind, key = _conversation_kind_and_key(cid)
        section_key = _section_title(kind)
        entries = sections.setdefault(section_key, [])
        chat_name = ""
        for msg in reversed(rows):
            candidate = str(msg.get("chat_name") or "").strip()
            if candidate:
                chat_name = candidate
                break
        conv_lines: list[str] = [
            _conversation_heading(kind, key, dir_names, chat_name or None)
        ]
        last_time_label: str | None = None
        newest_ts = ""
        for msg in rows:
            ts = str(msg.get("received_at") or "")
            if ts > newest_ts:
                newest_ts = ts
            time_label = format_relative_time_label(msg.get("received_at"))
            if time_label and time_label != last_time_label:
                conv_lines.append(f"--- {time_label} ---")
                last_time_label = time_label
            role = str(msg.get("role") or "").strip()
            speaker = str(msg.get("speaker") or "").strip()
            content = str(msg.get("content") or "")
            if role == "user" and kind == "whatsapp_group":
                parsed = _parse_shared_group_sender_prefix(content)
                if parsed is not None:
                    speaker, content = parsed
            if not speaker:
                speaker = "soul" if role == "assistant" else (role or "unknown")
            conv_lines.append(f"[{speaker}]: {content}")
        entries.append((newest_ts, "\n".join(conv_lines)))

    lines: list[str] = []
    for section_title, entries in sections.items():
        if not entries:
            continue
        entries.sort(key=lambda e: e[0])
        blocks = [block for _, block in entries]
        if lines:
            lines.append("")
        lines.append(section_title)
        lines.append("")
        lines.append("\n\n".join(blocks))
    return "\n".join(lines).strip()
