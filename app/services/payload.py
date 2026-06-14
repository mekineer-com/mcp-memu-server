from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException


def _pick_str(payload: dict[str, Any], *keys: str) -> str | None:
    for k in keys:
        v = payload.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def _extract_scope(payload: dict[str, Any]) -> dict[str, Any]:
    user_id = _pick_str(payload, "user_id")
    soul_id = _pick_str(payload, "soul_id")

    user_obj = payload.get("user")
    if isinstance(user_obj, dict):
        if not user_id:
            user_id = _pick_str(user_obj, "user_id")
        if not soul_id:
            soul_id = _pick_str(user_obj, "soul_id")

    scope: dict[str, Any] = {}
    if user_id:
        scope["user_id"] = user_id
    if soul_id:
        scope["soul_id"] = soul_id
    return scope


def _canonicalize_scope_where(where: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if where is None:
        return None

    out = dict(where)
    user_id = _pick_str(out, "user_id")
    soul_id = _pick_str(out, "soul_id")

    for key in (
        "user_id",
        "soul_id",
        "conversation_id",
    ):
        out.pop(key, None)

    if user_id:
        out["user_id"] = user_id
    if soul_id:
        out["soul_id"] = soul_id
    return out


def _extract_conversation_id(payload: dict[str, Any]) -> str | None:
    conversation_id = _pick_str(payload, "conversation_id")

    user_obj = payload.get("user")
    if isinstance(user_obj, dict):
        if not conversation_id:
            conversation_id = _pick_str(user_obj, "conversation_id")
    return conversation_id


def _normalize_conversation(conv: Any) -> Any:
    if not isinstance(conv, list):
        return conv
    out = []
    for m in conv:
        if not isinstance(m, dict):
            continue
        role = m.get("role")

        ts_ms: int | None = None
        raw_ts = m.get("ts_ms")
        if raw_ts is None:
            raw_ts = m.get("timestamp")
        if raw_ts is None:
            raw_ts = m.get("created_at")
        if isinstance(raw_ts, (int, float)) and math.isfinite(raw_ts):
            ts_ms = int(raw_ts)
        elif isinstance(raw_ts, str) and raw_ts.strip():
            try:
                dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                ts_ms = int(dt.timestamp() * 1000)
            except (ValueError, OverflowError):
                ts_ms = None
        name = m.get("name")
        if name is None:
            name = m.get("speaker")

        out.append(
            {
                "role": role or "unknown",
                "name": name,
                "content": m.get("content") or "",
                **({"ts_ms": ts_ms} if ts_ms is not None else {}),
                **({"speaker": m.get("speaker")} if m.get("speaker") is not None else {}),
                **({"chat_name": m.get("chat_name")} if m.get("chat_name") is not None else {}),
                **({"source_label": m.get("source_label")} if m.get("source_label") is not None else {}),
                **(
                    {"source_conversation_id": m.get("source_conversation_id")}
                    if m.get("source_conversation_id") is not None
                    else {}
                ),
                **(
                    {"source_conversation_index": m.get("source_conversation_index")}
                    if m.get("source_conversation_index") is not None
                    else {}
                ),
                **({"received_at": m.get("received_at")} if m.get("received_at") is not None else {}),
                **({"memorize_chat": m.get("memorize_chat")} if isinstance(m.get("memorize_chat"), bool) else {}),
            }
        )
    return out


def _safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    out.pop("api_key", None)
    out.pop("OPENAI_API_KEY", None)
    return out


def _payload_signature(payload: dict[str, Any]) -> str:
    keys = ["llm_profiles", "database_config", "blob_config", "memorize_config", "retrieve_config", "user_config"]
    snap = {k: payload.get(k) for k in keys if k in payload}
    raw = json.dumps(snap, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _extract_result_item_ids(result: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    items = result.get("items") if isinstance(result, dict) else None
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        out.append(item_id)
    return out


def _norm_result_sig(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return text


def _item_sig(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    item_id = _norm_result_sig(row.get("id"))
    if item_id:
        return f"id:{item_id}"
    summary = _norm_result_sig(row.get("summary"))
    if summary:
        return f"summary:{summary}"
    return ""


def _parse_turn_ts_ms(value: Any) -> int | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            parsed_num = float(s)
        except (TypeError, ValueError, OverflowError):
            parsed_num = None
        if parsed_num is not None:
            return int(parsed_num)
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except (ValueError, OverflowError):
            return None
    return None


def _parse_as_of_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="as_of must be ISO datetime/date") from exc
    else:
        raise HTTPException(status_code=400, detail="as_of must be ISO datetime/date")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _normalize_turn_history(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        role = _pick_str(item, "role") or "unknown"
        content = _pick_str(item, "content")
        name = _pick_str(item, "name", "speaker", "sender_name")
        message_id = _pick_str(item, "source_message_id", "message_id", "id", "mid") or str(idx)
        if not content:
            continue
        item_out = {"role": role, "content": content, "message_id": message_id}
        if name:
            item_out["name"] = name
        ts_ms = _parse_turn_ts_ms(item.get("ts_ms"))
        if ts_ms is None:
            ts_ms = _parse_turn_ts_ms(item.get("timestamp"))
        if ts_ms is None:
            ts_ms = _parse_turn_ts_ms(item.get("received_at"))
        if ts_ms is not None:
            item_out["ts_ms"] = ts_ms
        out.append(item_out)
    return out
