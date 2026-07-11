from __future__ import annotations


def canonical_conversation_id(conversation_id: str | None) -> str:
    cid = str(conversation_id or "").strip()
    prefix = "whatsapp:group:"
    marker = "@g.us"
    if cid.startswith(prefix) and marker in cid[len(prefix):]:
        rest = cid[len(prefix):]
        return f"{prefix}{rest.split(marker, 1)[0]}{marker}"
    return cid
