from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _prefer_nonempty_text(incoming: Any, existing: Any) -> str:
    incoming_text = str(incoming or "").strip()
    if incoming_text:
        return incoming_text
    return str(existing or "").strip()


def _apply_tension_updates(
    existing_tensions: list[dict[str, Any]],
    *,
    tension_remove: list[str],
    tension_add: list[dict[str, Any]],
    normalize_trait_strength: Callable[[Any], float],
) -> list[dict[str, Any]]:
    remove_set = {str(text or "").strip() for text in tension_remove if str(text or "").strip()}
    updated = [item for item in existing_tensions if str(item.get("between") or "").strip() not in remove_set]

    for tension in tension_add:
        between = str(tension.get("between") or "").strip()
        if not between:
            continue
        for existing_tension in updated:
            if str(existing_tension.get("between") or "").strip() != between:
                continue
            existing_tension["root"] = _prefer_nonempty_text(
                tension.get("root"),
                existing_tension.get("root"),
            )
            existing_tension["implication"] = _prefer_nonempty_text(
                tension.get("implication"),
                existing_tension.get("implication"),
            )
            existing_tension["strength"] = normalize_trait_strength(tension.get("strength"))
            break
        else:
            updated.append(
                {
                    "type": "tension",
                    "between": between,
                    "root": str(tension.get("root") or "").strip(),
                    "implication": str(tension.get("implication") or "").strip(),
                    "strength": normalize_trait_strength(tension.get("strength")),
                }
            )

    return updated
