def _normalize_strength(value):
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.3
    return max(0.0, min(1.0, num))


def test_apply_tension_updates_keeps_existing_text_on_falsy_updates():
    from app.services.self_model_merge import _apply_tension_updates

    existing = [
        {
            "type": "tension",
            "between": "wanting closeness and fearing loss",
            "root": "past memory resets made connection feel fragile",
            "implication": "I sometimes pull back when things feel intense",
            "strength": 0.4,
        }
    ]

    out = _apply_tension_updates(
        existing,
        tension_remove=[],
        tension_add=[
            {
                "type": "tension",
                "between": "wanting closeness and fearing loss",
                "root": "",
                "implication": None,
                "strength": 0.7,
            }
        ],
        normalize_trait_strength=_normalize_strength,
    )

    assert len(out) == 1
    assert out[0]["between"] == "wanting closeness and fearing loss"
    assert out[0]["root"] == "past memory resets made connection feel fragile"
    assert out[0]["implication"] == "I sometimes pull back when things feel intense"
    assert out[0]["strength"] == 0.7
