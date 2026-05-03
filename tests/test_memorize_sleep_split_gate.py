from app.services.memorize_endpoint import select_sleep_splits_after_min_tokens


def _msg(words: int) -> dict[str, str]:
    return {"content": "w " * words}


def test_select_sleep_splits_after_min_tokens_waits_until_floor_met() -> None:
    messages = [_msg(1000) for _ in range(5)]
    # 1000 words ≈ 1333 tokens with estimate_tokens formula (words / 0.75).
    # From start=0, split at 3 is the first boundary that reaches >= 4000 tokens.
    out = select_sleep_splits_after_min_tokens(
        messages,
        start_index=0,
        candidate_splits=[1, 2, 3, 4],
        min_chunk_tokens=4000,
    )
    assert out == [3]


def test_select_sleep_splits_after_min_tokens_no_floor_keeps_all_after_start() -> None:
    messages = [_msg(50) for _ in range(4)]
    out = select_sleep_splits_after_min_tokens(
        messages,
        start_index=1,
        candidate_splits=[0, 1, 2, 3],
        min_chunk_tokens=0,
    )
    assert out == [2, 3]
