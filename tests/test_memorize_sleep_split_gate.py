from datetime import UTC, datetime

from app.services.memorize_endpoint import (
    select_sleep_splits_after_min_tokens,
    unmemorized_sleep_gap_detected,
)


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


def test_unmemorized_sleep_gap_detected_requires_floor_before_split() -> None:
    def _ts(y: int, m: int, d: int, hh: int, mm: int = 0) -> int:
        return int(datetime(y, m, d, hh, mm, tzinfo=UTC).timestamp() * 1000)

    history = [
        {"ts_ms": _ts(2026, 1, 1, 1, 0), "content": "small"},
        {"ts_ms": _ts(2026, 1, 1, 7, 0), "content": "small"},  # qualifying sleep split at idx=1
        {"ts_ms": _ts(2026, 1, 1, 7, 1), "content": "w " * 4000},  # floor exists only after the split
    ]
    assert (
        unmemorized_sleep_gap_detected(
            history,
            digest_cursor=-1,
            safe={"time_zone_offset_min": 0},
            logger=None,
            min_chunk_tokens=4000,
            sleep_split_min_lull_seconds=3 * 60 * 60,
        )
        is False
    )
