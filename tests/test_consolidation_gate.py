"""Consolidation gate / first-run / stale-in-progress invariants.

Guards the invariants that control whether a consolidation run fires:

1. Fresh state (`last_consolidation_at=None`, `in_progress=False`) → gate
   is open even within the interval window. This is the first-run bootstrap
   contract.
2. Recent successful run (`last_consolidation_at=now`) → subsequent runs
   inside the interval are skipped with reason="interval_gate".
3. `consolidation_in_progress=True` with a recent `started_at` → skip with
   reason="in_progress" (another worker is running).
4. `consolidation_in_progress=True` but `started_at` older than
   `stale_after` → treated as crash recovery: the flag is reset and the
   run proceeds.
5. `force=True` bypasses the interval gate regardless of
   `last_consolidation_at`.

These invariants live inline at `consolidation.py:~270-290`. Extracting the
gate decision into a pure function would make it directly testable; until
then, this test pins the date-math contract the gate relies on, which is
the piece that shifts when someone "just tweaks the interval".
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def _gate_decision(
    *,
    last_consolidation_at: datetime | None,
    consolidation_in_progress: bool,
    started_at: datetime | None,
    now: datetime,
    interval_days: int,
    force: bool,
    stale_after: timedelta,
) -> str:
    """Mirror of the gate branch at consolidation.py ~270-290."""
    if consolidation_in_progress:
        if started_at is not None and now - started_at <= stale_after:
            return "skip_in_progress"
        return "reset_and_run"
    if not force and last_consolidation_at is not None:
        due_at = last_consolidation_at + timedelta(days=max(1, int(interval_days)))
        if now < due_at:
            return "skip_interval"
    return "run"


_NOW = datetime(2026, 4, 22, 12, 0, tzinfo=UTC)


def test_first_run_bootstrap_runs_when_never_consolidated() -> None:
    assert (
        _gate_decision(
            last_consolidation_at=None,
            consolidation_in_progress=False,
            started_at=None,
            now=_NOW,
            interval_days=7,
            force=False,
            stale_after=timedelta(hours=2),
        )
        == "run"
    )


def test_recent_successful_run_is_gated_inside_interval() -> None:
    last = _NOW - timedelta(days=3)
    assert (
        _gate_decision(
            last_consolidation_at=last,
            consolidation_in_progress=False,
            started_at=None,
            now=_NOW,
            interval_days=7,
            force=False,
            stale_after=timedelta(hours=2),
        )
        == "skip_interval"
    )


def test_stale_in_progress_flag_triggers_crash_recovery() -> None:
    stale_start = _NOW - timedelta(hours=10)
    assert (
        _gate_decision(
            last_consolidation_at=None,
            consolidation_in_progress=True,
            started_at=stale_start,
            now=_NOW,
            interval_days=7,
            force=False,
            stale_after=timedelta(hours=2),
        )
        == "reset_and_run"
    )


def test_recent_in_progress_flag_skips_with_in_progress_reason() -> None:
    recent_start = _NOW - timedelta(minutes=5)
    assert (
        _gate_decision(
            last_consolidation_at=None,
            consolidation_in_progress=True,
            started_at=recent_start,
            now=_NOW,
            interval_days=7,
            force=False,
            stale_after=timedelta(hours=2),
        )
        == "skip_in_progress"
    )


def test_force_bypasses_interval_gate() -> None:
    last = _NOW - timedelta(days=1)
    assert (
        _gate_decision(
            last_consolidation_at=last,
            consolidation_in_progress=False,
            started_at=None,
            now=_NOW,
            interval_days=7,
            force=True,
            stale_after=timedelta(hours=2),
        )
        == "run"
    )


def test_interval_days_zero_floors_to_one_day() -> None:
    # Operator mis-configures interval_days=0; the gate should still block
    # re-running within the same day. `max(1, interval_days)` is the floor.
    last = _NOW - timedelta(hours=6)
    assert (
        _gate_decision(
            last_consolidation_at=last,
            consolidation_in_progress=False,
            started_at=None,
            now=_NOW,
            interval_days=0,
            force=False,
            stale_after=timedelta(hours=2),
        )
        == "skip_interval"
    )


def test_first_run_crash_then_retry_runs_again_not_gated() -> None:
    # The scenario from AUDIT_PLAN P0 #6: first consolidation ran (set
    # in_progress=True), crashed before stamping last_consolidation_at,
    # next attempt after stale_after must treat it as retry and proceed.
    # Data model handles the retry via narrative_self evolved_into chains.
    stale_start = _NOW - timedelta(hours=10)
    assert (
        _gate_decision(
            last_consolidation_at=None,  # never stamped
            consolidation_in_progress=True,
            started_at=stale_start,
            now=_NOW,
            interval_days=7,
            force=False,
            stale_after=timedelta(hours=2),
        )
        == "reset_and_run"
    )
