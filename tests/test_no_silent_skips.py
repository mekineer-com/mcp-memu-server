"""Guard against the silent-skip regression that hid a broken test env since baseline.

History: every test in test_main.py wrapped `from app import main` in
`try/except Exception as e: pytest.skip(...)`. The import failed (memu not
pip-installed in the server venv; runtime uses run.py's sys.path injection)
so the suite reported "1 passed, 20 skipped" on every run. Three code-drift
bugs landed uncaught before this was spotted.

This test fails loudly if anyone reintroduces the pattern.
"""

from __future__ import annotations

import re
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent

# Patterns that indicate a silent-skip wrapper around an import.
_BAD_PATTERNS = [
    # try/except Exception: pytest.skip(...) — the exact shape that hid the bug
    re.compile(r"except\s+Exception\s+as\s+\w+:\s*\n\s*pytest\.skip", re.MULTILINE),
    # Bare except: pytest.skip(...)
    re.compile(r"except\s*:\s*\n\s*pytest\.skip", re.MULTILINE),
    # pytest.skip with "compatibility issue" or similar import-hiding text
    re.compile(r'pytest\.skip\([^)]*compatibility\s+issue', re.IGNORECASE),
    re.compile(r'pytest\.skip\([^)]*Import\s+test\s+skipped', re.IGNORECASE),
]


def test_no_silent_import_skips_in_test_suite() -> None:
    offenders: list[tuple[Path, str]] = []
    for test_file in _TESTS_DIR.rglob("test_*.py"):
        if test_file.name == "test_no_silent_skips.py":
            continue
        source = test_file.read_text()
        for pattern in _BAD_PATTERNS:
            match = pattern.search(source)
            if match is not None:
                offenders.append((test_file.relative_to(_TESTS_DIR), match.group(0).strip()))

    assert not offenders, (
        "Silent-skip patterns detected. These hide import failures and "
        "cause tests to report green while doing nothing.\n"
        "Offenders:\n" + "\n".join(f"  {p}: {snippet}" for p, snippet in offenders)
    )
