# Running the test suite

```sh
cd /home/marcos/apps-codex/mcp-memu-server
PYTHONPATH=.:/home/marcos/apps-codex/memu/src ./.venv/bin/python -m pytest tests/ -q
```

Notes:
- Use the local `./.venv/bin/python` (has fastapi). Neither system python3 nor `/home/marcos/apps-codex/.venv` have fastapi installed.
- `PYTHONPATH=.` loads the server's `app/` package; `memu/src` loads the `memu` engine that `app/services/consolidation.py` imports.
- Current state: 47 pass, 1 skipped.
