# mcp-memu-server — FastAPI Server Index

> This file exists so agents can orient themselves without scanning the tree.
> `.claudeignore` blocks auto-scan of this directory — read this first.

## Layout

```
mcp-memu-server/
├── app/main.py          # ALL endpoints & business logic (~3,500 lines, monolithic)
├── app/database.py      # SQLAlchemy async engine + session factory
├── app/models/base.py   # Declarative ORM base
├── app/api/v1/          # Empty — routes not yet split from main.py
├── app/services/        # Empty — future extraction point
├── run.py               # Entry point: config load, sys.path setup, uvicorn start
├── config.json          # Runtime config (llm, storage, listen, categories, memu path)
├── config.example.json  # Template
├── tests/test_main.py   # Minimal smoke test
├── alembic/             # DB migration scripts
├── storage/             # Default SQLite DB + resource dir
├── Makefile             # make install/run/test/check
└── pyproject.toml       # Python 3.12+ deps
```

## Key Endpoints (all in `app/main.py`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/memorize` | POST | Extract memories from conversation text |
| `/retrieve` | POST | Query memories (rag or llm method) |
| `/diary/generate` | POST | Generate diary entry from recent memories |
| `/categories` | GET | List all categories |
| `/categories/search` | POST | Search categories |
| `/conversation/{id}/state` | GET | Conversation working state |
| `/clear` | POST | Delete memories in scope |
| `/config` | GET/POST | Read or update runtime config |
| `/reload` | POST | Reload config from disk |
| `/diag/*` | GET | Diagnostic pages (recent memories, SQLite browser) |

## How It Connects to memu-1.4.0

```python
from memu.app import MemoryService    # main facade
from memu.prompts.diary import ...     # diary prompts
from memu.prompts.memory_type import ...  # type prompts
```

- `config.json` → `memu.path` points to memu-1.4.0/src
- `run.py` inserts that path into `sys.path[0]`
- `MemoryService` instances cached per unique (llm_profiles + db_config) pair in `_SERVICES` dict

## Task → Where to Look

| Task | File |
|------|------|
| Add API endpoint | `app/main.py` — add `@app.post/get` handler, use `_get_service_from_payload()` |
| Modify memorize flow | `app/main.py` → `_run_memorize()` (calls `svc.memorize()`) |
| Modify retrieval flow | `app/main.py` → `_run_retrieve()` (calls `svc.retrieve()`) |
| Change config shape | `config.json` + `app/main.py` → `_load_config()` |
| Add route group | Create `app/api/v1/{group}.py` with APIRouter, include in main.py |

## Config (`config.json`)

```
llm:        provider, api_key, base_url, chat_model, embed_model
storage:    resources_dir, metadata_store (provider + dsn)
listen:     host, port
memu:       path (to memu-1.4.0/src)
categories: defaults[], allow_dynamic, thresholds
retrieve:   method (rag|llm)
```

## Key Env Var Overrides

| Variable | Purpose |
|----------|---------|
| `MEMU_SERVER_CONFIG` | Path to config.json |
| `MEMU_SERVER_HOST/PORT` | Listen address |
| `OPENAI_API_KEY` | LLM API key |
| `STORAGE_PATH` | SQLite base dir |
| `DATABASE_URL` | PostgreSQL DSN (optional) |
