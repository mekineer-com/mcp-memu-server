# mcp-memu-server — FastAPI Server Index

> This file exists so agents can orient themselves without scanning the tree.
> `.claudeignore` blocks auto-scan of this directory — read this first.

## Layout

```
mcp-memu-server/
├── app/main.py              # Endpoints + remaining business logic (~2,900 lines)
├── app/db.py                # SQLite helpers, schema ensures, JSON marshalling
├── app/database.py          # SQLAlchemy async engine + session factory
├── app/models/base.py       # Declarative ORM base
├── app/services/diary.py    # Diary generation, self-model update, intention creation
├── app/services/state.py    # Conversation state read/write/search
├── app/api/v1/              # Empty — routes not yet split from main.py
├── run.py                   # Entry point: config load, sys.path setup, uvicorn start
├── config.json              # Runtime config (llm, storage, listen, categories, memu path)
├── config.example.json      # Template
├── tests/test_main.py       # Minimal smoke test
├── alembic/                 # DB migration scripts
├── storage/                 # Default SQLite DB + resource dir
├── Makefile                 # make install/run/test/check
└── pyproject.toml           # Python 3.12+ deps
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

## Extracted Modules

| Module | Purpose |
|--------|---------|
| `app/db.py` | `sqlite_ensure_*()`, `sqlite_connect()`, `json_to_db()`, `json_from_db()`, table column introspection |
| `app/services/diary.py` | `generate_diary()`, diary/self-model XML parsing, self-model load/format, intention creation. All diary+self-model+intention writes in one SQLite transaction. |
| `app/services/state.py` | `write_conversation_state()`, `conversation_state_from_row()`, cross-DB state search, `pending_diary_memory_ids` queue management |

## How It Connects to memu

```python
from memu.app import MemoryService    # main facade
from memu.prompts.diary import ...     # diary prompts
from memu.prompts.memory_type import ...  # type prompts
```

- `config.json` → `memu.path` points to memu/src
- `run.py` inserts that path into `sys.path[0]`
- `MemoryService` instances cached per unique (llm_profiles + db_config) pair in `_SERVICES` dict

## Task → Where to Look

| Task | File |
|------|------|
| Add API endpoint | `app/main.py` — add `@app.post/get` handler, use `_get_service_from_payload()` |
| Modify memorize flow | `app/main.py` → `_run_memorize()` (calls `svc.memorize()`) |
| Modify retrieval flow | `app/main.py` → `_run_retrieve()` (calls `svc.retrieve()`) |
| Modify diary/self-model | `app/services/diary.py` |
| Modify conversation state | `app/services/state.py` |
| Modify DB schema/helpers | `app/db.py` |
| Change config shape | `config.json` + `app/main.py` → `_load_config()` |
| Add route group | Create `app/api/v1/{group}.py` with APIRouter, include in main.py |

## Config (`config.json`)

```
llm:        provider, api_key, base_url, chat_model, embed_model
storage:    resources_dir, metadata_store (provider + dsn)
listen:     host, port
memu:       path (to memu/src)
categories: defaults[], allow_dynamic, thresholds
retrieve:   method (rag|llm)
```

## Config-Only Runtime

- Runtime configuration is read from `config.json`.
- Database DSN source is `storage.metadata_store.dsn` in `config.json`.
- `run.py` and app runtime no longer use `DATABASE_URL` / `DATABASE_*` env branches.
