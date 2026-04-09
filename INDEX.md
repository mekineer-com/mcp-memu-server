# mcp-memu-server — FastAPI Server Index

> This file exists so agents can orient themselves without scanning the tree.
> `.claudeignore` blocks auto-scan of this directory — read this first.

## Layout

```
mcp-memu-server/
├── app/main.py              # Endpoints + remaining business logic (~3,000 lines)
├── app/db.py                # SQLite helpers, schema ensures, JSON marshalling
├── app/database.py          # SQLAlchemy async engine + session factory
├── app/models/base.py       # Declarative ORM base
├── app/services/diary.py    # Diary generation, self-model update, intention creation
├── app/services/state.py    # Conversation state read/write/search
├── app/api/v1/              # Empty — routes not yet split from main.py
├── run.py                   # Entry point: config load, sys.path setup, single-instance pid guard, uvicorn start
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
| `/ping` | GET | Plugin ping (returns ok + serverInstanceId) |
| `/memorize` | POST | Extract memories from conversation text (also used by the optional sleep timer auto-digest worker); `force=true` batching via `_build_force_memorize_batches()` — prefers segment manifest ranges, fills gaps with `full.json`, falls back to token-window chunking |
| `/retrieve` | POST | Query memories (`rag` method; `llm` is internal-only for background flows) |
| `/conversation/{id}/retrieve` | POST | Retrieve + build turn prompt for chat path |
| `/conversation/{id}/turn` | POST | Soul turn loop: run LLM with turn contract, persist intentions + cache; system identity uses ST `soul_card` when provided, otherwise self-model-derived card (`narrative_self`); queues forced memorize when unmemorized chat tail exceeds `memorize.turn_history_token_budget` |
| `/conversation/{id}/turn/undo` | POST | Undo latest turn maintenance using `undo_snapshot` (single-step depth) |
| `/conversation/{id}/state` | GET/PATCH | Conversation working state |
| `/diary/generate` | POST | Generate diary entry from recent memories |
| `/intentions` | GET | List intentions from `intentions_life_goals` table (long-term, diary-managed) |
| `/intentions/{id}` | PATCH | Update intention status/priority |
| `/categories` | GET | List all categories |
| `/categories/search` | POST | Search categories |
| `/clear` | POST | Delete memories in scope |
| `/config` | GET/POST | Read or update runtime config |
| `/reload` | POST | Reload config from disk |
| `/diag/*` | GET | Diagnostic pages (recent memories, SQLite browser) |

## Extracted Modules

| Module | Purpose |
|--------|---------|
| `app/db.py` | `sqlite_ensure_*()`, `sqlite_connect()`, `json_to_db()`, `json_from_db()`, table column introspection |
| `app/services/diary.py` | Three-phase diary pipeline: `gather_diary_inputs()` builds anchor excerpt from queued episodes (with episode/time headers) and context (`memory_cache`, `intentions_active`, life goals); `run_diary_llm()` performs one forced-`method="llm"` `retrieve()` call per episode anchor to gather related background memories (IDs passed to self-model prompt), then writes diary/self-model prompts; `write_diary_outputs()` (lock-held) persists diary, self-model, intentions, summaries, life-goal updates, supersession writes for any `<supersedes>` IDs in soul observations (scope-validated; FTS-safe via `memory_item_repo.update_item()`), and `shaped_by` ID storage (parsed from `<shaped_by>` tags; stored as `extra.shaped_by_ids` on the written soul observation memory — provenance audit trail). `build_all_categories_summary()` — shared capped summary builder (100 tokens/category); used by diary state write and fed into memorize extraction for context enrichment. |
| `app/services/state.py` | `write_conversation_state()`, `conversation_state_from_row()`, cross-DB state search, `pending_diary_episode_ids` queue management |
| `app/services/turn_contract.py` | `make_turn_system_prompt()`, `build_turn_prompt()`, `parse_turn_contract()` — soul turn prompt construction and JSON contract parsing; temporal awareness: system prompt includes `Today is [date].` anchor; retrieved memory lines include relative-time labels from `happened_at` and `reinforced Nx` suffix from `reinforcement_count` |
| `app/services/intention_state.py` | `normalize_intentions_stack()`, `format_intentions_for_prompt()`, `upsert_intentions_stack_entries()` — intentions normalization and prompt formatting |

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
| Modify soul turn loop | `app/main.py` → `conversation_turn()` + `app/services/turn_contract.py` |
| Modify diary/self-model | `app/services/diary.py` |
| Modify conversation state | `app/services/state.py` |
| Modify turn intentions (working stack) | `app/services/intention_state.py` — reads/writes `intentions_active` JSON in conversation state |
| Modify life goals (long-term) | `app/services/diary.py` — `intentions_life_goals` DB table, managed exclusively by diary |
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
retrieve:   method (rag), apimw_enabled (bool; toggles turn-level APImw background retrieve)
memorize:   min_chunk_tokens, turn_history_token_budget, auto_memorize_on_sleep, sleep_timer_interval_seconds, supersede_similarity_threshold (default 0.75), enable_item_reinforcement (default true — enables reinforcement count roll-up on semantic dedupe merge)
debug:      log_prompts (bool) — dumps exact LLM prompt + response for memorize and diary steps to console
```

## Config-Only Runtime

- Runtime configuration is read from `config.json`.
- Database DSN source is `storage.metadata_store.dsn` in `config.json`.
- `run.py` and app runtime no longer use `DATABASE_URL` / `DATABASE_*` env branches.
