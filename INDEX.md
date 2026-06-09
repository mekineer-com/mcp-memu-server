# mcp-memu-server — FastAPI Server Index

> This file exists so agents can orient themselves without scanning the tree.
> `.claudeignore` blocks auto-scan of this directory — read this first.

## Layout

```
mcp-memu-server/
├── app/main.py              # Core orchestration + remaining endpoints (~3,000 lines)
├── app/config.py            # Runtime config load/save/mask + path + sqlite DSN helpers
├── app/db.py                # SQLite helpers, schema ensures, JSON marshalling
├── app/database.py          # SQLAlchemy async engine + session factory
├── app/models/base.py       # Declarative ORM base
├── app/services/consolidation.py # Weekly consolidation pipeline
├── app/services/diary.py    # Diary helper primitives used by consolidation
├── app/services/memorize_endpoint.py # /memorize orchestration + sleep-gap/token batching helpers
├── app/services/sqlite_scope.py # SQLite scoped-path/status helpers shared by endpoints
├── app/services/crud_endpoints.py # Categories/intentions/relationships/state/clear endpoint logic
├── app/services/state.py    # Conversation state read/write/search
├── app/api/v1/              # Empty — routes not yet split from main.py
├── run.py                   # Entry point: config load, sys.path setup, single-instance pid guard, uvicorn start
├── config.json              # Runtime config (llm, storage, listen, categories, memu path)
├── config.example.json      # Template
├── tests/                   # pytest suite; see `TESTING.md` for run command
├── alembic/                 # DB migration scripts
├── storage/                 # Default SQLite DB + resource dir
├── Makefile                 # make install/run/test/check
└── pyproject.toml           # Python 3.12+ deps
```

## Key Endpoints (registered from `app/main.py`; admin/diag handlers in `app/services/admin_routes.py`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/version` | GET | Build / server instance identity |
| `/admin/shutdown` | POST | Request graceful shutdown (drain mode) |
| `/admin/shutdown/status` | GET | Shutdown progress + active request counts |
| `/memorize` | POST | Extract memories from conversation text. User-initiated only ("Memorize Now" / "Re-memorize chat" buttons, `force=true`). `force=true` batching via `_build_force_memorize_batches()` — prefers segment manifest ranges, falls back to token-window chunking. Runtime progress tracks segments: server persists each selected segment to `segments/*.json`, passes one synthetic segment episode payload into memu batch extraction, and advances conversation cursor per completed segment. Auto-memorize is triggered server-side inside `/conversation/{id}/turn` (see that row). |
| `/retrieve` | POST | Query memories. Optional `as_of` applies temporal triple filtering (`valid_from`/`valid_to`) for graph retrieval. |
| `/timeline` | GET | Entity relationship timeline (`entity`, `user_id`, `soul_id`, optional `as_of`) for chronological graph inspection |
| `/conversation/{id}/retrieve` | POST | Retrieve + build turn prompt for chat path; when caller provides only `query` (no `queries`), server enriches retrieve context with identity/time anchor, `all_categories_summary`, `memory_cache`, `intentions`, and current-chat payload history before calling memu retrieve. Cross-chat merge now reads WhatsApp tails from Hermes source files (`sessions.json` + `state.db`) instead of local `messages` rows. |
| `/conversation/{id}/turn` | POST | Soul turn loop: requires extension-provided `prompt_override_payload` (prepared by `/conversation/{id}/retrieve`), runs LLM with turn contract, persists intentions + cache, and never re-runs retrieve inside turn; system identity uses ST `soul_card` when provided, otherwise self-model-derived card (`narrative_self`); optional free-turn metadata starts immediate Claude Code continuations or stores durable `follow_up` wakes in scoped SQLite; APImw fires background `_run_apimw()` pipeline when cadence is met (configured by `retrieve.apimw_cadence`), skips when a prior APImw job is still in flight, runs step A topic statement + step B retrieve + step C second-pass rewrite retrieve when query changes + combined D+E+F selection/edges/intention+cache update, then writes `prior_context`, `retrieval_ids_since_consolidation`, `prior_context_ids_since_consolidation`, `memory_cache`, `intentions_active` and edge invalidations/additions; queues forced memorize only from primary-chat tails (background-chat tails excluded from segment trigger). Background (`memorize_chat=false`) chats trigger adapter-sourced rolling-summary updates on turn lulls. Cross-memorize feed reads `memorize_chat=true` tails from source adapters (WhatsApp Hermes + ST snapshots) and `memorize_chat=false` tails from Hermes row-id cursors via `rolling_summary_cursor_id`; no turn-time raw-message append path remains in server |
| `/conversation/{id}/turn/undo` | POST | Undo latest turn maintenance using `undo_snapshot` (single-step depth) |
| `/integration/memu/turn` | POST | MCP-facing single-call turn wrapper: internally runs conversation-retrieve (`build_turn_prompt=true`) then conversation-turn with prompt override payload |
| `/integration/memu/retrieve` | POST | MCP-facing retrieve wrapper |
| `/integration/memu/memorize` | POST | MCP-facing memorize trigger wrapper (`force` supported) |
| `/integration/memu/consolidate` | POST | MCP-facing force consolidation wrapper |
| `/integration/memu/intentions` | POST | MCP-facing intentions read wrapper |
| `/integration/memu/state` | POST | State get/patch wrapper (`action=get|patch`); currently excluded from MCP discovery surface (`include_operations`) |
| `/conversation/{id}/state` | GET/PATCH | Conversation working state |
| `/conversation/{id}/consolidation/force` | POST | Force consolidation now (bypasses interval gate, still lock-safe) |
| `/souls/{soul_id}/intentions` | GET | List intentions from `intentions_life_goals` table (long-term, consolidation-managed) |
| `/intentions/{id}` | PATCH | Update intention status/priority |
| `/souls/{soul_id}/relationships` | GET/POST | List or create user-declared relationship entities (`memu_entities` rows with `properties.origin=user_declared`) |
| `/souls/{soul_id}/relationships/{speaker_id}` | PATCH/DELETE | Update or soft-delete one relationship entity (`entity:*` only; reserved prefixes rejected) |
| `/souls/{soul_id}/narrative_suggestion` | POST | Snapshot previous `narrative_self` with `evolved_into` chain before overwrite; extension surfaces this via the Memorize Now menu |
| `/categories` | GET | List all categories |
| `/categories/search` | POST | Search categories |
| `/clear` | POST | Delete memories in scope |
| `/config` | GET/POST | Read or update runtime config |
| `/reload` | POST | Reload config from disk |
| `/diag`, `/diag/calls`, `/diag/http`, `/diag/sqlite`, `/diag/sqlite/counts`, `/diag/sqlite/recent` | GET | Diagnostic pages (recent memories, SQLite browser, last 50 memorize/retrieve calls, HTTP introspection). Read-only (no schema writes/migrations on diag reads; never use diag calls for DB bootstrap) |

## Extracted Modules

| Module | Purpose |
|--------|---------|
| `app/config.py` | Config/runtime helpers: `load_config()`, `save_config()`, `mask_config()`, storage path normalization, sqlite DSN scoping, default llm profile assembly, soul generation config I/O |
| `app/db.py` | `sqlite_ensure_*()`, `sqlite_connect()`, `json_to_db()`, `json_from_db()`, table column introspection |
| `app/services/consolidation.py` | Consolidation pipeline: gather queue/context, run one consolidation LLM call, write narrative_self + life-goal edits + companion memory + per-episode diary rows, and clear queue/flags. |
| `app/services/diary.py` | Diary helpers for consolidation: episode parsing/excerpts, diary XML parsing, and diary/companion memory write helpers. |
| `app/services/graph_edges.py` | Shared edge normalization + write/invalidate helpers used by APImw and consolidation (`caused_by`, `evokes`, `conflicts_with`, `parallels`, `shaped_by`). |
| `app/services/memorize_endpoint.py` | `/memorize` endpoint core, forced-memorize background runner (`run_memorize_episodes`), segment-file persistence (`chat_dir/segments/*.json`), rolling-summary row injection into segment payloads, progress/cancel handlers, and chat sleep-gap/token chunking helpers. |
| `app/services/conversation_sources.py` | Source adapters for cross-conversation reads: WhatsApp from Hermes (`sessions.json` + `state.db`) and SillyTavern from per-conversation snapshots under `resources/st_chats/*/latest_history.json`, with digest-cursor slicing, floor backfill, and row-id based WhatsApp reads for listen-only summarize boundaries |
| `app/services/admin_routes.py` | Health/version/shutdown/diag endpoint handlers (including `/diag` and MCP-prefixed diag aliases). |
| `app/services/payload.py` | Shared payload/scope/normalization helpers used across retrieve/turn/service-factory paths (scope extraction, turn history normalization, signature helpers, payload scrubbing). |
| `app/services/service_factory.py` | Service cache + payload-driven `MemoryService` construction, llm profile merge, and config readers (`apimw_*`, consolidation interval, retrieve config shaping). |
| `app/services/retrieve_orchestration.py` | Retrieve domain helpers + orchestration seam: query/where extraction, identity-context builder, retrieve rewrite constants/context-query assembly, and `_run_retrieve` implementation (called via thin wrapper in `main.py`). |
| `app/services/sqlite_scope.py` | SQLite scope plumbing used across endpoints: scoped db-path resolution, scope `WHERE` builder, state-db lookup/write wrappers, and lightweight file info/intention row helpers. |
| `app/services/crud_endpoints.py` | CRUD endpoint logic extracted from `main.py`: categories search/list, intentions, relationships, narrative suggestion, conversation state get/patch, clear-memory. |
| `app/services/mcp_tools.py` | MCP-facing wrapper contracts (`memu_*`): typed request models + thin orchestration over existing REST/turn endpoints |
| `app/services/state.py` | `write_conversation_state()`, `conversation_state_from_row()`, cross-DB state search, `pending_diary_episode_ids` queue management; includes background rollup state fields (`rolling_summary`, `rolling_summary_cursor_id`, `rolling_summary_updated_at`); error state fields: `last_background_error` + timestamp (written by APImw on failure, cleared on APImw success); `last_consolidation_error` + timestamp (written by consolidation pipeline, cleared on consolidation success) |
| `app/services/turn_contract.py` | `make_turn_system_prompt()`, `build_turn_prompt()`, `parse_turn_contract()` — soul turn prompt construction and JSON contract parsing; temporal awareness: system prompt includes `Today is [date].` anchor; `Retrieved memory context` section includes all-categories orientation first, then retrieved category/item hits; when retrieved items include speaker metadata it renders a compact `Speakers:` block and memory lines as `[memory_type][speaker_label]`; memory lines include relative-time labels from `happened_at` and `reinforced Nx` suffix from `reinforcement_count` |
| `app/services/intention_state.py` | `normalize_intentions_stack()`, `format_intentions_for_prompt()`, `upsert_intentions_stack_entries()` — intentions normalization and prompt formatting |

## How It Connects to memu

```python
from memu.app import MemoryService    # main facade
from memu.prompts.consolidation import consolidation  # consolidation prompt
from memu.prompts.memory_type import ...  # type prompts
```

- `config.json` → `memu.path` points to memu/src
- `run.py` inserts that path into `sys.path[0]`
- `MemoryService` instances cached per unique (llm_profiles + db_config) pair in `_SERVICES` dict

## Task → Where to Look

| Task | File |
|------|------|
| Add API endpoint | `app/main.py` — add `@app.post/get` handler (thin delegator); keep core logic in `app/services/*_endpoint.py` |
| Modify memorize flow | `app/main.py` → `_run_memorize()` (calls `svc.memorize()`) |
| Modify retrieval flow | `app/main.py` → `_run_retrieve()` (calls `svc.retrieve()`) |
| Modify soul turn loop | `app/main.py` → `conversation_turn()` + `app/services/turn_contract.py` |
| Modify consolidation / diary writes | `app/services/consolidation.py`, `app/services/diary.py` |
| Modify conversation state | `app/services/state.py` |
| Modify turn intentions (working stack) | `app/services/intention_state.py` — reads/writes `intentions_active` JSON in conversation state |
| Modify life goals (long-term) | `app/services/consolidation.py` — `intentions_life_goals` DB table updates during consolidation |
| Modify DB schema/helpers | `app/db.py` |
| Change config shape | `config.json` + `app/config.py` |
| Add route group | Create `app/api/v1/{group}.py` with APIRouter, include in main.py |

## Config (`config.json`)

```
llm:        provider, api_key, base_url, chat_model, embed_model
storage:    resources_dir, metadata_store (provider + dsn)
listen:     host, port
memu:       path (to memu/src)
categories: defaults[], dynamic-category thresholds
retrieve:   apimw_enabled (bool; toggles APImw pipeline), apimw_cadence (int, default 3; APImw runs every N soul messages), apimw_memory_count (int, default 20; APImw item.top_k), apimw_random_count (int, default 5; APImw random sample size)
memorize:   min_chunk_tokens (default 4000; floor for sleep-gap-triggered memorize), background_summary_tokens (default 1000; token threshold that triggers rolling-summary for memorize_chat=false chats), background_summary_min_tokens (default 100; minimum token threshold when rollup is queued from cross-memorize assembly), background_extra_messages_tokens (memu preprocessor floor for background message rendering)
consolidation_interval_days: cadence gate for consolidation after successful memorize runs (default 7)
debug:      log_prompts (bool) — dumps exact LLM prompt + response for memorize/consolidation steps to console
```

## Config-Only Runtime

- Runtime configuration is read from `config.json`.
- Database DSN source is `storage.metadata_store.dsn` in `config.json`.
- `run.py` and app runtime no longer use `DATABASE_URL` / `DATABASE_*` env branches.
