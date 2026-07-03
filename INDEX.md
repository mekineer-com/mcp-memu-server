# mcp-memu-server — FastAPI Server Index

> This file exists so agents can orient themselves without scanning the tree.
> `.claudeignore` blocks auto-scan of this directory — read this first.

## Layout

```
mcp-memu-server/
├── app/main.py              # Core orchestration + remaining endpoints, including graph `/graph`, `/pending`, `/memory/{id}`, `/category/{id}` wrappers
├── app/config.py            # Runtime config load/save/mask + path + sqlite DSN helpers
├── app/db.py                # SQLite helpers, schema ensures, JSON marshalling
├── app/database.py          # SQLAlchemy async engine + session factory
├── app/models/base.py       # Declarative ORM base
├── app/services/consolidation.py # Weekly consolidation pipeline
├── app/services/diary.py    # Diary helper primitives used by consolidation
├── app/services/memorize_endpoint.py # /memorize orchestration + sleep-gap/token batching helpers
├── app/services/activity_messages.py # Synthetic self-DM activity recap table helpers
├── app/services/whatsapp_outbounds.py # WhatsApp pending outbound queue helpers
├── app/services/free_turn.py # Free-turn continuation + scheduled follow-up helpers
├── app/services/cross_history.py # Cross-conversation history composition over source adapters
├── app/services/apimw.py # APImw background memory-weaving pipeline helpers
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
├── errors.log               # ERROR-level log (RotatingFileHandler, 512KB, 2 backups) — gitignored
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
| `/memorize` | POST | Extract memories from conversation text. User-initiated only ("Memorize Now" / "Re-memorize chat" buttons). `force=true` = bypass sleep-gap gate, memorize immediately (used by Memorize Now and the turn valve). `rebuild=true` = archive live DB, wipe segment files, reset cursor, re-acquire service (used by Re-memorize chat; implies force). `force=true` batching via `_build_force_memorize_batches()` — prefers segment manifest ranges, falls back to token-window chunking. Runtime progress tracks segments: server persists each selected segment to `segments/*.json`, passes one synthetic segment episode payload into memu batch extraction, and advances conversation cursor per completed segment. Auto-memorize is triggered server-side inside `/conversation/{id}/turn` (see that row). |
| `/retrieve` | POST | Query memories. Optional `as_of` applies temporal triple filtering (`valid_from`/`valid_to`) for graph retrieval. |
| `/timeline` | GET | Entity relationship timeline (`entity`, `user_id`, `soul_id`, optional `as_of`) for chronological graph inspection |
| `/conversation/{id}/retrieve` | POST | Retrieve + build turn prompt for chat path; when caller provides only `query` (no `queries`), server enriches retrieve context with identity/time anchor, `all_categories_summary`, `memory_cache`, `intentions`, and current-chat payload history before calling memu retrieve. Cross-chat merge now reads WhatsApp tails from Hermes source files (`sessions.json` + `state.db`) instead of local `messages` rows. |
| `/conversation/{id}/turn` | POST | Soul turn loop: requires extension-provided `prompt_override_payload` (prepared by `/conversation/{id}/retrieve`), runs LLM with turn contract, persists intentions + cache, and never re-runs retrieve inside turn; system identity uses ST `soul_card` when provided, otherwise self-model-derived card (`narrative_self`); optional free-turn metadata starts immediate Claude Code continuations or stores durable `follow_up` wakes in scoped SQLite; optional `attachment` field in turn contract names a workspace file to deliver as a WhatsApp document (validated inside workspace boundary); APImw fires background `_run_apimw()` pipeline when cadence is met; cadence is now global per soul (scoped `apimw_cadence.turn_count` table, shared across WhatsApp + ST), configured by `retrieve.apimw_cadence`; skips when a prior APImw job is still in flight; runs step A topic statement + step B retrieve + step C second-pass rewrite retrieve when query changes + combined D+E+F selection/edges/intention+cache update, then writes `prior_context`, `retrieval_ids_since_consolidation`, `prior_context_ids_since_consolidation`, `memory_cache`, `intentions_active` and edge invalidations/additions; `message_to_self` is soul-scoped (`soul_state.apimw_message_to_self`) so subconscious notes surface on any next turn regardless of platform; on APImw failure clears `prior_context` to avoid stale context persisting; queues forced memorize only from primary-chat tails (background-chat tails excluded from segment trigger). APImw tracked in `_BACKGROUND_TASKS`; shutdown drains in-flight APImw/consolidation/free-turn work before exit. Background (`memorize_chat=false`) chats trigger adapter-sourced rolling-summary updates on turn lulls. Cross-memorize feed reads `memorize_chat=true` tails from source adapters (WhatsApp Hermes + ST snapshots) and `memorize_chat=false` tails from Hermes row-id cursors via `rolling_summary_cursor_id`; no turn-time raw-message append path remains in server |
| `/conversation/{id}/turn/undo` | POST | Undo latest turn maintenance using `undo_snapshot` (single-step depth) |
| `/integration/memu/turn` | POST | MCP-facing single-call turn wrapper: internally runs conversation-retrieve (`build_turn_prompt=true`) then conversation-turn with prompt override payload |
| `/integration/atomic/chat_profile` | GET | Atomic-facing chat provider profile derived from `config.json` (`llm`), returned in Atomic settings shape for per-message use; includes API key, so callers must not log it |
| `/integration/atomic/search` | GET | Atomic-facing read-only memory search (`q`, `user_id`, `soul_id`, optional `limit`, `since_days`) backed by `GraphMixin.graph_search`; no turn/retrieve state machinery |
| `/integration/memu/retrieve` | POST | MCP-facing retrieve wrapper |
| `/integration/memu/memorize` | POST | MCP-facing memorize trigger wrapper (`force` supported) |
| `/integration/memu/consolidate` | POST | MCP-facing force consolidation wrapper |
| `/conversation/{id}/state` | GET/PATCH | Conversation working state |
| `/conversation/{id}/consolidation/force` | POST | Force consolidation now (consumes all pending segments, still lock-safe) |
| `/souls/{soul_id}/intentions` | GET | List intentions from `intentions_life_goals` table (long-term, consolidation-managed) |
| `/souls/{soul_id}/relationships` | GET/POST | List or create user-declared relationship entities (`memu_entities` rows with `properties.origin=user_declared`) |
| `/souls/{soul_id}/relationships/{speaker_id}` | PATCH/DELETE | Update or soft-delete one relationship entity (`entity:*` only; reserved prefixes rejected) |
| `/souls/{soul_id}/narrative_suggestion` | POST | Snapshot previous `narrative_self` with `evolved_into` chain before overwrite; extension surfaces this via the Memorize Now menu |
| `/categories` | GET | List all categories |
| `/categories/search` | POST | Search categories |
| `/clear` | POST | Delete memories in scope |
| `/config` | GET/POST | Read or update runtime config |
| `/reload` | POST | Reload config from disk |
| `/diag`, `/diag/calls`, `/diag/http`, `/diag/sqlite`, `/diag/sqlite/counts`, `/diag/sqlite/recent` | GET | Diagnostic pages (recent memories, SQLite browser, last 50 memorize/retrieve calls, HTTP introspection). Read-only (no schema writes/migrations on diag reads; never use diag calls for DB bootstrap) |
| `/diag/memorize/pending` | GET | Global memorize pressure (`?soul_id=`, optional `?user_id=`): summed unmemorized primary tokens across all conversations via the same source loaders the turn path uses, vs `min_chunk_tokens` threshold, plus `sleep_gap_ready`. Pure read, handler in `app/main.py` next to `_estimate_primary_memorize_tokens` |

## Extracted Modules

| Module | Purpose |
|--------|---------|
| `app/config.py` | Config/runtime helpers: `load_config()`, `save_config()`, `mask_config()`, storage path normalization, sqlite DSN scoping, default llm profile assembly, soul generation config I/O |
| `app/db.py` | `sqlite_ensure_*()`, `sqlite_connect()`, `json_to_db()`, `json_from_db()`, table column introspection |
| `app/services/consolidation.py` | Consolidation pipeline: gather pending segment queue/context, select an interval-sized chronological window, run one consolidation LLM call, write narrative_self + life-goal edits + companion memory + graph edges, and advance queue/flags. |
| `app/services/diary.py` | Legacy diary helpers for consolidation history. Avoid reintroducing diary/episode chat slices into current consolidation prompts. |
| `app/services/graph_edges.py` | Shared edge normalization + write/invalidate helpers used by APImw and consolidation (`caused_by`, `evokes`, `conflicts_with`, `parallels`, `shaped_by`). |
| `app/services/activity_messages.py` | `activity_messages` scoped-SQLite table helpers for synthetic self-DM activity recaps rendered under `My Activities:`; `main.py` keeps thin `_activity_*` wrappers for existing call sites/tests. |
| `app/services/whatsapp_outbounds.py` | `whatsapp_pending_outbounds` scoped-SQLite queue helpers for WhatsApp replies/attachments plus shared poll-marker helpers; endpoints and production callers still go through thin `main.py` wrappers. |
| `app/services/memorize_endpoint.py` | `/memorize` endpoint core, forced-memorize background runner (`run_memorize_segments`), segment-file persistence (`chat_dir/segments/*.json`), rolling-summary row injection into segment payloads, progress/cancel handlers, and chat sleep-gap/token chunking helpers; listen-only (memorize_chat=false) chats advance rolling_summary_cursor_id (not digest_cursor) at cross-memorize end — cursor stays parked, digest_cursor unchanged; background rollup runs skip `memorize_chat=true` chats; pre-memorize background-tail context uses step id `background_extra_messages` and does not write durable `rolling_summary` |
| `app/services/conversation_sources.py` | Source adapters for cross-conversation reads: WhatsApp from Hermes (`sessions.json` + `state.db`) and SillyTavern from per-conversation snapshots under `resources/st_chats/*/latest_history.json`, with digest-cursor slicing, floor backfill, and row-id based WhatsApp reads for listen-only summarize boundaries. ST snapshots are written atomically (temp+replace) post-turn so WhatsApp turns see completed ST exchanges; web-source chat matching now resolves display aliases by name lookup in whatsapp_contacts table (name/short_name/push_name/verified_name) so conversation IDs using display names resolve correctly; server WhatsApp alias-graph reads removed — server expects canonical `whatsapp:dm:<jid>` / `whatsapp:group:<jid>` from Hermes; `@lid` vs `@s.whatsapp.net` preserved in session matching; dead `_resolve_hermes_base` removed; two WhatsApp live-tail readers merged behind shared `_load_whatsapp_live_rows()` loader/formatter; `_hermes_base()` centralizes `HERMES_HOME` resolution |
| `app/services/cross_history.py` | Cross-conversation history composition: resolves configured source paths, loads current WhatsApp history, formats AI-facing all-chat history, assembles cross-tail/background memorize tails, and manages display-segment cleanup. Low-level source reads stay in `conversation_sources.py`; `main.py` keeps thin `_cross_*` wrappers for existing patch seams. |
| `app/services/apimw.py` | APImw background memory-weaving pipeline: retrieve candidate memories, synthesize prior-context updates/message-to-self, persist results, cadence checks, and background-task launch helpers. `main.py` keeps thin `_apimw*` wrappers and owns generic background-error helpers. |
| `app/services/admin_routes.py` | Health/version/shutdown/diag endpoint handlers (including `/diag` and MCP-prefixed diag aliases). |
| `app/services/payload.py` | Shared payload/scope/normalization helpers used across retrieve/turn/service-factory paths (scope extraction, turn history normalization, signature helpers, payload scrubbing). |
| `app/services/service_factory.py` | Service cache + payload-driven `MemoryService` construction, llm profile merge, and config readers (`apimw_*`, consolidation interval, retrieve config shaping); server `llm.step_models` inject step profile config when client `llm_profiles` are absent. |
| `app/services/retrieve_orchestration.py` | Retrieve domain helpers + orchestration seam: query/where extraction, identity-context builder, retrieve rewrite constants/context-query assembly, and `_run_retrieve` implementation (called via thin wrapper in `main.py`). |
| `app/services/sqlite_scope.py` | SQLite scope plumbing used across endpoints: scoped db-path resolution, scope `WHERE` builder, state-db lookup/write wrappers, and lightweight file info/intention row helpers. |
| `app/services/crud_endpoints.py` | CRUD endpoint logic extracted from `main.py`: categories search/list, intentions, relationships, narrative suggestion, conversation state get/patch, clear-memory. |
| `app/services/mcp_tools.py` | MCP-facing wrapper contracts (`memu_*`): typed request models + thin orchestration over existing REST/turn endpoints |
| `app/services/free_turn.py` | Free-turn continuation chain and `free_turn_followups` scoped-SQLite table helpers for scheduled wakes; `main.py` keeps wrappers so tests and endpoint wiring stay patchable. |
| `app/services/state.py` | `write_conversation_state()`, `conversation_state_from_row()`, cross-DB state search, `pending_diary_segment_ids` queue management; includes background rollup state fields (`rolling_summary`, `rolling_summary_cursor_id`, `rolling_summary_updated_at`); error state fields: `last_background_error` + timestamp (written by APImw on failure, cleared on APImw success); `last_consolidation_error` + timestamp (written by consolidation pipeline, cleared on consolidation success); cursor read helpers in `main.py`: `_cursor_from_row()` (effective digest cursor is `-1` until `last_memorize_at` set) and `_memorize_chat_from_row()` (SQLite NULL means default `true`) |
| `app/services/turn_contract.py` | `make_turn_system_prompt()`, `build_turn_prompt()`, `parse_turn_contract()`, `build_conversations_block()`, `build_turn_context_block()` — soul turn prompt construction and JSON contract parsing. `build_conversations_block()` is the single entry point for all AI-facing chat display (response/retrieve/APImw/consolidation all call it); `build_turn_context_block()` assembles the shared context block (categories, memories, chat display, working thoughts, intentions) used by both normal turns and APImw synthesis; temporal awareness: system prompt includes `Today is [date].` anchor; memory lines include relative-time labels and `reinforced Nx` suffix; episode memory type added to the turn-prompt memory legend only when retrieved episode items are present; APImw `message_to_self` renders as the next numbered `My Working Thoughts` line (not an indented continuation); `activity_recap` removed from normal turn/retrieve system prompts (explicit flag) — self-turn and free-turn continuations still include it |
| `app/services/intention_state.py` | `normalize_intentions_stack()`, `format_intentions_for_prompt()`, `upsert_intentions_stack_entries()` — intentions normalization and prompt formatting |
| `app/services/soul_state.py` (120 lines) | Soul-level singleton state (one row): schema ensure, read/write for `narrative_self`, `all_categories_summary`, `memory_cache`, `intentions_active`, and related soul-scoped fields in `soul_state` SQLite table. |
| `app/services/message_log.py` (85 lines) | Cross-conversation history rendering helpers: `format_merged_history()` via memu grouped-chat formatter, source-label derivation, WhatsApp identifier normalization, and activity-log block assembly. |
| `app/services/xml_utils.py` (22 lines) | Thin XML helpers: `extract_xml_fragment()` (regex-finds a root tag then parses it) and `xml_text()` (safe element text extraction); used by consolidation pipeline. |
| `app/services/narrative_self.py` (45 lines) | `snapshot_previous_narrative_self()` — writes a `narrative_self`-typed Triple to preserve the prior identity text with an `evolved_into` chain before consolidation overwrites it. |
| `app/services/segment.py` (88 lines) | Segment helpers for consolidation: `parse_segment_range()`, `build_segment_inputs()` (assembles per-segment context for the consolidation LLM call), `create_companion_memory()`. |
| `app/procedural.py` (279 lines) | Procedural-memory sidecar: ingests per-domain YAML corpora from `memu/procedural/<domain>.yaml` into a standalone `procedural.db` SQLite file with pre-computed embeddings; serves top-k cosine lookups at retrieve time. Soul never writes here. |

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
retrieve:   apimw_enabled (bool; toggles APImw pipeline), apimw_cadence (int, default 5; APImw runs every N successful soul turns across chats), apimw_memory_count (int, default 20; APImw item.top_k), apimw_random_count (int, default 5; APImw random sample size), mental_health_query (bool, default true; enables MH procedural retrieve step; per-request `mental_health_addon` overrides when explicitly boolean)
claude_code: claude_code_workspace (str; persistent workspace dir for soul session/resume calls), claude_code_permission_mode (str; e.g. bypassPermissions), claude_code_timeout_seconds (int, default 3600; per-call timeout)
memorize:   min_chunk_tokens (default 8000; floor for sleep-gap-triggered memorize), background_summary_tokens (default 1000; token threshold that triggers durable rolling-summary replacement for memorize_chat=false chats), background_extra_messages_tokens (default 100; pre-memorize background-tail summary floor appended to segment context, not a durable rollup trigger)
consolidation_interval_days: pending segment window size for consolidation after successful memorize runs (default 7)
debug:      log_prompts (bool) — dumps exact LLM prompt + response for memorize/consolidation steps to console
```

## Config-Only Runtime

- Runtime configuration is read from `config.json`.
- Database DSN source is `storage.metadata_store.dsn` in `config.json`.
- `run.py` and app runtime no longer use `DATABASE_URL` / `DATABASE_*` env branches.
