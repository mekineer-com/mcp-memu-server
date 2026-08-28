# mcp-memu-server — FastAPI Server Index

> This file exists so agents can orient themselves without scanning the tree.
> `.claudeignore` blocks auto-scan of this directory — read this first.

`# Marcos' reminder: a category is a dossier.`

## Layout

```
mcp-memu-server/
├── app/main.py              # Core orchestration + remaining endpoints
├── app/config.py            # Runtime config load/save/mask + path + sqlite DSN helpers
├── app/db.py                # SQLite helpers, schema ensures, JSON marshalling
├── app/models/base.py       # Declarative ORM base
├── app/services/consolidation.py
├── app/services/memorize_endpoint.py
├── app/services/activity_messages.py
├── app/services/whatsapp_outbounds.py
├── app/services/free_turn.py
├── app/services/cross_history.py
├── app/services/conversation_id.py
├── app/services/apimw.py
├── app/services/sqlite_scope.py
├── app/services/crud_endpoints.py
├── app/services/state.py
├── app/services/soul_summaries.py
├── run.py                   # Entry point: config load, sys.path setup, single-instance pid guard, uvicorn start
├── migrate_category_taxonomy.py # Offline inventory/discover/apply/validate migration; explicit DB only
├── config.json              # Runtime config (llm, storage, listen, dossier policy, memu path)
├── config.example.json      # Template
├── tests/                   # pytest suite (`./.venv/bin/python -m pytest -q tests/`)
├── alembic/                 # DB migration scripts
├── storage/                 # Default SQLite DB + resource dir
├── errors.log               # ERROR-level log (RotatingFileHandler, 512KB, 2 backups) — gitignored
├── Makefile                 # make install/run/test/check
└── pyproject.toml           # Python 3.12+ deps
```

## Key Endpoints (registered from `app/main.py`; admin/diag in `app/services/admin_routes.py`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serves bundled memU-ui if present, else a JSON health stub |
| `/health` | GET | Health check |
| `/version` | GET | Build / server instance identity |
| `/admin/shutdown` | POST | Request graceful shutdown (drain mode) |
| `/admin/shutdown/status` | GET | Shutdown progress + active request counts |
| `/memorize` | POST | Extract memories from conversation. `force=true` bypasses sleep-gap; `rebuild=true` wipes and resets cursor (implies force). Auto-memorize also fires inside `/conversation/{id}/turn`. |
| `/memorize/progress` | GET | Live memorize batch progress |
| `/memorize/cancel` | POST | Cancel the running memorize batch |
| `/retrieve` | POST | Query memories. Optional `as_of` for temporal triple filtering. |
| `/graph` | GET | Recent memory graph (items + edges) for graph clients |
| `/timeline` | GET | Entity relationship timeline |
| `/conversation/{id}/retrieve` | POST | Retrieve + build turn prompt: enriches query with identity, categories, memory cache, intentions, and current-chat history before calling memu. |
| `/conversation/{id}/turn` | POST | Soul turn loop: runs LLM with turn contract, persists intentions + cache, fires APImw in background on cadence, manages free-turn continuations and attachments. |
| `/conversation/{id}/turn/undo` | POST | Undo latest turn (single-step, `undo_snapshot`) |
| `/integration/memu/turn` | POST | MCP single-call turn wrapper: retrieve then turn |
| `/integration/mentra/health` | GET | Bearer-authenticated Mentra ingress health check; disabled by default |
| `/integration/mentra/session/start` | POST | Authenticated soul bootstrap plus constrained Gemini Live token; returns a fresh sitting ID and next device-conversation transcript sequence |
| `/integration/mentra/session/{id}/token` | POST | Mint a fresh constrained Gemini token for the unchanged active sitting before a replacement socket |
| `/integration/mentra/session/{id}/heartbeat` / `end` | POST | Renew or release one sitting-scoped Mentra lease |
| `/integration/mentra/session/{id}/recall` | POST | Sitting-scoped, read-only forced retrieve over the cursor-bounded Mentra tail; returns compact ID-free context for Gemini `SILENT` delivery |
| `/integration/mentra/session/{id}/transcripts/append` | POST | Redacted-validation, contiguous/idempotent transcript or sitting-summary append into the atomic Mentra snapshot; newly accepted eligible rows queue the shared auto-memorize path |
| `/integration/atomic/session_start` | POST | Atomic session bootstrap: stripped retrieve snapshot → seeds `chat:atomic-<uuid>` |
| `/integration/atomic/session_end` | POST | Atomic session close: accepts transcript + `activity_recap`, posts to memU memorize |
| `/integration/atomic/chat_profile` | GET | Atomic-facing LLM profile (includes API key — do not log) |
| `/integration/atomic/prompt_log` | POST | Atomic prompt-log sink (writes to `mcp-memu-server.log` when `debug.log_prompts` enabled) |
| `/integration/atomic/atoms` | GET | Paginated atom list with canonical dossier metadata; `category_id`/`tag_id` filter, `cursor` pagination |
| `/integration/atomic/tags` | GET | Category/tag list with kind, activity state, and counts |
| `/integration/atomic/entities` | GET/POST | Scoped entity list or create-always entity write |
| `/integration/atomic/entities/{entity_id}` | GET/PATCH/DELETE | Scoped entity detail, stable-ID name/free-text-type/alias edit, or safe deletion of an unreferenced extracted entity |
| `/integration/atomic/entities/{entity_id}/ignore` / `restore` | POST | Reversibly suppress or restore an extracted entity without changing its references |
| `/integration/atomic/entities/{entity_id}/merge-preview` / `merge` | GET/POST | Preview then atomically absorb one duplicate entity into the open canonical entity |
| `/integration/atomic/memories/{memory_id}/entities/{entity_id}` | PUT/DELETE | Transactionally attach/detach one current `mentions` edge |
| `/integration/atomic/canvas-source` | GET/POST | Canvas source: all memories + categories; POST accepts `atom_ids[]` for subset rebuilds |
| `/integration/atomic/neighborhood/{item_id}` | GET | Cosine-similar memory neighborhood |
| `/integration/atomic/similar/{item_id}` | GET | Pure cosine-similarity graph (no DB expansion) |
| `/integration/atomic/search` | GET | Read-only exact-`M#`/FTS/vector search with optional memory-only and pre-limit entity/dossier exclusions |
| `/integration/whatsapp/outbounds/claim` | POST | Claim queued WhatsApp replies/attachments for delivery |
| `/integration/whatsapp/outbounds/mark` | POST | Mark a claimed WhatsApp outbound sent or failed |
| `/integration/memu/retrieve` | POST | MCP retrieve wrapper |
| `/integration/memu/memorize` | POST | MCP memorize wrapper (`force` supported) |
| `/integration/memu/consolidate` | POST | MCP force-consolidation wrapper |
| `/conversation/{id}/state` | GET/PATCH | Conversation working state |
| `/conversation/{id}/consolidation/force` | POST | Force consolidation now (lock-safe) |
| `/souls/{soul_id}/intentions` | GET | Long-term life goals (`intentions_life_goals` table) |
| `/souls/{soul_id}/relationships` | GET/POST | User-declared relationship entities; POST may promote an exact entity ID |
| `/souls/{soul_id}/relationships/{speaker_id}` | PATCH/DELETE | Update or remove Relationship properties from one stable `entity:<entities.id>` reference |
| `/souls/{soul_id}/narrative_suggestion` | POST | Apply a soul-evaluated narrative change with history + old-self snapshot |
| `/pending` | GET | Review queue: unapproved memories/dossiers and persistent narrative self |
| `/soul-summary/{kind}` | PATCH | Journal and approve an Atomic manual correction with snapshot guard |
| `/soul-summary/{kind}/approve` | POST | Approve the displayed soul-summary value with snapshot guard |
| `/memory/{item_id}` | GET | Single memory detail with graph context |
| `/memory/{item_id}` | PATCH | Edit memory value (write-live, approve-later) |
| `/memory/{item_id}/approve` | POST | Bless current memory value |
| `/memory/{item_id}` | DELETE | Hard-delete with dependent cleanup (fts, edit_history, triples) |
| `/category/{category_id}` | PATCH | Edit dossier title/description/prose with snapshot guard |
| `/category/{category_id}/memory/{memory_id}` | PUT/DELETE | Snapshot-guarded dossier membership attach/detach |
| `/category/{category_id}/approve` | POST | Bless current category summary |
| `/categories` | GET | List all categories |
| `/categories/search` | POST | Search categories |
| `/clear` | POST | Delete memories in scope |
| `/config` | GET/POST | Read or update runtime config |
| `/reload` | POST | Reload config from disk |
| `/diag`, `/diag/calls`, `/diag/http`, `/diag/sqlite/*` | GET | Diagnostic pages. Read-only — never use for DB bootstrap. |
| `/diag/memorize/pending` | GET | Global memorize pressure: unmemorized tokens vs threshold + sleep-gap status |

## Extracted Modules

| Module | Purpose |
|--------|---------|
| `app/config.py` | `load_config()`, `save_config()`, `mask_config()`, storage path normalization, sqlite DSN scoping |
| `app/db.py` | `sqlite_ensure_*()`, `sqlite_connect()`, `json_to_db()`, `json_from_db()`, table column introspection |
| `app/services/consolidation.py` | Consolidation pipeline: segment queue → holistic due-dossier revision → anchor reflection → narrative_self + life-goals + companion memory + graph edges |
| `app/services/graph_edges.py` | Edge normalization + write/invalidate helpers (`caused_by`, `evokes`, `conflicts_with`, `parallels`, `shaped_by`) |
| `app/services/activity_messages.py` | `activity_messages` scoped-SQLite table for synthetic self-DM activity recaps (`My Activities:`) |
| `app/services/whatsapp_outbounds.py` | `whatsapp_pending_outbounds` scoped-SQLite queue for WhatsApp replies/attachments |
| `app/services/mentra_routes.py` | Authenticated Mentra boundary: sitting-scoped lease lifecycle, bootstrap/token mint, non-blocking memory recall, and contiguous transcript append/ack |
| `app/services/memorize_endpoint.py` | `/memorize` core: segment-file persistence, forced-memorize runner, rolling-summary injection, sleep-gap/token chunking, progress/cancel. Listen-only segments advance source cursors without producing memory, consuming rolling summaries, or retaining segment files. |
| `app/services/conversation_sources.py` | Source adapters: WhatsApp from `web_source.db`; ST/Atomic from resource snapshots; Mentra from `openalma/mentra/transcripts`. Handles atomic writes, cursor slicing, floor backfill, and role normalization. |
| `app/services/cross_history.py` | Cross-conversation history: formats AI-facing all-chat history, assembles cross-tail/background memorize feeds, manages display-segment cleanup |
| `app/services/apimw.py` | APImw background memory-weaving pipeline: retrieve → synthesize prior-context/message-to-self → persist |
| `app/services/admin_routes.py` | Health/version/shutdown/diag endpoint handlers |
| `app/services/payload.py` | Shared payload/scope/normalization helpers (scope extraction, turn history normalization, signature helpers) |
| `app/services/service_factory.py` | `MemoryService` cache + construction, llm profile merge, config readers |
| `app/services/retrieve_orchestration.py` | Retrieve domain helpers: query/where extraction, identity-context builder, `_run_retrieve` implementation |
| `app/services/sqlite_scope.py` | Scoped db-path resolution, scope `WHERE` builder, state-db lookup/write wrappers |
| `app/services/crud_endpoints.py` | CRUD endpoint logic: categories, intentions, relationships, narrative suggestion, conversation state, clear |
| `app/services/conversation_id.py` | WhatsApp group ID canonicalization to `whatsapp:group:<group@g.us>`. Called at all entrypoints — do not bypass or per-sender aliases will split state. |
| `app/services/mcp_tools.py` | MCP-facing wrapper contracts (`memu_*`) |
| `app/services/free_turn.py` | Free-turn continuation chain and `free_turn_followups` scoped-SQLite table for scheduled wakes |
| `app/services/state.py` | `write_conversation_state()`, `conversation_state_from_row()`, cross-DB state search, queue management. Canonicalizes WhatsApp group IDs before all reads/writes. |
| `app/services/turn_contract.py` | `make_turn_system_prompt()`, `build_turn_prompt()`, `parse_turn_contract()`, `build_conversations_block()`, `build_turn_context_block()` — soul turn prompt construction and JSON contract parsing. Single entry point for all AI-facing chat display. |
| `app/services/intention_state.py` | Intentions normalization and prompt formatting. Owns memory cache entry caps (`MAX_MEMORY_CACHE_ENTRIES`, `MAX_MEMORY_CACHE_ENTRY_CHARS`). |
| `app/services/soul_state.py` | Soul-level singleton state: `narrative_self`, `memory_cache`, `intentions_active`; the dossier index is projected from memU, never stored |
| `app/services/soul_summaries.py` | Journaled live/previous/approved soul-summary writes and review revision guards |
| `app/services/message_log.py` | Cross-conversation history rendering: `format_merged_history()`, source-label derivation, WhatsApp normalization, activity-log block |
| `app/services/xml_utils.py` | `extract_xml_fragment()`, `xml_text()` — used by consolidation pipeline |
| `app/services/narrative_self.py` | `snapshot_previous_narrative_self()` — writes `evolved_into` Triple before consolidation overwrites identity |
| `app/services/segment.py` | Consolidation segment helpers: `parse_segment_range()`, `build_segment_inputs()`, `create_companion_memory()` |
| `app/procedural.py` | Procedural-memory sidecar: YAML corpora → `procedural.db` with embeddings; top-k cosine lookup at retrieve time. Soul never writes here. |

## How It Connects to memu

```python
from memu.app import MemoryService    # main facade
from memu.prompts.consolidation import anchors, dossiers  # two-call reflection prompts
from memu.prompts.memory_type import ...  # type prompts
```

- `config.json` → `memu.path` points to memu/src
- `run.py` inserts that path into `sys.path[0]`
- `MemoryService` instances cached per unique (llm_profiles + db_config) pair in `_SERVICES` dict

## Task → Where to Look

| Task | File |
|------|------|
| Add API endpoint | `app/main.py` — thin handler; logic in `app/services/*` |
| Modify memorize flow | `app/services/memorize_endpoint.py` |
| Modify retrieval flow | `app/services/retrieve_orchestration.py` |
| Modify soul turn loop | `app/main.py` → `conversation_turn()` + `app/services/turn_contract.py` |
| Modify consolidation | `app/services/consolidation.py` |
| Modify conversation state | `app/services/state.py` |
| Modify turn intentions (working stack) | `app/services/intention_state.py` |
| Modify life goals (long-term) | `app/services/consolidation.py` — `intentions_life_goals` table |
| Modify DB schema/helpers | `app/db.py` |
| Change config shape | `config.json` + `app/config.py` |
| Migrate legacy categories to dossiers | `migrate_category_taxonomy.py` + root `PLAN_category_taxonomy_slice_G_migration.md` |

## Config (`config.json`)

```
llm:        provider, api_key, base_url, chat_model, embed_model
storage:    resources_dir, sqlite_dir, metadata_store (provider + dsn)
hermes:     home, state_db_path, sessions_index_path, whatsapp_web_source_db (Channels data paths)
mentra:     enabled, gemini_api_key, model, voice, integration_bearer_token
mcp:        http_path, sse_path
listen:     host, port
memu:       path (to memu/src)
python:     executable, force_venv
pid_file:   server pid path
categories: dynamic dossier cluster size and revision target words
retrieve:   apimw_enabled, apimw_cadence, apimw_memory_count, apimw_random_count, mental_health_query
claude_code*: top-level keys, not a block — claude_code, claude_code_model, claude_code_effort, claude_code_permission_mode, claude_code_settings, claude_code_workspace, claude_code_timeout_seconds
memorize:   min_chunk_tokens, episodes_per_segment, background_summary_tokens, background_extra_messages_tokens, enable_confidence_normalization, semantic_dedupe_enabled, semantic_dedupe_similarity_threshold
procedural: yaml_dir, db_path
consolidation_interval_days: pending segment window (default 7)
turn_response_sentences: soul reply length target
debug:      log_prompts (bool)
```

## Config-Only Runtime

- Runtime configuration is read from `config.json`.
- Database DSN source is `storage.metadata_store.dsn` in `config.json`.
- `run.py` and app runtime no longer use `DATABASE_URL` / `DATABASE_*` env branches.
