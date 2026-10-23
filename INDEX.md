# mcp-memu-server — FastAPI Server Index

> This file exists so agents can orient themselves without scanning the tree.
> `.claudeignore` blocks auto-scan of this directory — read this first.

`# Marcos' reminder: a category is a dossier.`

For the end-to-end flow, source ownership, save boundaries and Retry, read
[MEMORIZE_PIPELINE.md](MEMORIZE_PIPELINE.md). This index locates code; the map
explains how the client, server and engine stages fit together.

## Layout

```
mcp-memu-server/
├── app/main.py              # Core orchestration + remaining endpoints
├── app/config.py            # Runtime config load/save/mask + path + sqlite DSN helpers
├── app/db.py                # SQLite helpers, schema ensures, JSON marshalling
├── app/models/base.py       # Declarative ORM base
├── app/services/consolidation.py # Shared runner, read-only prompt preparation, Resource-owned segment reconstruction and selected prior context
├── app/services/memorize_endpoint.py # Shared extraction with same-transaction primary checkpoint/phase publication; rolling state stays rollup-owned
├── app/services/activity_messages.py
├── app/services/whatsapp_outbounds.py
├── app/services/free_turn.py
├── app/services/cross_history.py
├── app/services/import_routes.py # Imported-source admission, bounded history, saved-segment recovery and shared cross-chat Memorize Retry/handoff
├── app/services/conversation_sources.py # Client-owned source reads; import SQL selects immutable mode/position, bounded historical reads and indexed current floors
├── app/services/conversation_id.py
├── app/services/apimw.py
├── app/services/sqlite_scope.py
├── app/services/souls.py
├── app/services/owner.py          # One create-once OpenAlma user identity; shared loopback setup guard
├── app/services/crud_endpoints.py
├── app/services/graph_routes.py   # Graph/query read HTTP routes; scoped resolver supplied by main
├── app/services/entity_routes.py  # Entity HTTP routes and private helpers; scoped resolver supplied by main
├── app/services/review_routes.py  # Review routes; category prose/title/description snapshot + revision checked in the caller-owned write transaction
├── app/services/state.py # Caller-transactional phase/checkpoint patches and fresh Memorize failure preservation; ordinary/import cursors remain separate
├── app/services/soul_summaries.py # SQL-only summary writes; shared journal finishing after caller commit
├── app/services/soul_state.py # Shared Soul state, durable Memorize failure and common activity-pause predicate
├── run.py                   # Entry point: config load, sys.path setup, single-instance pid guard, uvicorn start
├── migrate_release.py       # Explicit, rerunnable release migrations; subconscious identifier rename and one-time expiration of legacy queued Prior Context prose
├── _archive/database-tools/ # Historical one-time tools; not part of runtime or active tests
├── config.json              # Runtime config (llm, storage, listen, dossier policy, memu path)
├── config.example.json      # Minimal installable template
├── config.json.md           # Complete setting reference
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
| `/memorize` | POST | Extract memories from a frozen captured conversation. ST sends saved rows, original positions, `source_message_count` and `intent` (`manual`, `automatic`, `publish_only`); source publication precedes paid admission. Only accepted processing gets a task. `force=true` bypasses sleep-gap; explicit `rebuild=true` archives/reset with the Soul's name binding preserved, but briefly refuses while that Soul has registered photo-filing work. Other clients retain turn-driven auto-memorize. |
| `/imports/register` | POST | Register/reuse a stored imported chat. First registration fixes historical eligibility using scoped saved conversation segments; refuse it while memory work is running. Re-register never extends the bound or resets progress/errors. No model call. |
| `/imports/validate` | POST | Validate new current rows against the actual turn allowance. Return the complete Soul-wide pending period, selected-source processed dates and prospective deferred-history status against the fixed bound. Unreadable sources fail validation. No model call, source insertion or Memorize scheduling. |
| `/imports/process`, `/imports/retry` | POST | Start tracked historical work, returning promptly. Default one batch; `continuous=true` repeats successful bounded batches under one claim. Failure requires explicit Retry. |
| `/imports/continuation` | POST | Change the running import task's continuation choice; the current batch finishes. No restart resumption or second busy registry. |
| `/imports/status` | GET | Scoped imported-chat checkpoint, error, matching task activity/continuation and its progress; `followup_error` exposes failed memory work after historical completion. Unrelated Soul work is not import activity. |
| `/memorize/retry` | POST | Retry failed ordinary Memorize or a completed import's waiting handoff. Unfinished import history must finish in Echo first. |
| `/memorize/progress` | GET | Live memorize batch progress |
| `/memorize/cancel` | POST | Cancel the running memorize batch |
| `/retrieve` | POST | Query memories. Optional `as_of` for temporal triple filtering. |
| `/graph` | GET | Recent memory graph (items + edges) for graph clients |
| `/timeline` | GET | Entity relationship timeline |
| `/conversation/{id}/retrieve` | POST | Retrieve + build turn prompt: enriches query with identity, categories, memory cache, intentions, and current-chat history before calling memu. |
| `/conversation/{id}/turn` | POST | Soul turn loop: runs LLM with turn contract, may annul current intentions, persists cache, fires APImw in background on cadence, and manages continuations/attachments. |
| `/conversation/{id}/turn/undo` | POST | Undo only the snapshot matching the reply's `turn_id`; missing/stale IDs leave state intact. Then best-effort delete its uncited annulment reflections |
| `/conversation/{id}/source` | DELETE | Retire a permanently deleted ST chat from fresh history and rolling context. Preserve filed memories, accepted work, checkpoints and pending Resources; refuse late source publication. |
| `/integration/memu/turn` | POST | MCP single-call turn wrapper: retrieve then turn |
| `/integration/memu/sensory-search` | POST | Scoped explicit visual-memory candidate search over separate media and caption lanes |
| `/integration/mentra/health` | GET | Keyless private-network Mentra ingress health check |
| `/integration/mentra/installation/seen` | POST | Record an installed Iris build for the shared owner; optional stock setup ticket is consumed with its first app report, allowing same-ID repeats only |
| `/integration/mentra/host/seen` | POST | Record OpenAlma Mentra's own installation ID, native host version and default name before Iris exists; preserve renamed labels |
| `/mentra/installations` | POST | Loopback-only stock setup ticket reservation; optional `device_session_id` binds an Update ticket to an existing stock app, preserving its ID/name |
| `/mentra/installations/{device_session_id}` | PATCH / DELETE | Loopback-only Rename / Forget metadata, through the existing JSON writer/lock; Forget refuses an active sitting and never deletes chats |
| `/integration/mentra/earcons/{name}.wav` | GET | Serve a bundled audio cue; public, unauthenticated |
| `/integration/mentra/status` | GET | Keyless reported-installation discovery (pending setup tickets stay private), scoped active lease, global busy/start claims, latest-sitting transcript gap and scoped failed photo IDs. Read-only: does not restart filing work. Existing leases remain visible for Stop protection. Static earcons remain public. |
| `/integration/mentra/session/start` | POST | Private-network scoped Soul bootstrap plus constrained Gemini Live token; requires and echoes `photo_work_protocol: "iris-photo-work-v2"`. Redacted validation precedes side effects. Optional `recovery: {sitting_id, events}` publishes original numbered rows through shared transcript merge/publication before context/token; returns `recovery: {sitting_id, ack_sequence}` (last supplied row), or null when absent. Conflicts return 409 without overwriting history. Refuses paused Souls; returns a fresh sitting ID and next shared `mentra:iris` transcript sequence. Same installation can replace its own lease; another installation cannot take over |
| `/integration/mentra/session/{id}/token` | POST | Mint a fresh constrained Gemini token for the unchanged active sitting before a replacement socket; same six NON_BLOCKING tools as Start, including argument-free `nothing_to_add` handled locally by Iris (no new endpoint). The application marker does not change Google's `v1beta` socket contract |
| `/integration/mentra/session/{id}/heartbeat` / `end` | POST | Renew or release one sitting-scoped Mentra lease; heartbeat includes nullable `pause_reason` and repeats each image-processing failure until that image succeeds or the sitting ends. Existing token renewal, append and End remain available while paused |
| `/integration/mentra/session/{id}/recall` | POST | Sitting-scoped, read-only forced retrieve over the cursor-bounded Mentra tail; returns compact ID-free context for Gemini `SILENT` delivery |
| `/integration/mentra/session/{id}/working-state` | POST | Read current working thoughts and intentions for the active Soul/sitting; no bootstrap snapshot |
| `/integration/mentra/session/{id}/working-thought` | POST | Append a bounded working thought through the existing Soul lock and state writer; busy returns not saved |
| `/integration/mentra/session/{id}/annul-intention` | POST | Remove a current intention using shared completion-memory/state-save cleanup; recheck the sitting after preparation, never automatically replay |
| `/integration/mentra/session/{id}/snapshot` | POST | Accept one durable image snapshot for background processing |
| `/integration/mentra/session/{id}/snapshot/replay` | POST | Replay a stored snapshot with its scoped media reference, including recovery of an unconfirmed upload. Includes `filing: {caption, caption_event_id, status}` when a receipt with matching owner/Soul/image/media identity exists; read-only, no recaption or worker launch |
| `/integration/mentra/session/{id}/snapshot/finalize` | POST | Under an active scoped lease, accept a stored original, caption and stable `caption_event_id` directly, independently of timeline append. The durable receipt owns idempotency and server-received `source_timestamp`; conflicting captions return 409. Missing/ambiguous scoped originals return 404. Photos use the existing ordered worker, with explicit Retry for failures. Gemini embedding config and DB profile remain required for new handoffs |
| `/integration/mentra/photos/{image_id}/retry` / `discard` | POST | Owner/Soul-scoped explicit recovery of a persisted failed photo filing, available without an active sitting. Retry reuses the original caption/worker; Discard keeps original bytes. Finalize returns `durable: true` only after scoped filing ownership is persisted; Start reconciles interrupted queued/running filings, never reported failures. |
| `/integration/mentra/session/{id}/transcripts/append` | POST | Redacted-validation, contiguous/idempotent transcript, photo-caption, image, gap, or sitting-summary append into the atomic Mentra snapshot; optional user-transcript `mic_source` is display metadata, not replay identity. Captions retain event/media metadata and read as internal visual perceptions, not heard speech. Conversational rows queue shared auto-memorize while gap markers never enter AI history |
| `/integration/atomic/session_start` | POST | Atomic session bootstrap: paused Souls refused before session/history writes; stripped retrieve snapshot → seeds `chat:atomic-<uuid>` |
| `/integration/atomic/session_end` | POST | Atomic session close: persists transcript, then atomically records the activity recap and End marker |
| `/integration/atomic/chat_profile` | GET | Scoped `user_id` + `soul_id` generation gate and LLM profile (includes API key — do not log); paused Soul returns 409 |
| `/integration/atomic/prompt_log` | POST | Atomic prompt-log sink (writes to `mcp-memu-server.log` when `debug.log_prompts` enabled) |
| `/integration/atomic/atoms` | GET | Paginated atom list with canonical dossier metadata; `category_id`/`tag_id` filter, `cursor` pagination |
| `/integration/atomic/tags` | GET | Category/tag list with kind, activity state, and counts |
| `/integration/atomic/entities` | GET/POST | Scoped entity list or create-always entity write |
| `/integration/atomic/entities/{entity_id}` | GET/PATCH/DELETE | Scoped entity detail, stable-ID name/free-text-type/alias edit, or deletion with hidden-link cleanup in one transaction. Current mentions from active memories and identity/dossier/speaker safeguards still block deletion |
| `/integration/atomic/entities/{entity_id}/ignore` / `restore` | POST | Reversibly suppress or restore an extracted entity without changing its references |
| `/integration/atomic/entities/{entity_id}/merge-preview` / `merge` | GET/POST | Preview then atomically absorb one duplicate entity into the open canonical entity |
| `/integration/atomic/memories/{memory_id}/entities/{entity_id}` | PUT/DELETE | Transactionally attach/detach one current `mentions` edge |
| `/integration/atomic/canvas-source` | GET/POST | Canvas source: all memories + categories; POST accepts `atom_ids[]` for subset rebuilds |
| `/integration/atomic/neighborhood/{item_id}` | GET | Cosine-similar memory neighborhood |
| `/integration/atomic/similar/{item_id}` | GET | Pure cosine-similarity graph (no DB expansion) |
| `/integration/atomic/search` | GET | Read-only exact-`M#`/FTS/vector search with optional memory-only and pre-limit entity/dossier exclusions |
| `/integration/whatsapp/outbounds/claim` | POST | Claim queued WhatsApp replies/attachments for delivery |
| `/integration/whatsapp/outbounds/mark` | POST | Mark a claimed WhatsApp outbound sent or failed using its exact returned `claimed_at`; stale claims receive 409, missing claim evidence 400. Insert/mark never create a missing Soul DB |
| `/integration/memu/retrieve` | POST | MCP retrieve wrapper |
| `/integration/memu/memorize` | POST | MCP memorize wrapper (`force` supported) |
| `/integration/memu/consolidate` | POST | MCP force-consolidation wrapper |
| `/conversation/{id}/state` | GET/PATCH | Conversation working state |
| `/conversation/{id}/consolidation/force` | POST | Force consolidation now (lock-safe) |
| `/conversation/{id}/consolidation/retry` | POST | Manual retry of a failed consolidation (409 when the last run did not fail). Failed paid runs never retry automatically. |
| `/memorize/retry` | POST | Scoped launcher recovery through existing all-chat assembly, including activity and context-only tails when the original chat has no new tail. Unavailable WhatsApp input is logged and skipped without advancing its cursor; other unreadable sources still require restoration. Saved Resource work remains recoverable; activity stays paused until Retry succeeds |
| `/souls/{soul_id}/relationships` | GET/POST | User-declared relationship entities; POST may promote an exact entity ID |
| `/souls/{soul_id}/relationships/{speaker_id}` | PATCH/DELETE | Update or remove Relationship properties from one stable `entity:<entities.id>` reference |
| `/souls/{soul_id}/narrative_suggestion` | POST | Apply a soul-evaluated narrative change with history + old-self snapshot; reject concurrent self-description changes before saving |
| `/owner` | GET/POST | Loopback-only read/create contract for the one OpenAlma user identity |
| `/integration/mentra/owner` | GET | Keyless private-network Iris discovery of the shared owner; creation belongs to the launcher's loopback `/owner` route |
| `/pending` | GET | Review queue: unapproved memories/dossiers and persistent narrative self |
| `/soul-summary/{kind}` | PATCH | Journal and approve an Atomic manual correction with snapshot guard |
| `/soul-summary/{kind}/approve` | POST | Approve the displayed soul-summary value with snapshot guard |
| `/memory/{item_id}` | GET | Single memory detail with graph context |
| `/memory/{item_id}` | PATCH | Snapshot-guarded memory edit (write-live, approve-later) |
| `/memory/{item_id}/approve` | POST | Snapshot-guarded approval of the displayed memory value |
| `/memory/{item_id}` | DELETE | Snapshot-guarded hard-delete with dependent cleanup (fts, edit_history, triples) |
| `/category/{category_id}` | PATCH | Edit dossier title/description/prose with snapshot guard |
| `/category/{category_id}/memory/{memory_id}` | PUT/DELETE | Snapshot-guarded dossier membership attach/detach |
| `/category/{category_id}/approve` | POST | Bless current category summary |
| `/categories` | GET | List all categories |
| `/categories/search` | POST | Search categories |
| `/clear` | POST | Delete memories in scope |
| `/souls` | GET/POST | Discover or create canonical scoped soul databases; keyless private-network Mentra alias at `/integration/mentra/souls` |
| `/souls/{soul_id}` | GET | Read immutable per-Soul human name without creating schema or a service; optional `include_narrative=true` adds a live WAL-aware narrative read. Shared Mentra alias |
| `/integration/atomic/memories` | POST | Create one exact, approved human-authored `knowledge` memory for an owner+soul scope |
| `/config` | GET/POST | Read or update runtime config; model patches merge mapping rows without erasing other settings, and mask nested provider credentials |
| `/reload` | POST | Reload config from disk |
| `/diag`, `/diag/calls`, `/diag/http`, `/diag/sqlite/*` | GET | Diagnostic pages. Read-only — never use for DB bootstrap. |
| `/diag/memorize/pending` | GET | Scoped Soul memorize pressure, read-only `context_load_tokens` forecast at the threshold, source warnings, recovery state and historical import-claim activity |

Iris history, media and Memorize use one `conversation_sources.MENTRA_CONVERSATION_ID`
(`mentra:iris`) within each existing owner/Soul scope. App IDs identify installations
and sitting ownership, not separate chats. No old-chat conversion or fallback reader.

### Soul Setup Contract

`GET /souls` returns exact `*.db` filename stems except the configured base DB and
symlinks. `POST /souls` accepts `{soul_id, use_existing: bool}`;
new Souls initialize their internal human name from the saved OpenAlma owner. It returns
`{soul_id, created: bool}`.
Both methods share their implementation with `/integration/mentra/souls` over
the trusted private network, with no connection key.
Local routes follow the trusted-local API convention. `GET /souls/{soul_id}`
returns `{soul_id, user_name}` through a direct read-only query; missing bindings
fail rather than creating or migrating a DB. `include_narrative=true` also returns
`narrative_self` using a separate WAL-aware read-only connection; this opt-in read
may create SQLite side files but never schema, state or an AI service. Creation
publishes the stored name with the staged DB; existing bindings stay unchanged.
Existing-Soul consent does not overwrite it. Scope/speaker IDs remain unchanged;
the stored name supplies human anchor and extraction/reflection prompt labels.

Only `POST /souls` publishes a regular-file soul database. Any scoped request for an unknown
soul fails; read/turn/outbound paths never create identity as a side effect.

Exact existing names require consent: 409 `detail.reason=existing_exact` plus
`detail.message`; consent returns `created: false`. New names are published atomically.
Soul IDs are trimmed exact names: case and spaces remain identity-significant.
Discovery never opens database contents; an unreadable directory remains an error.
Exact-name chat directories intentionally replace old sanitized-name directories; no compatibility lookup is performed.

Picker integration: keep installation records unchanged. Call scoped Mentra status
with the selected user/soul/device to retain gap visibility after the lease ends.
Unscoped status lists installations and owner-scoped live sittings; it does not
guess a chat from the last reporting app.

Mentra Start accepts optional `model`: exactly `gemini-3.8-live` or
`gemini-3.8-live-extended-thinking`. Omission retains the configured model,
including custom IDs; an explicit choice needs no configured model and never
edits config. Start returns the resolved model; the lease pins token renewal.
Only the exact Thinking model adds `generationConfig.thinkingConfig.thinkingLevel: LOW`.
Status exposes nullable `configured_model` and scoped `active_model` as information;
Start remains authoritative.

## Extracted Modules

| Module | Purpose |
|--------|---------|
| `app/config.py` | `load_config()`, `save_config()`, `mask_config()`, `normalize_model_mappings()`, `merge_model_config()`, operation-specific profile selection, storage path normalization, sqlite DSN scoping |
| `app/db.py` | `sqlite_ensure_*()`, `sqlite_connect()`, `json_to_db()`, `json_from_db()`, table column introspection |
| `app/services/consolidation.py` | Shared coordinator: Resource-owned chronological evidence → reusable read-only prompt preparation → committed dossier checkpoint → anchors and, for ordinary runs, weekly reflection → caller-owned final transaction. Ordinary cadence is time OR full-prompt size, reserving one average span; every eligible ordinary run consumes one whole captured set. Historical mode selects capacity-bounded import prefixes without weekly effects or ordinary clock updates; anchors retain whole-life context. Main supplies mode-aware running claims/lock/history callbacks; cache/journals finish after commit. Echo invokes the historical pipeline; its launcher workflow remains unfinished. |
| `app/services/graph_edges.py` | Edge normalization + write/invalidate helpers (`caused_by`, `evokes`, `conflicts_with`, `parallels`, `shaped_by`) |
| `app/services/activity_messages.py` | `activity_messages` scoped-SQLite table for synthetic self-DM activity recaps (`My Activities:`); accepts a caller-owned transaction for Atomic End |
| `app/services/whatsapp_outbounds.py` | `whatsapp_pending_outbounds` scoped-SQLite queue for WhatsApp replies/attachments |
| `app/services/mentra_routes.py` | Keyless private-network Mentra boundary: private-network discovery/session work, always-readable status, sitting-scoped lease lifecycle, bootstrap/token mint, non-blocking recall, transcript append/ack, and durable image snapshot/finalize. Image finalize stays unavailable until Gemini embedding config and DB profile are both active. |
| `app/services/memorize_endpoint.py` | `/memorize` core: exact supporting context saved with segment files, primary checkpoint/phase publication, sleep-gap/token chunking, progress/cancel. Historical mode uses registered import progress and original source positions, not filtered-list offsets, and enables engine-owned actual-model prompt budgets. Background inputs stay supporting context; only the rollup advances their rolling state. WhatsApp routing (`full`/`listen_only`) is independent of Mem on/off. |
| `app/services/conversation_sources.py` | Source adapters: WhatsApp from `web_source.db`; ST/Atomic/Replika from resource chat snapshots (`st_chats` / `atomic_chats` / `replika_chats`); Mentra from `openalma/mentra/transcripts`. ST snapshots preserve original message positions and the unfiltered saved count. Handles atomic writes, cursor slicing, floor backfill, and role normalization. |
| `app/services/import_routes.py` | Echo's scoped registration, unpaid current-tail validation, Process/Retry/status and bounded outer historical loop. Holds the existing claim through extraction plus one consolidation, resumes pending spans first, records the failed phase, and releases coalesced ordinary work. |
| `app/services/cross_history.py` | Shared all-chat formatting and source selection: current Channels policy or captured Retry flags; live background summary + unsummarized tail; saved supporting context for reflection; display-segment cleanup |
| `app/services/apimw.py` | APImw background memory-weaving pipeline: retrieve → synthesize prior-context/subconscious insight → persist references; shared unpaid Prior Context hydration/rendering for replies and import sizing |
| `app/services/admin_routes.py` | Health/version/shutdown/diag endpoint handlers |
| `app/services/windows_asyncio.py` | Windows plain-TCP reset cleanup using the standard Proactor teardown; subprocess transports and non-Windows event loops stay unchanged |
| `app/services/payload.py` | Shared payload/scope/normalization helpers; `_label_owner_messages` copies proven-owner lines to the Soul's canonical `user_name`, with optional live `source_owner_id` evidence and unchanged source data/IDs |
| `app/services/service_factory.py` | `MemoryService` cache + construction, llm profile merge, config readers, profile-aware semantic-dedupe defaults (per embedding profile; numeric config values are operator recovery overrides only) |
| `app/services/retrieve_orchestration.py` | Retrieve domain helpers: query/where extraction, identity-context builder, `_run_retrieve` implementation |
| `app/services/sqlite_scope.py` | Scoped db-path resolution, scope `WHERE` builder, state-db lookup/write wrappers |
| `app/services/souls.py` | Canonical per-soul SQLite identity metadata and shared local/Mentra discovery-create routes |
| `app/services/crud_endpoints.py` | CRUD endpoint logic: categories, intentions, relationships, narrative suggestion, conversation state, clear |
| `app/services/conversation_id.py` | WhatsApp group ID canonicalization to `whatsapp:group:<group@g.us>`. Called at all entrypoints — do not bypass or per-sender aliases will split state. |
| `app/services/mcp_tools.py` | MCP-facing wrapper contracts (`memu_*`) |
| `app/services/free_turn.py` | Free-turn continuation chain and `free_turn_followups` scoped-SQLite table for scheduled wakes |
| `app/services/state.py` | `write_conversation_state()` with optional caller-owned transaction, `conversation_state_from_row()`, cross-DB state search, queue management. Canonicalizes WhatsApp group IDs before all reads/writes. |
| `app/services/turn_contract.py` | `make_turn_system_prompt()`, `build_turn_prompt()`, `parse_turn_contract()`, `build_conversations_block()`, `build_turn_context_block()` — soul turn prompt construction and JSON contract parsing. Single entry point for all AI-facing chat display. |
| `app/services/intention_state.py` | Intentions normalization and prompt formatting. Owns memory cache entry caps (`MAX_MEMORY_CACHE_ENTRIES`, `MAX_MEMORY_CACHE_ENTRY_CHARS`). |
| `app/services/soul_state.py` | Soul-level singleton state: `narrative_self`, `memory_cache`, `intentions_active`; the dossier index is projected from memU, never stored |
| `app/services/soul_summaries.py` | Journaled live/previous/approved soul-summary writes and review revision guards |
| `app/services/message_log.py` | Cross-conversation history rendering: `format_merged_history()`, source-label derivation, WhatsApp normalization, activity-log block |
| `app/services/xml_utils.py` | Consolidation root extraction and scalar validation; prose token shielding/normalization reuses memU `utils/xml.py` |
| `app/services/narrative_self.py` | `snapshot_previous_narrative_self()` — writes `evolved_into` Triple before consolidation overwrites identity |
| `app/services/segment.py` | Consolidation segment helpers: `parse_segment_range()`, `build_segment_inputs()`, `create_companion_memory()` |
| `app/procedural.py` | Procedural-memory sidecar: YAML corpora → `procedural.db` with embeddings; top-k cosine lookup at retrieve time. Soul never writes here. |

## How It Connects to memu

```python
from memu.app import MemoryService    # main facade
from memu.prompts.consolidation import anchors, dossiers, weekly  # three-stage reflection prompts
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
| Historical database tools | `_archive/database-tools/` (reference only; includes taxonomy migration, embedding rebuild/bakeoff and release migration runner with their tests) |

## Config (`config.json`)

`config.example.json` is the minimal installable shape. `config.json.md` is the complete reference
for defaults, advanced settings, embedding compatibility, and privacy-sensitive diagnostics.

## Config-Only Runtime

- Runtime configuration is read from `config.json`.
- Database DSN source is `storage.metadata_store.dsn` in `config.json`.
- `run.py` and app runtime no longer use `DATABASE_URL` / `DATABASE_*` env branches.
