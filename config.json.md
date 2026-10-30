# `config.json` reference

`config.json` is ordinary JSON, so it cannot contain comments. This file documents every
server setting. `config.example.json` contains only the settings a typical installation needs;
all omitted settings use the defaults described here.

Relative paths are resolved from the `mcp-memu-server` directory. Restart the server after
editing settings unless an endpoint explicitly reloads them.

## `llm`

The default chat-generation profile.

| Setting | Default | Meaning |
| --- | --- | --- |
| `provider` | `"openai"` | Backend adapter. The default adapter also supports OpenAI-compatible APIs. |
| `api_key` | `""` | API credential for chat generation. |
| `base_url` | `"https://api.openai.com/v1"` | Provider API base URL. |
| `chat_model` | `""` | Chat model used unless a step-specific model overrides it. Required for normal operation. |
| `context_window_tokens` | `null` | Conservative usable input/output context allowance for this provider/model. Required before paid Memorize and consolidation calls, and for integrated Atomic API chat (must exceed 20,000); set beside the model. Local metadata, not an API parameter. |
| `temperature` | provider default | Default sampling temperature. Omit or use `null` to leave it to the provider. |
| `max_tokens` | provider default | Default maximum generated tokens. Omit or use `null` to leave it to the provider. |
| `endpoint_overrides` | `{}` | Nonstandard endpoint paths for an OpenAI-compatible provider. |
| `client_backend` | `"httpx"` | Legacy compatibility key; the current server uses its HTTP client directly. |

### `llm.embedding`

OpenAlma owns the embedding provider, model, and database profile. The launcher shows the
currently supported model; changing models requires a managed per-Soul rebuild.

| Setting | Default | Meaning |
| --- | --- | --- |
| `api_key` | `""` | Gemini embedding API credential. |
| `base_url` | Gemini API | Optional compatible endpoint override. |
| `endpoint_overrides` | `{}` | Optional embedding endpoint path override. |

Changing the embedding model does not convert an existing Soul database. Existing vectors must
be rebuilt into a new database and atomically installed; mixing embedding spaces is rejected.

### `llm.step_models`

Optional memory-step settings. Missing entries use the base API connection in
`llm`. Each entry is either a model name on that connection or a complete
independent connection object using the same fields as the base: `provider`,
`base_url`, `api_key`, `chat_model`, and optional `context_window_tokens`,
`max_tokens`, `temperature` and `endpoint_overrides`. Only OpenAI-compatible
HTTP profiles (`provider: "openai"`) are supported. An independent connection
must supply its own key, including an explicit empty string for a keyless endpoint.

- `preprocess`: document/media preprocessing and episode routing.
- `memory_extract`: memory extraction.
- `category_update`: dossier filing and revision.
- `reflection`: retrieval sufficiency and related reflection work.
- `consolidation`: periodic consolidation.

For example, a different model on the base connection needs only
`"memory_extract": "example-model"`; it does not copy the base URL or key.
A model different from the base needs its own configured context-window size.
An independent step connection inherits base capacity only when its model and
URL match the base. Keys are never inherited into independent connections.

ST is one editor of this shared file. Loading its extension reads hand edits
without writing remembered choices over them. Advanced Mapping copies selected
ST connections into these step entries; they work with ST closed. Disabling it
asks for confirmation, removes active model overrides and their matching
per-step capacities, and retains profile/model choices in ST for re-enabling.
While ST is open, active entries follow their selected SRMM, ST-selected or
named-profile source. ST stores only source markers; config.json keeps the
resolved values for use with ST closed. Selecting config.json keeps those values
and follows hand edits without warnings.

SillyTavern Replies (SRMM) lives entirely in ST and accompanies its reply,
Narrative Suggestion and APImw synthesis requests. It is not stored in this file.
Other OpenAlma App Replies edits the base API connection. Global `claude_code`
selects Claude Code for server replies/memory work while SRMM remains API-backed;
hiding advanced controls in CLI mode does not disable or erase their settings.
Iris live voice settings remain separate.

Partial config saves preserve unrelated settings. A `null` step entry in a
partial save deletes that entry; empty mapping sections are removed rather than
saved as placeholders. Nested keys are masked in config responses; do not save
masked values. The ST plugin resolves profile secrets on the backend.

### `llm.step_context_window_tokens`

Optional positive-integer capacities keyed by the same five step names. A named
capacity overrides the step connection's capacity. No capacity is guessed for
a different model. Input allowance is 80% of (context size minus configured
`max_tokens`), or 80% of context if no output cap is set. Each enabled stage
must fit separately; estimates provide headroom, not guaranteed token accuracy.
The Launcher links to this config file and shows estimated Context load.

### `llm.step_temperatures`

Optional temperature overrides for the same five step names. Empty strings or omitted entries
use `llm.temperature` or the provider default.

## `storage`

| Setting | Default | Meaning |
| --- | --- | --- |
| `resources_dir` | sibling `memu/resources` directory | Files and media associated with memories. |
| `sqlite_dir` | directory containing the configured DSN | Directory containing per-Soul SQLite databases and `owner.txt`. |

### `storage.metadata_store`

| Setting | Default | Meaning |
| --- | --- | --- |
| `provider` | `"sqlite"` | Metadata backend. Only SQLite is supported. |
| `dsn` | `"sqlite:///:memory:"` | Base SQLite DSN or filesystem path. OpenAlma derives each Soul database from `sqlite_dir`. |
| `ddl_mode` | `"create"` | memU schema mode: `create` or `validate`. |
The stored embedding profile is managed by OpenAlma and is not a user setting.

## `memu`, `listen`, `python`, and process settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `memu.path` | sibling `memu/src` if present, otherwise empty | Path to the memU package's import directory. |
| `listen.host` | `"127.0.0.1"` | HTTP bind address. |
| `listen.port` | `8099` | HTTP port. |
| `python.force_venv` | `false` | Re-execute through `mcp-memu-server/.venv` when starting via `run.py`. |
| `python.executable` | `""` | Reserved compatibility setting; `run.py` currently uses the running interpreter or local venv. |
| `pid_file` | `.memu-server.pid` in the server directory | PID file used for single-instance lifecycle control. |

## `memorize`

| Setting | Default | Meaning |
| --- | --- | --- |
| `min_chunk_tokens` | `8000` | Unmemorized conversation size that triggers automatic memorization. Lowering it lowers the Launcher's projected Context load at 100%, not an existing backlog. Sleep-gap batches can exceed it. `0` disables the size gate. |
| `background_summary_tokens` | `1000` | Target size of the rolling background summary. |
| `background_extra_messages_tokens` | `100` | Per non-Memorize chat: shorter tails stay raw; tails at or above this size get separate summaries in one batched call. Existing rolling summaries stay verbatim. |
| `enable_confidence_normalization` | `false` | Normalize extracted confidence values within a batch. |
| `semantic_dedupe_enabled` | `true` | Run post-persist semantic soft-merging. This is an operator recovery switch for either supported embedding profile; it does not make an unknown profile valid. |
| `semantic_dedupe_similarity_threshold` | `"default"` | Use the active embedding profile's internal threshold: `0.89` for `text-embedding-3-large:3072`, `0.90` for `gemini-embedding-2:3072`. A numeric operator override is accepted for recovery or recalibration. |

Semantic dedupe writes `merged_into` on the redundant memory; it does not hard-delete the row.
The Gemini default is a bootstrap from 11 reviewed historical duplicate pairs, not a
completed calibration on new Gemini-era `memorize()` output; no such output existed when it was set.

## `categories`

| Setting | Default | Meaning |
| --- | --- | --- |
| `dynamic_category_cluster_size` | `10` | Minimum unresolved proposal cluster used when considering a new dossier. |
| `category_summary_target_words` | `300` | Target maximum length for generated dossier summaries. |

## `retrieve`

| Setting | Default | Meaning |
| --- | --- | --- |
| `apimw_enabled` | `true` | Enable the periodic subconscious/APImw pass. |
| `apimw_cadence` | `5` | Eligible turn cadence for APImw. |
| `apimw_memory_count` | `20` | Number of retrieved memories offered to APImw. |
| `apimw_random_count` | `5` | Additional random memories offered to APImw. |
| `mental_health_query` | `false` | Shared Mental Health Addon setting for ST and WhatsApp. Include the configured mental-health retrieval angle when enabled. |

The engine currently ranks memory items with fixed weights `0.5` similarity, `0.3` recency,
and `0.2` importance. `recency_decay_days` and lower-level retrieval settings are memU request
settings, not top-level `config.json` settings today.

## `hermes`

Paths used to read cross-channel history. Defaults point to the sibling
`hermes-channels/data` directory using config-directory-relative paths. All
relative values below resolve beside `config.json`, not the process working
directory. Absolute and `~` paths remain explicit overrides. Existing absolute
configurations are not guessed or rebased when an installation moves; update
those deliberately if they should follow the move.

| Setting | Meaning |
| --- | --- |
| `home` | Hermes Channels data directory. |
| `state_db_path` | Legacy Channels state and activation database. |
| `sessions_index_path` | Hermes session index. |
| `whatsapp_web_source_db` | Canonical wwebjs WhatsApp history database. |

## `mentra`

Iris requires a trusted private connection, not a public endpoint. Devices allowed
to reach this endpoint can access the Soul's Iris data; trust the permitted devices
and network access rules. Keep MCP on loopback behind the configured narrow Mentra
ingress. Gemini credentials and constrained provider session tokens remain required.

| Setting | Default | Meaning |
| --- | --- | --- |
| `public_base_url` | `""` | OpenAlma URL reachable over the trusted private connection, used by launcher/distribution integration. |
| `gemini_api_key` | `""` | Gemini Live credential used to mint constrained session tokens. |
| `model` | current Gemini native-audio preview model | Gemini Live model. |
| `voice` | `"Kore"` | Gemini Live output voice. |
| `session_warning_seconds` | `0` | Client warning lead time before a session limit; `0` disables the warning. |

## Claude Code backend

These are top-level keys rather than a nested object.

| Setting | Default | Meaning |
| --- | --- | --- |
| `claude_code` | `false` | Use Claude Code CLI instead of the HTTP chat backend. |
| `claude_code_model` | `"claude-opus-4-7"` | Claude Code model. |
| `claude_code_context_window_tokens` | `null` | Actual CLI model/account context size; required before paid CLI Memorize and consolidation calls, and must be above 20000 for Atomic Claude Code chat. API capacities/output caps do not apply to CLI calls. |
| `claude_code_effort` | `"medium"` | Claude Code effort level. |
| `claude_code_permission_mode` | `"default"` | Claude Code permission mode. |
| `claude_code_settings` | `""` | Optional Claude Code settings path or value. |
| `claude_code_workspace` | `"~/.cache/memu-claude-workspace"` | Neutral workspace for calls that do not use a Soul session; Atomic Claude Code chat keeps one workspace per Soul under `atomic/<hash>` inside it. |
| `claude_code_timeout_seconds` | `3600` | CLI call timeout. |

Atomic Claude Code chat requires Claude Code installed and signed in under the
account running the memory server, on Linux, macOS or native Windows. It also needs
Atomic's Claude session helper, which OpenAlma setup installs. A missing or broken
helper reports `admission_error` and refuses new native work; install or repair it.
There is no silent API fallback.

On Windows with long-path support disabled, a long user-profile path can prevent
native cleanup from saving its backup. Before starting new sittings, set
`claude_code_workspace` to a short writable path such as `C:\\OpenAlma-workspace`.
This workaround passed with a 21-character username; arbitrary longer/custom
paths are not certified. For existing sittings, enable Windows long-path support
and restart the memory server instead: changing the workspace does not relocate
their bound native sessions.

Attachment sending requires an explicit, nonblank `claude_code_workspace` in
server configuration. Turn prompts and attachment validation use that configured
directory; only readable files resolving inside it may be attached. The neutral
backend workspace default above does not grant attachment access. When the key
is omitted or blank, turns cannot send file attachments, even if the backend
uses its default working directory.

## Other services

| Setting | Default | Meaning |
| --- | --- | --- |
| `mcp.http_path` | `"/mcp"` | Streamable HTTP MCP mount and diagnostic prefix. |
| `mcp.sse_path` | `"/sse"` | Legacy SSE MCP mount. |
| `procedural.yaml_dir` | `"../memu/procedural"` | Procedural-memory YAML directory. |
| `procedural.db_path` | `"../memu/sqlite/procedural.db"` | Rebuildable procedural-memory cache. |
| `consolidation_interval_days` | `7` | Days between successful consolidations; the next memorize processes pending conversations when due. |
| `turn_response_sentences` | `3` | Target Soul response length in sentences. |
| `debug.log_prompts` | `false` | Log full model prompts. These logs can contain private conversation and memory content. |
