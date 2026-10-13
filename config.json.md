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
| `context_window_tokens` | `null` | Actual model's total input/output context size. Required before paid Memorize and consolidation calls; set beside the model. Local metadata, not an API parameter. |
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

Canonical advanced model selectors. Missing entries use the base API settings
in `llm`, including `llm.chat_model`. Existing scalar model names are normalized
in memory into configured profiles under that base provider, preserving their
explicit step capacity; loading an existing file does not rewrite it.

- `preprocess`: document/media preprocessing and episode routing.
- `memory_extract`: memory extraction.
- `category_update`: dossier filing and revision.
- `reflection`: retrieval sufficiency and related reflection work.
- `consolidation`: periodic consolidation.

### Model selector sources

`llm.reply_models` contains `srmm` (SillyTavern Replies) and `other_apps`.
Reply and advanced rows use these sources:

- `{"source": "config"}`: use the base API profile, not another mapping row.
- `{"source": "srmm"}`: advanced-only inheritance of the complete SRMM API profile.
- `{"source": "st_native"}`: use the shared `llm.st_selected.resolved` snapshot.
- `{"source": "configured", "selection": {...}, "resolved": {...}}`: retain the qualified ST provider/model choice and its backend-resolved profile.

Resolved profiles use canonical `provider: "openai"`, `base_url`, `api_key`,
`chat_model`, and optional `context_window_tokens`, `max_tokens`, `temperature`,
and `endpoint_overrides`. The protocol must be OpenAI-compatible; a company or
model name does not establish compatibility. An empty API key is valid for an
explicitly keyless endpoint. Different models need their own configured capacity;
missing capacity is left unset for the existing paid-work preflight to refuse.

The plugin resolves secrets on the backend. Config responses mask nested profile
keys; saves must not echo those masked values. Changed mapping rows replace that
row only, preserving unrelated rows, base settings and embedding credentials.
Startup reloads the actual file; normal autosave is sufficient. No live file
synchronization or shutdown save is required.

Global `claude_code` hides but preserves API mappings. SillyTavern replies,
Narrative Suggestions and ST APImw synthesis remain API-backed through SRMM.
Other replies and memory operations use the global mode; only applicable API
rows are resolved when needed. Iris live voice settings remain separate.

### `llm.step_context_window_tokens`

`llm.step_context_window_tokens` uses the same five step names. A non-null named
capacity overrides the selected profile's capacity for every selector source,
including SRMM inheritance, before validation. Without an override, the selected
profile's capacity is preserved; an unrelated base model's capacity is not used.
Config-source SRMM also retains its existing `step_context_window_tokens.srmm`
capacity contribution, with the named step's override taking precedence.
Capacities must be positive integers. Input allowance is 80% of (context size
minus configured `max_tokens`),
or 80% of context if no output cap is set. Each enabled stage must fit separately.
The estimator is approximate; this is headroom, not guaranteed token accuracy.
The Launcher Memorize section links to this config file and shows estimated
Context load at the configured threshold. Actual prepared requests are checked
separately; changing the model must not reuse another model's capacity.

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
| `mental_health_query` | `true` | Include the configured mental-health retrieval angle. |

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
| `enabled` | `false` | Enable Mentra discovery and session routes. Status remains readable when disabled to protect active sittings from Stop; static earcons stay public. |
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
| `claude_code_context_window_tokens` | `null` | Actual CLI model/account context size; required before paid CLI Memorize and consolidation calls. API capacities/output caps do not apply to CLI calls. |
| `claude_code_effort` | `"medium"` | Claude Code effort level. |
| `claude_code_permission_mode` | `"default"` | Claude Code permission mode. |
| `claude_code_settings` | `""` | Optional Claude Code settings path or value. |
| `claude_code_workspace` | `"~/.cache/memu-claude-workspace"` | Neutral workspace for calls that do not use a Soul session. |
| `claude_code_timeout_seconds` | `3600` | CLI call timeout. |

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
