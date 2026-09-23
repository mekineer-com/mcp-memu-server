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
| `temperature` | provider default | Default sampling temperature. Omit or use `null` to leave it to the provider. |
| `max_tokens` | provider default | Default maximum generated tokens. Omit or use `null` to leave it to the provider. |
| `endpoint_overrides` | `{}` | Nonstandard endpoint paths for an OpenAI-compatible provider. |
| `embed_model` | `""` | Legacy fallback used only when `llm.embedding.embed_model` is absent. Prefer the nested embedding block. |
| `client_backend` | `"httpx"` | Legacy compatibility key; the current server uses its HTTP client directly. |

### `llm.embedding`

The embedding profile may use different credentials and a different provider from chat.

| Setting | Default | Meaning |
| --- | --- | --- |
| `provider` | inherited from `llm.provider` | Embedding backend: `openai`, `gemini`, or `doubao`. |
| `api_key` | inherited from `llm.api_key` | Embedding API credential. |
| `base_url` | inherited from `llm.base_url` | Embedding API base URL. |
| `embed_model` | inherited from `llm.embed_model` | Embedding model. OpenAlma currently accepts stored profiles `text-embedding-3-large:3072` and `gemini-embedding-2:3072`. |
| `endpoint_overrides` | inherited | Nonstandard embedding endpoint path. |

Changing the embedding model does not convert an existing Soul database. Existing vectors must
be rebuilt into a new database and atomically installed; mixing embedding spaces is rejected.

### `llm.step_models`

Optional chat-model overrides. Empty strings use `llm.chat_model`.

- `preprocess`: document/media preprocessing and episode routing.
- `memory_extract`: memory extraction.
- `category_update`: dossier filing and revision.
- `reflection`: retrieval sufficiency and related reflection work.
- `consolidation`: periodic consolidation.

### `llm.step_temperatures`

Optional temperature overrides for the same five step names. Empty strings or omitted entries
use `llm.temperature` or the provider default.

## `storage`

| Setting | Default | Meaning |
| --- | --- | --- |
| `resources_dir` | user memU resources directory | Files and media associated with memories. |
| `sqlite_dir` | directory containing the configured DSN | Directory containing per-Soul SQLite databases and `owner.txt`. |

### `storage.metadata_store`

| Setting | Default | Meaning |
| --- | --- | --- |
| `provider` | `"sqlite"` | Metadata backend. Only SQLite is supported. |
| `dsn` | `"sqlite:///:memory:"` | Base SQLite DSN or filesystem path. OpenAlma derives each Soul database from `sqlite_dir`. |
| `ddl_mode` | `"create"` | memU schema mode: `create` or `validate`. |
| `embedding_profile` | inferred | Stored model and width guard, such as `gemini-embedding-2:3072`. Usually omit it; if present, it must match `llm.embedding.embed_model`. |

## `memu`, `listen`, `python`, and process settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `memu.path` | detected `~/apps/memu` or empty | Path to the memU repository root or `src` directory. |
| `listen.host` | `"127.0.0.1"` | HTTP bind address. |
| `listen.port` | `8099` | HTTP port. |
| `python.force_venv` | `false` | Re-execute through `mcp-memu-server/.venv` when starting via `run.py`. |
| `python.executable` | `""` | Reserved compatibility setting; `run.py` currently uses the running interpreter or local venv. |
| `pid_file` | `.memu-server.pid` in the server directory | PID file used for single-instance lifecycle control. |

## `memorize`

| Setting | Default | Meaning |
| --- | --- | --- |
| `min_chunk_tokens` | `8000` | Unmemorized conversation size that triggers automatic memorization. `0` disables the size gate. |
| `episodes_per_segment` | `3` | Maximum routed episodes per memorization segment. |
| `background_summary_tokens` | `1000` | Target size of the rolling background summary. |
| `background_extra_messages_tokens` | `100` | Recent-message allowance added around background summarization. |
| `enable_confidence_normalization` | `false` | Normalize extracted confidence values within a batch. |
| `semantic_dedupe_enabled` | `true` | Run post-persist semantic soft-merging. This is an operator recovery switch, not a normal user setting. |
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
`hermes-channels/data` directory.

| Setting | Meaning |
| --- | --- |
| `home` | Hermes Channels data directory. |
| `state_db_path` | Legacy Channels state and activation database. |
| `sessions_index_path` | Hermes session index. |
| `whatsapp_web_source_db` | Canonical wwebjs WhatsApp history database. |

## `mentra`

| Setting | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Allow new Iris/Mentra sessions. Existing status and cleanup paths remain available. |
| `public_base_url` | `""` | Public OpenAlma URL used by launcher/distribution integration. |
| `gemini_api_key` | `""` | Gemini Live credential used to mint constrained session tokens. |
| `model` | current Gemini native-audio preview model | Gemini Live model. |
| `voice` | `"Kore"` | Gemini Live output voice. |
| `integration_bearer_token` | `""` | Shared bearer token for Iris integration endpoints. |
| `session_warning_seconds` | `0` | Client warning lead time before a session limit; `0` disables the warning. |

## Claude Code backend

These are top-level keys rather than a nested object.

| Setting | Default | Meaning |
| --- | --- | --- |
| `claude_code` | `false` | Use Claude Code CLI instead of the HTTP chat backend. |
| `claude_code_model` | `"claude-opus-4-7"` | Claude Code model. |
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
