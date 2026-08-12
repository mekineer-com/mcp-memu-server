# mcp-memu-server

Local FastAPI server that wraps the `memu` memory engine and exposes it as an HTTP API. Handles conversation state, consolidation, the soul turn loop, and orchestration between clients and the engine.

Part of the memU local stack — private fork based on [memU v1.4.0](https://github.com/NevaMind-AI/memU/blob/v1.4.0/README.md), not affiliated with NevaMind-AI.

> **One soul, many chats.** Each `soul_id` has its own memory database. Multiple conversations can share one soul — each has its own cursor and manifest, and retrieval pulls from all of them. Consolidation is soul-scoped (weekly). Use different `soul_id` values for separate personalities.

---

## Requirements

- Python 3.12+
- `memu` engine checked out and pointed to via `config.json`
- An LLM provider API key (OpenAI-compatible)
- No Docker required

---

## Quick start

```bash
# 1. Copy the example config (config.example.json has default values)
cp config.example.json config.json

# 2. Edit config.json — at minimum:
#    llm.api_key, storage.metadata_store.dsn, memu.path

# 3. Install deps
pip install -e .

# 4. Start the server
python run.py
# or: uvicorn app.main:app --host 127.0.0.1 --port 8099
```

The server runs on `http://127.0.0.1:8099` by default.

---

## Config (`config.json`)

At minimum, set `llm.api_key`, `llm.chat_model`, `llm.embed_model`, `storage.metadata_store.dsn`, and `memu.path`. See `config.example.json` for the full reference including `step_models`, `step_temperatures`, memorize toggles, and retrieve settings. Relative paths are resolved from the `mcp-memu-server/` directory.

---

## Key endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/conversation/{id}/memorize` | POST | Extract memories from conversation (async, returns 202) |
| `/retrieve` | POST | Query memories |
| `/souls/{soul_id}/narrative_suggestion` | POST | Submit narrative_self revision suggestion |
| `/souls/{soul_id}/relationships` | CRUD | Manage declared relationships |
| `/conversation/{id}/retrieve` | POST | Retrieve + build turn prompt (RAG + prior context) |
| `/conversation/{id}/turn` | POST | Soul turn loop: run LLM, persist intentions + cache |
| `/conversation/{id}/messages/append` | POST | Append messages (with `memorize_chat` flag) |
| `/conversation/{id}/state` | GET/PATCH | Conversation working state (includes `memorize_chat`) |
| `/conversation/{id}/consolidation/force` | POST | Force consolidation now (bypass interval gate) |
| `/intentions` | GET | List active intentions |
| `/intentions/{id}` | PATCH | Update intention status/priority |
| `/categories` | GET | List all categories |
| `/clear` | POST | Delete memories in scope |
| `/config` | GET/POST | Read or update runtime config |
| `/diag/*` | GET | Diagnostic pages (recent memories, SQLite browser) |

See `INDEX.md` for the full endpoint list and task→file guide.

---

## Development

```bash
# Syntax check
python3 -m py_compile app/main.py

# Run tests
make test

# Tail logs
tail -f mcp-memu-server.log
```

---

## License

GPLv3. See `LICENSE` and `NOTICE`.
