# mcp-memu-server

Local FastAPI server that wraps the `memu` memory engine and exposes it as an HTTP API. Handles conversation state, consolidation, the soul turn loop, and orchestration between clients and the engine.

This is part of the memU local stack — a private fork, not affiliated with NevaMind-AI.

> **Single conversation per soul.** Each soul should have exactly one conversation. Consolidation (the weekly self-review pipeline) is soul-scoped and tracks state per conversation; running multiple conversations under the same soul will produce divergent weekly reviews. Multi-conversation support is tracked as a "maybe" item in the roadmap.

---

## Requirements

- Python 3.12+
- `memu` engine checked out and pointed to via `config.json`
- An LLM provider API key (OpenAI-compatible)
- No Docker required

---

## Quick start

```bash
# 1. Copy the example config
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

```json
{
  "llm": {
    "provider": "openai",
    "api_key": "sk-...",
    "base_url": "https://api.openai.com/v1",
    "chat_model": "gpt-4o",
    "embed_model": "text-embedding-3-large"
  },
  "storage": {
    "metadata_store": { "provider": "sqlite", "dsn": "../memu/sqlite/memu.db" }
  },
  "listen": { "host": "127.0.0.1", "port": 8099 },
  "memu": { "path": "../memu/src" },
  "categories": {
    "defaults": [
      { "name": "Identity", "description": "..." },
      { "name": "Preferences", "description": "..." },
      { "name": "Relationships", "description": "..." },
      { "name": "Experiences", "description": "..." }
    ]
  },
  "retrieve": { "method": "rag" },
  "consolidation_interval_days": 7
}
```

See `config.example.json` for the full reference. Relative paths in config are resolved from the `mcp-memu-server/` directory.

---

## Key endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/memorize` | POST | Extract memories from conversation (async, returns 202) |
| `/retrieve` | POST | Query memories |
| `/conversation/{id}/retrieve` | POST | Retrieve + build turn prompt (RAG + prior context) |
| `/conversation/{id}/turn` | POST | Soul turn loop: run LLM, persist intentions + cache |
| `/conversation/{id}/state` | GET/PATCH | Conversation working state |
| `/conversation/{id}/consolidation/force` | POST | Force consolidation now (bypass interval gate) |
| `/intentions` | GET | List active intentions |
| `/intentions/{id}` | PATCH | Update intention status/priority |
| `/categories` | GET | List all categories |
| `/clear` | POST | Delete memories in scope |
| `/config` | GET/POST | Read or update runtime config |
| `/diag/*` | GET | Diagnostic pages (recent memories, SQLite browser) |

See `INDEX.md` for the full endpoint list and task→file guide.

---

## Codebase orientation

```
mcp-memu-server/
├── app/main.py              # All endpoints + business logic
├── app/db.py                # SQLite helpers
├── app/database.py          # SQLAlchemy async engine
├── app/services/consolidation.py # Consolidation pipeline
├── app/services/diary.py    # Diary helper primitives
├── app/services/state.py    # Conversation state management
├── app/services/turn_contract.py   # Soul turn prompt construction
├── app/services/intention_state.py # Intentions normalization
├── run.py                   # Entry point
└── config.json              # Runtime config (not committed)
```

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
