# Python 3.12 (Alpine) install notes

This server is configured for Python 3.12 on Alpine.

## Runtime rule

Use local source paths from `config.json`:

- `memu.path` points to the local memu source tree
- `storage.metadata_store.dsn` controls the database path
- `llm.api_key` / model fields control LLM runtime

No runtime path in this repo requires switching to a newer Python.

## Quick setup

```sh
python3.12 -m venv .venv --system-site-packages
.venv/bin/python -m pip install -U pip setuptools wheel
.venv/bin/pip install -e .
```

## Start

```sh
.venv/bin/python run.py
```
