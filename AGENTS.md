# MCP memU Server

Keep README.md for people choosing, installing and using this project. Put durable
working guidance here, code navigation in an existing index, and task history in
the relevant plan or audit; do not turn this file into a session log.

Read [INDEX.md](INDEX.md) for code and endpoint navigation. Before changing
Memorize, follow [MEMORIZE_PIPELINE.md](MEMORIZE_PIPELINE.md) across reads,
writes and recovery. MCP owns client-independent orchestration and conversation
state; memU owns the memory engine.

## Runtime Dependencies

`requirements.lock` pins the joint server/memU runtime for the first coordinated
beta and subsequent installation, update and recovery. Regenerate from this
directory and verify changed packages before shipping:

```sh
uv pip compile --universal --python-version 3.12 pyproject.toml ../memu/pyproject.toml -o requirements.lock
```

`uv.lock` is separate development tooling, not the installer runtime lock.
Runtime pins do not remove obsolete installed packages or pin build tools.
