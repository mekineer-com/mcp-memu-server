#!/usr/bin/env sh
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Missing .venv. Create it: python3 -m venv .venv" >&2
  exit 1
fi
. .venv/bin/activate
# Default port chosen to avoid clashing with SillyTavern's own server.
HOST=${MEMU_SERVER_HOST:-127.0.0.1}
PORT=${MEMU_SERVER_PORT:-8099}
exec uvicorn app.main:app --host "$HOST" --port "$PORT"
