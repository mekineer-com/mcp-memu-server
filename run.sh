#!/usr/bin/env sh
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Missing .venv. Create it: python3 -m venv .venv" >&2
  exit 1
fi
.venv/bin/python -u run.py
