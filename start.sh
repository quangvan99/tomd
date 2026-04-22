#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

if [[ -z "$(docker images -q tomd:latest 2>/dev/null)" ]]; then
  echo "[start] image tomd:latest not found — building..."
  docker compose build
else
  echo "[start] image tomd:latest already built"
fi

docker compose up -d

echo "[start] server.py launched on http://localhost:9000 (logs: ./log.sh)"
