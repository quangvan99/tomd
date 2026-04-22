#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

if [[ -z "$(docker images -q tomd:latest 2>/dev/null)" ]]; then
  echo "[run] image tomd:latest not found — building..."
  docker compose build
fi

if ! docker compose ps --status running --services 2>/dev/null | grep -q '^tomd$'; then
  docker compose up -d
  until docker compose exec -T tomd true >/dev/null 2>&1; do sleep 0.5; done
fi

docker compose exec -T tomd python /app/run.py "$@"
