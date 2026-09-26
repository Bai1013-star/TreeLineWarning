#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")"

if [ ! -f .env.production ]; then
    echo "Missing deployment/.env.production" >&2
    exit 1
fi

docker compose --env-file .env.production config --quiet
docker compose --env-file .env.production up -d --build --remove-orphans
docker compose --env-file .env.production ps
