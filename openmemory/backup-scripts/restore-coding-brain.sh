#!/usr/bin/env bash
set -euo pipefail

# Restore script for the Coding Brain docker-compose stack.
# Usage: ./restore-coding-brain.sh <backup_name>

BACKUP_DIR="${HOME}/coding-brain-backups"
BACKUP_NAME="${1:-}"

if [ -z "${BACKUP_NAME}" ]; then
  echo "Usage: ./restore-coding-brain.sh <backup_name>"
  echo ""
  echo "Available backups:"
  ls -1 "${BACKUP_DIR}" 2>/dev/null || echo "  (none found in ${BACKUP_DIR})"
  exit 1
fi

BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

if [ ! -d "${BACKUP_PATH}" ]; then
  echo "Error: Backup not found at ${BACKUP_PATH}"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${COMPOSE_DIR}/docker-compose.yml"
COMPOSE_PROJECT="${COMPOSE_PROJECT:-codingbrain}"
COMPOSE_ARGS=(--project-directory "${COMPOSE_DIR}" -f "${COMPOSE_FILE}")

TAR_IMAGE="${TAR_IMAGE:-alpine:3.20}"

BASE_VOLUMES=(
  codingbrain_postgres_data
  codingbrain_valkey_data
  codingbrain_qdrant_storage
  codingbrain_neo4j_data
  codingbrain_neo4j_logs
  codingbrain_opensearch_data
)

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: Missing required command: $1"
    exit 1
  fi
}

resolve_volume_name() {
  local base="$1"
  local candidate="${COMPOSE_PROJECT}_${base}"
  if docker volume inspect "${candidate}" >/dev/null 2>&1; then
    echo "${candidate}"
    return
  fi
  if docker volume inspect "${base}" >/dev/null 2>&1; then
    echo "${base}"
    return
  fi
  docker volume ls --format '{{.Name}}' \
    --filter "label=com.docker.compose.project=${COMPOSE_PROJECT}" \
    --filter "label=com.docker.compose.volume=${base}" | head -n 1
}

resolve_volume_tar() {
  local base="$1"
  local actual="$2"
  local base_tar="${BACKUP_PATH}/volumes/${base}.tar.gz"
  local actual_tar="${BACKUP_PATH}/volumes/${actual}.tar.gz"
  if [ -f "${base_tar}" ]; then
    echo "${base_tar}"
    return
  fi
  if [ -f "${actual_tar}" ]; then
    echo "${actual_tar}"
    return
  fi
}

restore_volume() {
  local base="$1"
  local actual="$2"
  local tar_path
  tar_path="$(resolve_volume_tar "${base}" "${actual}")"
  if [ -z "${tar_path}" ]; then
    echo "   ! missing tar for ${base} (${actual})"
    return
  fi

  if ! docker volume inspect "${actual}" >/dev/null 2>&1; then
    echo "   ! volume not found for ${base}: ${actual} (skipping)"
    return
  fi

  local entries
  entries=$(tar -tzf "${tar_path}" | head -n 2 | wc -l | tr -d ' ')
  if [ "${entries}" -le 1 ]; then
    echo "   ! ${tar_path} appears empty (skipping restore)"
    return
  fi

  echo "   - ${base} -> ${actual}"
  docker run --rm \
    -v "${actual}":/target \
    -v "$(dirname "${tar_path}")":/backup:ro \
    "${TAR_IMAGE}" sh -c "find /target -mindepth 1 -delete && tar xzf \"/backup/$(basename "${tar_path}")\" -C /target"
}

echo "=== Coding Brain Restore ==="
echo "Restoring from: ${BACKUP_NAME}"
echo ""
echo "WARNING: This will overwrite existing Docker volumes and data."
echo "Press Ctrl+C to cancel, or Enter to continue..."
read -r

require_cmd docker

if docker ps --format '{{.Names}}' | grep -q '^codingbrain-'; then
  echo "Stopping services..."
  docker compose "${COMPOSE_ARGS[@]}" down
  sleep 3
fi

echo "[1/6] Docker image restore"
if [ -f "${BACKUP_PATH}/docker-images.tar" ]; then
  docker load -i "${BACKUP_PATH}/docker-images.tar"
  echo "   ✓ docker-images.tar loaded"
elif [ -d "${BACKUP_PATH}/docker-images" ]; then
  tmp_dir="$(mktemp -d -t docker-images.XXXXXX)"
  tmp_tar="${tmp_dir}/docker-images.tar"
  tar -cf "${tmp_tar}" -C "${BACKUP_PATH}/docker-images" .
  docker load -i "${tmp_tar}"
  rm -rf "${tmp_dir}"
  echo "   ✓ docker-images/ loaded"
else
  echo "   ! docker-images.tar not found (skipping)"
fi

echo "[2/6] Volume restore"
for base in "${BASE_VOLUMES[@]}"; do
  actual="$(resolve_volume_name "${base}")"
  if [ -z "${actual}" ]; then
    echo "   ! volume not found for ${base}"
    continue
  fi
  restore_volume "${base}" "${actual}"
done

echo "[3/6] PostgreSQL logical restore"
POSTGRES_DUMP="${BACKUP_PATH}/postgres.dump"
POSTGRES_RESTORED=false
if [ -f "${POSTGRES_DUMP}" ]; then
  docker compose "${COMPOSE_ARGS[@]}" up -d postgres

  for i in $(seq 1 30); do
    if docker exec codingbrain-postgres sh -lc 'pg_isready -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-codingbrain}"' >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done

  docker cp "${POSTGRES_DUMP}" codingbrain-postgres:/tmp/postgres.dump
  docker exec codingbrain-postgres sh -lc 'PGPASSWORD="${POSTGRES_PASSWORD:-}" pg_restore --clean --if-exists --no-owner --no-privileges -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-codingbrain}" /tmp/postgres.dump'
  docker exec codingbrain-postgres rm -f /tmp/postgres.dump
  POSTGRES_RESTORED=true
  echo "   ✓ postgres.dump restored"
else
  echo "   ! postgres.dump not found (skipping)"
fi

if [ "${POSTGRES_RESTORED}" = "true" ]; then
  docker compose "${COMPOSE_ARGS[@]}" down
fi

echo "[4/6] Valkey fallback restore"
VALKEY_DUMP="${BACKUP_PATH}/valkey_dump.rdb"
VALKEY_AOF="${BACKUP_PATH}/valkey_appendonly.aof"
VALKEY_VOLUME_TAR_BASE="${BACKUP_PATH}/volumes/codingbrain_valkey_data.tar.gz"
VALKEY_VOLUME_TAR_ACTUAL="${BACKUP_PATH}/volumes/${COMPOSE_PROJECT}_codingbrain_valkey_data.tar.gz"
VALKEY_VOLUME_NAME="$(resolve_volume_name codingbrain_valkey_data)"
if [ ! -f "${VALKEY_VOLUME_TAR_BASE}" ] && [ ! -f "${VALKEY_VOLUME_TAR_ACTUAL}" ]; then
  if [ -n "${VALKEY_VOLUME_NAME}" ] && docker volume inspect "${VALKEY_VOLUME_NAME}" >/dev/null 2>&1; then
    if [ -f "${VALKEY_DUMP}" ]; then
      docker run --rm -v "${VALKEY_VOLUME_NAME}":/target -v "${BACKUP_PATH}":/backup:ro "${TAR_IMAGE}" \
        sh -c "cp /backup/valkey_dump.rdb /target/dump.rdb"
      echo "   ✓ valkey_dump.rdb restored"
    fi
    if [ -f "${VALKEY_AOF}" ]; then
      docker run --rm -v "${VALKEY_VOLUME_NAME}":/target -v "${BACKUP_PATH}":/backup:ro "${TAR_IMAGE}" \
        sh -c "cp /backup/valkey_appendonly.aof /target/appendonly.aof"
      echo "   ✓ valkey_appendonly.aof restored"
    fi
  else
    echo "   ! valkey volume not found; skipping fallback copy"
  fi
else
  echo "   ! valkey volume tar present; skipping fallback copy"
fi

echo "[5/6] Repository restore (optional)"
if [ "${RESTORE_REPO:-0}" = "1" ]; then
  REPO_TARGET="${RESTORE_REPO_DIR:-${REPO_ROOT}}"
  mkdir -p "${REPO_TARGET}"
  if [ -f "${BACKUP_PATH}/coding-brain-repo.tar.gz" ]; then
    tar xzf "${BACKUP_PATH}/coding-brain-repo.tar.gz" -C "${REPO_TARGET}"
    echo "   ✓ repo restored to ${REPO_TARGET}"
  else
    echo "   ! coding-brain-repo.tar.gz not found (skipping)"
  fi
else
  echo "   ! set RESTORE_REPO=1 to restore the repo snapshot"
fi

echo "[6/6] Restore complete"
echo ""
echo "Start services with: docker compose -f ${COMPOSE_FILE} up -d"
echo ""
