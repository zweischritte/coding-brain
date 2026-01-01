#!/usr/bin/env bash
set -euo pipefail

# Full-stack backup for the Coding Brain docker-compose stack.
# Usage: ./backup-coding-brain.sh [backup_name]

BACKUP_DIR="${HOME}/coding-brain-backups"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_NAME="${1:-backup_${TIMESTAMP}}"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${COMPOSE_DIR}/docker-compose.yml"
COMPOSE_PROJECT="${COMPOSE_PROJECT:-codingbrain}"

TAR_IMAGE="${TAR_IMAGE:-alpine:3.20}"

BASE_VOLUMES=(
  codingbrain_postgres_data
  codingbrain_valkey_data
  codingbrain_qdrant_storage
  codingbrain_neo4j_data
  codingbrain_neo4j_logs
  codingbrain_opensearch_data
)

CONTAINERS=(
  codingbrain-postgres
  codingbrain-valkey
  codingbrain-qdrant
  codingbrain-opensearch
  codingbrain-neo4j
  codingbrain-mcp
  codingbrain-indexing-worker
  codingbrain-ui
  codingbrain-docs
)

IMAGES=(
  pgvector/pgvector:pg16
  valkey/valkey:8.0.2-alpine
  qdrant/qdrant:v1.12.5
  opensearchproject/opensearch:2.13.0
  neo4j:5.26.4-community
  codingbrain/mcp
  codingbrain/ui:latest
  python:3.12-alpine
)

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: Missing required command: $1"
    exit 1
  fi
}

json_array() {
  local items=("$@")
  if [ "${#items[@]}" -eq 0 ]; then
    printf '[]'
    return
  fi
  local joined
  joined=$(printf '"%s",' "${items[@]}")
  printf '[%s]' "${joined%,}"
}

is_container_running() {
  docker ps --format '{{.Names}}' | grep -qx "$1"
}

backup_volume() {
  local base="$1"
  local volume="$2"
  local dest="$3"
  if ! docker volume inspect "${volume}" >/dev/null 2>&1; then
    echo "   ! volume not found for ${base}: ${volume} (skipping)"
    return
  fi
  echo "   - ${base} (${volume}) -> ${dest}"
  docker run --rm \
    -v "${volume}":/source:ro \
    -v "${BACKUP_PATH}/volumes":/backup \
    "${TAR_IMAGE}" tar czf "/backup/${dest}" -C /source .
  local tar_path="${BACKUP_PATH}/volumes/${dest}"
  if [ -f "${tar_path}" ]; then
    local entries
    entries=$(tar -tzf "${tar_path}" | head -n 2 | wc -l | tr -d ' ')
    if [ "${entries}" -le 1 ]; then
      echo "     ! ${dest} appears empty"
    fi
  fi
}

mkdir -p "${BACKUP_PATH}/docker/inspect/containers"
mkdir -p "${BACKUP_PATH}/docker/inspect/images"
mkdir -p "${BACKUP_PATH}/volumes"

echo "=== Coding Brain Backup ==="
echo "Creating backup: ${BACKUP_NAME}"
echo "Location: ${BACKUP_PATH}"
echo "Compose file: ${COMPOSE_FILE}"
echo ""

require_cmd docker

VOLUME_SPECS=()
VOLUME_ACTUALS=()
for base in "${BASE_VOLUMES[@]}"; do
  candidate="${COMPOSE_PROJECT}_${base}"
  if docker volume inspect "${candidate}" >/dev/null 2>&1; then
    actual="${candidate}"
  elif docker volume inspect "${base}" >/dev/null 2>&1; then
    actual="${base}"
  else
    actual="$(docker volume ls --format '{{.Name}}' \
      --filter "label=com.docker.compose.project=${COMPOSE_PROJECT}" \
      --filter "label=com.docker.compose.volume=${base}" | head -n 1)"
  fi
  if [ -n "${actual}" ]; then
    VOLUME_SPECS+=("${base}|${actual}")
    VOLUME_ACTUALS+=("${actual}")
  else
    echo "   ! volume not found for ${base}"
  fi
done

echo "[1/8] PostgreSQL logical dump (pg_dump)"
if is_container_running "codingbrain-postgres"; then
  docker exec codingbrain-postgres sh -lc 'pg_dump -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-codingbrain}" -Fc -f /tmp/postgres.dump'
  docker cp codingbrain-postgres:/tmp/postgres.dump "${BACKUP_PATH}/postgres.dump"
  docker exec codingbrain-postgres rm -f /tmp/postgres.dump
  echo "   ✓ postgres.dump"
else
  echo "   ! postgres container not running; skipping pg_dump"
fi

echo "[2/8] Valkey snapshot (RDB/AOF)"
if is_container_running "codingbrain-valkey"; then
  docker exec codingbrain-valkey sh -lc 'valkey-cli save'
  if docker exec codingbrain-valkey sh -lc 'test -f /data/dump.rdb'; then
    docker cp codingbrain-valkey:/data/dump.rdb "${BACKUP_PATH}/valkey_dump.rdb"
    echo "   ✓ valkey_dump.rdb"
  else
    echo "   ! dump.rdb not found after save"
  fi
  if docker exec codingbrain-valkey sh -lc 'test -f /data/appendonly.aof'; then
    docker cp codingbrain-valkey:/data/appendonly.aof "${BACKUP_PATH}/valkey_appendonly.aof"
    echo "   ✓ valkey_appendonly.aof"
  fi
else
  echo "   ! valkey container not running; skipping RDB/AOF copy"
fi

echo "[3/8] Docker volume backups"
for spec in "${VOLUME_SPECS[@]}"; do
  base="${spec%%|*}"
  actual="${spec##*|}"
  backup_volume "${base}" "${actual}" "${base}.tar.gz"
done
echo "   ✓ volumes backed up"

echo "[4/8] Container and image metadata"
docker ps -a > "${BACKUP_PATH}/docker/ps.txt"
docker images > "${BACKUP_PATH}/docker/images.txt"
if docker compose -f "${COMPOSE_FILE}" config >/dev/null 2>&1; then
  docker compose -f "${COMPOSE_FILE}" config > "${BACKUP_PATH}/docker/compose.resolved.yml"
fi
for container in "${CONTAINERS[@]}"; do
  if docker inspect "${container}" >/dev/null 2>&1; then
    docker inspect "${container}" > "${BACKUP_PATH}/docker/inspect/containers/${container}.json"
  fi
done
for spec in "${VOLUME_SPECS[@]}"; do
  base="${spec%%|*}"
  actual="${spec##*|}"
  if docker volume inspect "${actual}" >/dev/null 2>&1; then
    docker volume inspect "${actual}" > "${BACKUP_PATH}/docker/inspect/${base}.json"
  fi
done
for image in "${IMAGES[@]}"; do
  if docker image inspect "${image}" >/dev/null 2>&1; then
    docker image inspect "${image}" > "${BACKUP_PATH}/docker/inspect/images/$(echo "${image}" | tr '/:' '__').json"
  fi
done
echo "   ✓ docker metadata captured"

echo "[5/8] Docker image archive"
IMAGES_FOUND=()
for image in "${IMAGES[@]}"; do
  if docker image inspect "${image}" >/dev/null 2>&1; then
    IMAGES_FOUND+=("${image}")
  fi
done
if [ "${#IMAGES_FOUND[@]}" -gt 0 ]; then
  docker save "${IMAGES_FOUND[@]}" -o "${BACKUP_PATH}/docker-images.tar"
  echo "   ✓ docker-images.tar"
else
  echo "   ! no images found to save"
fi

echo "[6/8] Repository snapshot"
tar czf "${BACKUP_PATH}/coding-brain-repo.tar.gz" --exclude .git -C "${REPO_ROOT}" .
echo "   ✓ coding-brain-repo.tar.gz"

echo "[7/8] Git metadata"
if command -v git >/dev/null 2>&1; then
  git -C "${REPO_ROOT}" rev-parse HEAD > "${BACKUP_PATH}/git_head.txt" 2>/dev/null || true
  git -C "${REPO_ROOT}" status --porcelain > "${BACKUP_PATH}/git_status.txt" 2>/dev/null || true
fi

echo "[8/8] Backup metadata"
cat > "${BACKUP_PATH}/backup_info.json" <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "backup_name": "${BACKUP_NAME}",
  "compose_project": "${COMPOSE_PROJECT}",
  "compose_file": "${COMPOSE_FILE}",
  "repo_root": "${REPO_ROOT}",
  "volumes": $(json_array "${BASE_VOLUMES[@]}"),
  "volumes_actual": $(json_array "${VOLUME_ACTUALS[@]}"),
  "containers": $(json_array "${CONTAINERS[@]}")
}
EOF

TOTAL_SIZE=$(du -sh "${BACKUP_PATH}" | cut -f1)
echo ""
echo "=== Backup Complete ==="
echo "Location: ${BACKUP_PATH}"
echo "Total:    ${TOTAL_SIZE}"
echo ""
