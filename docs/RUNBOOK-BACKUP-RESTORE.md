# Backup and Restore Runbook

**Document Version**: 1.0
**Last Updated**: 2025-12-28
**Applies to**: coding-brain production deployments

---

## Table of Contents

1. [Overview](#overview)
2. [Backup Strategy](#backup-strategy)
3. [PostgreSQL Backup & Restore](#postgresql-backup--restore)
4. [Neo4j Backup & Restore](#neo4j-backup--restore)
5. [Qdrant Backup & Restore](#qdrant-backup--restore)
6. [OpenSearch Backup & Restore](#opensearch-backup--restore)
7. [Valkey Backup & Restore](#valkey-backup--restore)
8. [Full System Backup](#full-system-backup)
9. [Disaster Recovery Procedures](#disaster-recovery-procedures)
10. [Verification Checklist](#verification-checklist)
11. [Troubleshooting](#troubleshooting)

---

## Overview

This runbook documents backup and restore procedures for all data stores in the coding-brain system:

| Service | Data Store Type | Volume | Backup Method | Priority |
|---------|----------------|--------|---------------|----------|
| PostgreSQL 16 | Relational + Vector | `codingbrain_postgres_data` | pg_dump (custom format) | Critical |
| Neo4j 5.26 | Graph | `codingbrain_neo4j_data` | neo4j-admin dump | Critical |
| Qdrant 1.12 | Vector | `codingbrain_qdrant_storage` | Snapshot API | High |
| OpenSearch 2.13 | Search Index | `codingbrain_opensearch_data` | Snapshot Repository | High |
| Valkey 8.0 | Cache/Session | `codingbrain_valkey_data` | RDB + AOF | Medium |

**Priority Definitions**:
- **Critical**: Data loss is unacceptable; restore within 1 hour RTO
- **High**: Significant impact; restore within 4 hours RTO
- **Medium**: Rebuild possible from other sources; restore within 24 hours RTO

---

## Backup Strategy

### Recommended Schedule

| Service | Frequency | Retention | Storage Location |
|---------|-----------|-----------|------------------|
| PostgreSQL | Every 6 hours | 7 days (28 backups) | `/backups/postgres/` |
| Neo4j | Daily | 7 days | `/backups/neo4j/` |
| Qdrant | Daily | 7 days | `/backups/qdrant/` |
| OpenSearch | Daily | 7 days | `/backups/opensearch/` |
| Valkey | Every 6 hours | 3 days | `/backups/valkey/` |

### Pre-Backup Checklist

Before performing any backup:

1. [ ] Verify service is healthy: `docker compose ps`
2. [ ] Check available disk space: `df -h /backups`
3. [ ] Verify no migrations in progress (PostgreSQL)
4. [ ] Note current timestamp for backup filename

### Environment Variables Required

```bash
# Export these before running backup commands
export POSTGRES_USER=codingbrain
export POSTGRES_DB=codingbrain
export BACKUP_DIR=/backups
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)
```

---

## PostgreSQL Backup & Restore

PostgreSQL is the primary relational database containing memories, apps, users, feedback, and experiments.

### Backup Procedure

**1. Create backup directory (if not exists)**

```bash
mkdir -p ${BACKUP_DIR}/postgres
```

**2. Perform pg_dump with custom format (recommended)**

```bash
docker compose exec postgres pg_dump \
  -U ${POSTGRES_USER} \
  -d ${POSTGRES_DB} \
  -Fc \
  -Z 6 \
  --verbose \
  > ${BACKUP_DIR}/postgres/codingbrain_${TIMESTAMP}.dump
```

**Options explained**:
- `-Fc`: Custom format (supports parallel restore, selective restore)
- `-Z 6`: Compression level 6 (balanced speed/size)
- `--verbose`: Show progress

**3. Verify backup created successfully**

```bash
# Check file exists and has reasonable size
ls -lh ${BACKUP_DIR}/postgres/codingbrain_${TIMESTAMP}.dump

# Verify backup integrity
docker compose exec postgres pg_restore \
  --list \
  /backups/postgres/codingbrain_${TIMESTAMP}.dump > /dev/null
echo "Backup verified: exit code $?"
```

**4. Alternative: SQL format backup (human-readable)**

```bash
docker compose exec postgres pg_dump \
  -U ${POSTGRES_USER} \
  -d ${POSTGRES_DB} \
  --format=plain \
  | gzip > ${BACKUP_DIR}/postgres/codingbrain_${TIMESTAMP}.sql.gz
```

### Restore Procedure

**1. Stop application to prevent writes**

```bash
docker compose stop codingbrain-mcp codingbrain-ui
```

**2. Drop and recreate database (CAUTION: destructive)**

```bash
docker compose exec postgres psql -U ${POSTGRES_USER} -c "
  SELECT pg_terminate_backend(pid)
  FROM pg_stat_activity
  WHERE datname = '${POSTGRES_DB}' AND pid <> pg_backend_pid();
"

docker compose exec postgres psql -U ${POSTGRES_USER} -c "DROP DATABASE IF EXISTS ${POSTGRES_DB};"
docker compose exec postgres psql -U ${POSTGRES_USER} -c "CREATE DATABASE ${POSTGRES_DB};"
```

**3. Restore from custom format backup**

```bash
docker compose exec -T postgres pg_restore \
  -U ${POSTGRES_USER} \
  -d ${POSTGRES_DB} \
  --verbose \
  --jobs=4 \
  < ${BACKUP_DIR}/postgres/codingbrain_${TIMESTAMP}.dump
```

**Options explained**:
- `--jobs=4`: Parallel restore with 4 workers
- `-T`: Disable pseudo-TTY (required for piped input)

**4. Verify restore**

```bash
docker compose exec postgres psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "
  SELECT
    (SELECT COUNT(*) FROM memories) as memories_count,
    (SELECT COUNT(*) FROM apps) as apps_count,
    (SELECT COUNT(*) FROM users) as users_count;
"
```

**5. Restart application**

```bash
docker compose start codingbrain-mcp codingbrain-ui
```

### Point-in-Time Recovery (PITR)

For PITR, enable WAL archiving in postgresql.conf:

```bash
# Not enabled by default; requires volume configuration
# See PostgreSQL documentation for WAL archiving setup
```

---

## Neo4j Backup & Restore

Neo4j stores the CODE_* graph with code symbols, relationships, and cross-repository dependencies.

### Backup Procedure

**1. Create backup directory**

```bash
mkdir -p ${BACKUP_DIR}/neo4j
```

**2. Stop the database for consistent backup (recommended for production)**

For online backup (Enterprise only), use `neo4j-admin backup`. For Community edition:

```bash
# Stop Neo4j for offline backup
docker compose stop neo4j

# Create dump
docker run --rm \
  -v codingbrain_neo4j_data:/data \
  -v ${BACKUP_DIR}/neo4j:/backups \
  neo4j:5.26.4-community \
  neo4j-admin database dump neo4j --to-path=/backups/

# Rename with timestamp
mv ${BACKUP_DIR}/neo4j/neo4j.dump ${BACKUP_DIR}/neo4j/neo4j_${TIMESTAMP}.dump

# Restart Neo4j
docker compose start neo4j
```

**3. Verify backup**

```bash
ls -lh ${BACKUP_DIR}/neo4j/neo4j_${TIMESTAMP}.dump
# File should be >1KB for a non-empty database
```

### Restore Procedure

**1. Stop all services depending on Neo4j**

```bash
docker compose stop codingbrain-mcp codingbrain-ui neo4j
```

**2. Clear existing data and restore**

```bash
# Remove existing data (CAUTION: destructive)
docker volume rm codingbrain_neo4j_data
docker volume create codingbrain_neo4j_data

# Restore from dump
docker run --rm \
  -v codingbrain_neo4j_data:/data \
  -v ${BACKUP_DIR}/neo4j:/backups \
  neo4j:5.26.4-community \
  neo4j-admin database load neo4j \
    --from-path=/backups/neo4j_${TIMESTAMP}.dump \
    --overwrite-destination=true
```

**3. Restart services**

```bash
docker compose start neo4j
# Wait for Neo4j to be healthy
docker compose up -d codingbrain-mcp codingbrain-ui
```

**4. Verify restore**

```bash
docker compose exec neo4j cypher-shell -u ${NEO4J_USERNAME} -p ${NEO4J_PASSWORD} \
  "MATCH (n) RETURN labels(n)[0] as label, count(n) as count ORDER BY count DESC LIMIT 10;"
```

---

## Qdrant Backup & Restore

Qdrant stores vector embeddings for semantic search across memories and code.

### Backup Procedure

**1. Create snapshot via REST API**

```bash
# List all collections
curl -s http://localhost:6433/collections | jq '.result.collections[].name'

# Create snapshot for each collection
for collection in $(curl -s http://localhost:6433/collections | jq -r '.result.collections[].name'); do
  echo "Creating snapshot for collection: ${collection}"
  curl -X POST "http://localhost:6433/collections/${collection}/snapshots" | jq
done
```

**2. Copy snapshots to backup location**

```bash
mkdir -p ${BACKUP_DIR}/qdrant

# Snapshots are stored in the Qdrant storage volume
docker compose exec qdrant ls -la /qdrant/storage/snapshots/

# Copy all snapshots
docker cp codingbrain-qdrant:/qdrant/storage/snapshots/ ${BACKUP_DIR}/qdrant/${TIMESTAMP}/
```

**3. List available snapshots**

```bash
curl -s http://localhost:6433/collections/embeddings/snapshots | jq
```

### Restore Procedure

**1. Copy snapshot back to Qdrant container**

```bash
docker cp ${BACKUP_DIR}/qdrant/${TIMESTAMP}/ codingbrain-qdrant:/qdrant/storage/snapshots/
```

**2. Recover collection from snapshot**

```bash
# Get snapshot name
SNAPSHOT_NAME=$(ls ${BACKUP_DIR}/qdrant/${TIMESTAMP}/ | head -1)

# Recover collection
curl -X PUT "http://localhost:6433/collections/embeddings/snapshots/recover" \
  -H "Content-Type: application/json" \
  -d "{\"location\": \"file:///qdrant/storage/snapshots/${SNAPSHOT_NAME}\"}"
```

**3. Verify restore**

```bash
curl -s http://localhost:6433/collections/embeddings | jq '.result.points_count'
```

### Full Qdrant Volume Backup (Alternative)

For complete backup including all collections:

```bash
# Stop Qdrant
docker compose stop qdrant

# Backup entire volume
docker run --rm \
  -v codingbrain_qdrant_storage:/source:ro \
  -v ${BACKUP_DIR}/qdrant:/backup \
  alpine tar czf /backup/qdrant_full_${TIMESTAMP}.tar.gz -C /source .

# Restart Qdrant
docker compose start qdrant
```

---

## OpenSearch Backup & Restore

OpenSearch provides full-text and hybrid search capabilities.

### Setup Snapshot Repository (One-time)

**1. Create backup directory on host**

```bash
mkdir -p ${BACKUP_DIR}/opensearch
```

**2. Mount backup directory in OpenSearch container**

Add to docker-compose.yml opensearch service:

```yaml
volumes:
  - codingbrain_opensearch_data:/usr/share/opensearch/data
  - ${BACKUP_DIR}/opensearch:/mnt/backups
```

**3. Register snapshot repository**

```bash
curl -X PUT "http://localhost:9200/_snapshot/backups" \
  -H 'Content-Type: application/json' \
  -d '{
    "type": "fs",
    "settings": {
      "location": "/mnt/backups",
      "compress": true
    }
  }'
```

### Backup Procedure

**1. Create snapshot**

```bash
curl -X PUT "http://localhost:9200/_snapshot/backups/snapshot_${TIMESTAMP}?wait_for_completion=true" \
  -H 'Content-Type: application/json' \
  -d '{
    "indices": "*",
    "ignore_unavailable": true,
    "include_global_state": true
  }'
```

**2. Verify snapshot**

```bash
curl -s "http://localhost:9200/_snapshot/backups/snapshot_${TIMESTAMP}" | jq
```

**3. List all snapshots**

```bash
curl -s "http://localhost:9200/_snapshot/backups/_all" | jq '.snapshots[].snapshot'
```

### Restore Procedure

**1. Close indices that will be restored (if they exist)**

```bash
curl -X POST "http://localhost:9200/_all/_close"
```

**2. Restore from snapshot**

```bash
curl -X POST "http://localhost:9200/_snapshot/backups/snapshot_${TIMESTAMP}/_restore" \
  -H 'Content-Type: application/json' \
  -d '{
    "indices": "*",
    "ignore_unavailable": true,
    "include_global_state": true
  }'
```

**3. Verify restore**

```bash
curl -s "http://localhost:9200/_cat/indices?v"
```

### Alternative: Volume Backup

```bash
docker compose stop opensearch

docker run --rm \
  -v codingbrain_opensearch_data:/source:ro \
  -v ${BACKUP_DIR}/opensearch:/backup \
  alpine tar czf /backup/opensearch_full_${TIMESTAMP}.tar.gz -C /source .

docker compose start opensearch
```

---

## Valkey Backup & Restore

Valkey stores ephemeral session data, caches, and DPoP replay prevention tokens.

### Backup Procedure

**1. Trigger background save**

```bash
docker compose exec valkey valkey-cli BGSAVE
```

**2. Wait for save to complete**

```bash
# Check LASTSAVE timestamp
docker compose exec valkey valkey-cli LASTSAVE
```

**3. Copy RDB file**

```bash
mkdir -p ${BACKUP_DIR}/valkey
docker cp codingbrain-valkey:/data/dump.rdb ${BACKUP_DIR}/valkey/dump_${TIMESTAMP}.rdb
```

**4. Also backup AOF if enabled**

```bash
docker cp codingbrain-valkey:/data/appendonly.aof ${BACKUP_DIR}/valkey/appendonly_${TIMESTAMP}.aof 2>/dev/null || echo "AOF not found (expected if AOF disabled)"
```

### Restore Procedure

**1. Stop Valkey**

```bash
docker compose stop valkey
```

**2. Copy backup files**

```bash
docker cp ${BACKUP_DIR}/valkey/dump_${TIMESTAMP}.rdb codingbrain-valkey:/data/dump.rdb
```

**3. Restart Valkey**

```bash
docker compose start valkey
```

**4. Verify restore**

```bash
docker compose exec valkey valkey-cli DBSIZE
docker compose exec valkey valkey-cli INFO keyspace
```

### Note on Valkey Data

Valkey data is typically ephemeral:
- Session data: Regenerated on login
- DPoP cache: Security tokens with TTL
- Rate limit buckets: Reset naturally

For most disaster scenarios, Valkey can be safely restarted empty without data loss impact.

---

## Full System Backup

For complete disaster recovery, back up all services together.

### Full Backup Script

```bash
#!/bin/bash
set -e

BACKUP_DIR=/backups
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH=${BACKUP_DIR}/full_${TIMESTAMP}

echo "=== Full System Backup Starting: ${TIMESTAMP} ==="
mkdir -p ${BACKUP_PATH}

# 1. PostgreSQL
echo ">>> Backing up PostgreSQL..."
docker compose exec -T postgres pg_dump \
  -U ${POSTGRES_USER} \
  -d ${POSTGRES_DB} \
  -Fc -Z 6 \
  > ${BACKUP_PATH}/postgres.dump

# 2. Neo4j (requires stop)
echo ">>> Backing up Neo4j..."
docker compose stop neo4j
docker run --rm \
  -v codingbrain_neo4j_data:/data \
  -v ${BACKUP_PATH}:/backups \
  neo4j:5.26.4-community \
  neo4j-admin database dump neo4j --to-path=/backups/
docker compose start neo4j

# 3. Qdrant
echo ">>> Backing up Qdrant..."
docker compose stop qdrant
docker run --rm \
  -v codingbrain_qdrant_storage:/source:ro \
  -v ${BACKUP_PATH}:/backup \
  alpine tar czf /backup/qdrant.tar.gz -C /source .
docker compose start qdrant

# 4. OpenSearch
echo ">>> Backing up OpenSearch..."
docker compose stop opensearch
docker run --rm \
  -v codingbrain_opensearch_data:/source:ro \
  -v ${BACKUP_PATH}:/backup \
  alpine tar czf /backup/opensearch.tar.gz -C /source .
docker compose start opensearch

# 5. Valkey
echo ">>> Backing up Valkey..."
docker compose exec valkey valkey-cli BGSAVE
sleep 5
docker cp codingbrain-valkey:/data/dump.rdb ${BACKUP_PATH}/valkey.rdb

# 6. Create manifest
echo ">>> Creating backup manifest..."
cat > ${BACKUP_PATH}/manifest.json << EOF
{
  "timestamp": "${TIMESTAMP}",
  "services": {
    "postgres": "postgres.dump",
    "neo4j": "neo4j.dump",
    "qdrant": "qdrant.tar.gz",
    "opensearch": "opensearch.tar.gz",
    "valkey": "valkey.rdb"
  },
  "versions": {
    "postgres": "pgvector/pgvector:pg16",
    "neo4j": "neo4j:5.26.4-community",
    "qdrant": "qdrant/qdrant:v1.12.5",
    "opensearch": "opensearchproject/opensearch:2.13.0",
    "valkey": "valkey/valkey:8.0.2-alpine"
  }
}
EOF

echo "=== Full System Backup Complete: ${BACKUP_PATH} ==="
ls -lh ${BACKUP_PATH}/
```

### Full Restore Script

```bash
#!/bin/bash
set -e

BACKUP_PATH=$1
if [ -z "$BACKUP_PATH" ]; then
  echo "Usage: $0 /path/to/backup/full_YYYYMMDD_HHMMSS"
  exit 1
fi

echo "=== Full System Restore Starting from: ${BACKUP_PATH} ==="

# Stop all services
docker compose down

# 1. Restore PostgreSQL
echo ">>> Restoring PostgreSQL..."
docker compose up -d postgres
sleep 10
docker compose exec -T postgres psql -U ${POSTGRES_USER} -c "DROP DATABASE IF EXISTS ${POSTGRES_DB};"
docker compose exec -T postgres psql -U ${POSTGRES_USER} -c "CREATE DATABASE ${POSTGRES_DB};"
docker compose exec -T postgres pg_restore \
  -U ${POSTGRES_USER} \
  -d ${POSTGRES_DB} \
  --verbose \
  < ${BACKUP_PATH}/postgres.dump

# 2. Restore Neo4j
echo ">>> Restoring Neo4j..."
docker volume rm codingbrain_neo4j_data 2>/dev/null || true
docker volume create codingbrain_neo4j_data
docker run --rm \
  -v codingbrain_neo4j_data:/data \
  -v ${BACKUP_PATH}:/backups \
  neo4j:5.26.4-community \
  neo4j-admin database load neo4j \
    --from-path=/backups/neo4j.dump \
    --overwrite-destination=true

# 3. Restore Qdrant
echo ">>> Restoring Qdrant..."
docker volume rm codingbrain_qdrant_storage 2>/dev/null || true
docker volume create codingbrain_qdrant_storage
docker run --rm \
  -v codingbrain_qdrant_storage:/target \
  -v ${BACKUP_PATH}:/backup:ro \
  alpine tar xzf /backup/qdrant.tar.gz -C /target

# 4. Restore OpenSearch
echo ">>> Restoring OpenSearch..."
docker volume rm codingbrain_opensearch_data 2>/dev/null || true
docker volume create codingbrain_opensearch_data
docker run --rm \
  -v codingbrain_opensearch_data:/target \
  -v ${BACKUP_PATH}:/backup:ro \
  alpine tar xzf /backup/opensearch.tar.gz -C /target

# 5. Restore Valkey
echo ">>> Restoring Valkey..."
docker compose up -d valkey
sleep 5
docker cp ${BACKUP_PATH}/valkey.rdb codingbrain-valkey:/data/dump.rdb
docker compose restart valkey

# Start all services
echo ">>> Starting all services..."
docker compose up -d

echo "=== Full System Restore Complete ==="
docker compose ps
```

---

## Disaster Recovery Procedures

### Scenario 1: Single Service Failure

1. Check service logs: `docker compose logs <service>`
2. Attempt restart: `docker compose restart <service>`
3. If data corruption suspected, restore from latest backup
4. Verify data integrity with service-specific checks

### Scenario 2: Full System Recovery

1. Provision new infrastructure
2. Install Docker and Docker Compose
3. Clone repository and configure `.env`
4. Copy backup files to `/backups`
5. Run full restore script
6. Verify all services healthy
7. Update DNS/load balancer

### Scenario 3: Partial Data Loss

1. Identify affected service
2. Stop dependent services
3. Restore affected service from backup
4. Verify data consistency across services
5. Rebuild indexes if necessary (OpenSearch, Qdrant)

### RTO/RPO Summary

| Service | RPO (Data Loss) | RTO (Recovery Time) |
|---------|-----------------|---------------------|
| PostgreSQL | 6 hours | 1 hour |
| Neo4j | 24 hours | 2 hours |
| Qdrant | 24 hours | 1 hour |
| OpenSearch | 24 hours | 1 hour |
| Valkey | N/A (ephemeral) | 10 minutes |

---

## Verification Checklist

### Post-Backup Verification

- [ ] Backup file exists and size > 0
- [ ] Backup file modified within expected timeframe
- [ ] Backup integrity check passes (pg_restore --list, tar -tzf)
- [ ] Backup transferred to off-site storage (if applicable)
- [ ] Verification logged with timestamp

### Post-Restore Verification

- [ ] All services healthy: `docker compose ps`
- [ ] PostgreSQL row counts match pre-backup
- [ ] Neo4j node counts match pre-backup
- [ ] Qdrant vector counts match pre-backup
- [ ] OpenSearch document counts match pre-backup
- [ ] Application endpoints responding
- [ ] Test user can log in and access data

### Automated Verification Script

See `scripts/verify_backup.py` for automated nightly verification.

---

## Troubleshooting

### PostgreSQL

**Problem**: pg_restore fails with "database is being accessed by other users"

```bash
# Terminate connections
docker compose exec postgres psql -U ${POSTGRES_USER} -c "
  SELECT pg_terminate_backend(pid)
  FROM pg_stat_activity
  WHERE datname = '${POSTGRES_DB}';
"
```

**Problem**: Backup file is corrupted

```bash
# Verify backup
docker compose exec postgres pg_restore --list backup.dump
# If fails, restore from older backup
```

### Neo4j

**Problem**: neo4j-admin dump fails with "database in use"

```bash
# Stop Neo4j completely
docker compose stop neo4j
# Then run dump
```

**Problem**: neo4j-admin load fails with version mismatch

```bash
# Use same Neo4j version for restore as was used for backup
# Check manifest.json for backup version
```

### Qdrant

**Problem**: Snapshot recovery fails

```bash
# Check Qdrant logs
docker compose logs qdrant
# Ensure snapshot file is accessible
docker compose exec qdrant ls -la /qdrant/storage/snapshots/
```

### OpenSearch

**Problem**: Snapshot repository not registered

```bash
# Re-register repository
curl -X PUT "http://localhost:9200/_snapshot/backups" \
  -H 'Content-Type: application/json' \
  -d '{"type": "fs", "settings": {"location": "/mnt/backups"}}'
```

### Valkey

**Problem**: RDB restore not working

```bash
# Ensure Valkey is stopped before copying RDB
docker compose stop valkey
# Copy file
docker cp dump.rdb codingbrain-valkey:/data/dump.rdb
# Start Valkey
docker compose start valkey
```

---

## Appendix: Cron Configuration

### Automated Backup Crontab

```bash
# PostgreSQL every 6 hours
0 */6 * * * /opt/codingbrain/scripts/backup_postgres.sh >> /var/log/codingbrain/backup.log 2>&1

# Neo4j daily at 2 AM
0 2 * * * /opt/codingbrain/scripts/backup_neo4j.sh >> /var/log/codingbrain/backup.log 2>&1

# Qdrant daily at 3 AM
0 3 * * * /opt/codingbrain/scripts/backup_qdrant.sh >> /var/log/codingbrain/backup.log 2>&1

# OpenSearch daily at 4 AM
0 4 * * * /opt/codingbrain/scripts/backup_opensearch.sh >> /var/log/codingbrain/backup.log 2>&1

# Valkey every 6 hours
0 */6 * * * /opt/codingbrain/scripts/backup_valkey.sh >> /var/log/codingbrain/backup.log 2>&1

# Full backup weekly on Sunday at 1 AM
0 1 * * 0 /opt/codingbrain/scripts/backup_full.sh >> /var/log/codingbrain/backup.log 2>&1

# Nightly verification at 5 AM
0 5 * * * /opt/codingbrain/scripts/verify_backup.py >> /var/log/codingbrain/verify.log 2>&1
```

---

**Document Maintainer**: Platform Team
**Review Cycle**: Quarterly or after infrastructure changes
