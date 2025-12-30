# Deployment Runbook

**Document Version**: 1.0
**Last Updated**: 2025-12-28
**Applies to**: coding-brain production deployments

---

## Table of Contents

1. [Overview](#overview)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Blue-Green Deployment](#blue-green-deployment)
4. [Canary Deployment](#canary-deployment)
5. [Rollback Procedures](#rollback-procedures)
6. [Database Migrations](#database-migrations)
7. [Health Check Verification](#health-check-verification)
8. [Post-Deployment Verification](#post-deployment-verification)
9. [Troubleshooting](#troubleshooting)
10. [Emergency Procedures](#emergency-procedures)

---

## Overview

This runbook documents deployment procedures for the coding-brain system using blue-green and canary strategies to enable zero-downtime deployments.
This repo ships a single `openmemory/docker-compose.yml`; blue/green and canary examples below assume you maintain your own environment-specific compose overrides.

### Deployment Strategies

| Strategy | Use Case | Risk Level | Rollback Time |
|----------|----------|------------|---------------|
| Blue-Green | Major releases, breaking changes | Low | Instant |
| Canary | Minor releases, feature flags | Very Low | Instant |
| Rolling | Patches, hotfixes | Medium | 5-10 minutes |

### Environment Architecture

```
Production (Blue/Green Setup)
├── Load Balancer (nginx/traefik)
│   ├── Blue Environment (Active)
│   │   ├── codingbrain-mcp-blue
│   │   ├── postgres-blue (replica)
│   │   └── supporting services
│   └── Green Environment (Standby)
│       ├── codingbrain-mcp-green
│       ├── postgres-green (replica)
│       └── supporting services
└── Shared Services (Single Instance)
    ├── PostgreSQL Primary
    ├── Neo4j
    ├── Qdrant
    ├── OpenSearch
    └── Valkey
```

---

## Pre-Deployment Checklist

Before starting any deployment:

### Code Verification
- [ ] All CI tests passing on the target commit
- [ ] Security scan completed with no CRITICAL vulnerabilities
- [ ] Code review approved
- [ ] Version tag created: `git tag -a vX.Y.Z -m "Release X.Y.Z"`

### Environment Verification
- [ ] Backup created within last 6 hours (see RUNBOOK-BACKUP-RESTORE.md)
- [ ] Backup verification passed
- [ ] Standby environment healthy
- [ ] Sufficient disk space (> 20% free)
- [ ] No ongoing incidents

### Communication
- [ ] Deployment window announced
- [ ] On-call engineer notified
- [ ] Rollback plan documented
- [ ] Stakeholders informed

### Required Credentials
- [ ] Docker registry access
- [ ] SSH access to deployment targets
- [ ] Database credentials for migrations
- [ ] Load balancer control access

---

## Blue-Green Deployment

Blue-green deployment maintains two identical production environments, with traffic switched between them.

### Step 1: Prepare Green Environment

**1.1 Verify Green is healthy and idle**

```bash
# Check Green environment status
docker compose -f docker-compose.green.yml ps
curl -sf http://green.internal:8765/health/ready || echo "Green not ready"
```

**1.2 Pull new image to Green**

```bash
# Pull the new version
docker pull codingbrain/mcp:${NEW_VERSION}

# Update Green environment
docker compose -f docker-compose.green.yml pull codingbrain-mcp
```

**1.3 Apply configuration changes**

```bash
# Update environment variables
cp .env.production .env.green
# Edit .env.green with any new configuration

# Apply database migrations (if any)
# See "Database Migrations" section below
```

### Step 2: Deploy to Green

**2.1 Stop and restart Green with new version**

```bash
docker compose -f docker-compose.green.yml down
docker compose -f docker-compose.green.yml up -d
```

**2.2 Wait for Green to become healthy**

```bash
# Wait for health check to pass
for i in {1..30}; do
  if curl -sf http://green.internal:8765/health/ready; then
    echo "Green is healthy"
    break
  fi
  echo "Waiting for Green... ($i/30)"
  sleep 10
done
```

**2.3 Run smoke tests on Green**

```bash
# Basic API health
curl http://green.internal:8765/health/live
curl http://green.internal:8765/health/ready
curl http://green.internal:8765/health/deps

# Functional smoke tests
API_BASE_URL=http://green.internal:8765 ./scripts/smoke_test.sh
```

### Step 3: Switch Traffic

**3.1 Update load balancer configuration**

```bash
# For nginx
cat > /etc/nginx/conf.d/upstream.conf << 'EOF'
upstream codingbrain {
    # Switch from Blue to Green
    # server blue.internal:8765;  # Comment out Blue
    server green.internal:8765;   # Enable Green
}
EOF

nginx -s reload
```

**3.2 Verify traffic routing**

```bash
# Check requests are going to Green
for i in {1..5}; do
  curl -s http://app.example.com/health/live | grep -q "ok" && echo "Request $i: OK"
done
```

**3.3 Monitor for errors**

```bash
# Watch error rates for 5 minutes
watch -n 5 'curl -s http://metrics.internal/api/v1/query?query=rate(http_requests_total{status=~"5.."}[1m])'
```

### Step 4: Confirm Deployment

**4.1 Verify all health endpoints**

```bash
curl http://app.example.com/health/live
curl http://app.example.com/health/ready
curl http://app.example.com/health/deps
```

**4.2 Check application metrics**

- Request latency P95 < 500ms
- Error rate < 0.1%
- Memory usage stable
- No connection pool exhaustion

**4.3 Mark Blue as standby**

```bash
# Blue becomes the new standby
echo "Blue is now standby at version $(docker compose -f docker-compose.blue.yml exec codingbrain-mcp cat /version)"
```

---

## Canary Deployment

Canary deployment gradually shifts traffic to the new version.

### Step 1: Deploy Canary Instance

**1.1 Create canary deployment**

```bash
docker compose -f docker-compose.canary.yml up -d
```

**1.2 Verify canary health**

```bash
curl http://canary.internal:8765/health/ready
```

### Step 2: Configure Traffic Split

**2.1 Initial split: 1% to canary**

```bash
# nginx weighted upstream
cat > /etc/nginx/conf.d/upstream.conf << 'EOF'
upstream codingbrain {
    server main.internal:8765 weight=99;
    server canary.internal:8765 weight=1;
}
EOF
nginx -s reload
```

**2.2 Monitor canary metrics**

```bash
# Compare error rates
MAIN_ERRORS=$(curl -s 'http://metrics/query?q=rate(http_errors{env="main"}[5m])')
CANARY_ERRORS=$(curl -s 'http://metrics/query?q=rate(http_errors{env="canary"}[5m])')
echo "Main: ${MAIN_ERRORS}, Canary: ${CANARY_ERRORS}"
```

### Step 3: Gradual Traffic Increase

**3.1 Traffic progression schedule**

| Stage | Canary % | Wait Time | Success Criteria |
|-------|----------|-----------|------------------|
| 1 | 1% | 5 min | Error rate < 0.1%, latency normal |
| 2 | 5% | 10 min | Same as above |
| 3 | 25% | 15 min | Same as above |
| 4 | 50% | 15 min | Same as above |
| 5 | 100% | - | Full rollout |

**3.2 Update weights**

```bash
# Stage 2: 5% to canary
cat > /etc/nginx/conf.d/upstream.conf << 'EOF'
upstream codingbrain {
    server main.internal:8765 weight=95;
    server canary.internal:8765 weight=5;
}
EOF
nginx -s reload
```

### Step 4: Complete Rollout

**4.1 Full traffic to new version**

```bash
# 100% to new version
cat > /etc/nginx/conf.d/upstream.conf << 'EOF'
upstream codingbrain {
    server canary.internal:8765;  # New version
}
EOF
nginx -s reload
```

**4.2 Update main deployment**

```bash
docker compose -f docker-compose.yml pull
docker compose -f docker-compose.yml up -d
```

**4.3 Remove canary**

```bash
docker compose -f docker-compose.canary.yml down
```

---

## Rollback Procedures

### Instant Rollback (Blue-Green)

**If issues detected after traffic switch:**

```bash
# Switch traffic back to Blue (previous version)
cat > /etc/nginx/conf.d/upstream.conf << 'EOF'
upstream codingbrain {
    server blue.internal:8765;   # Restore Blue
    # server green.internal:8765;  # Disable Green
}
EOF
nginx -s reload

# Verify rollback
curl http://app.example.com/health/live
```

### Canary Rollback

**If issues detected during canary:**

```bash
# Remove canary from rotation
cat > /etc/nginx/conf.d/upstream.conf << 'EOF'
upstream codingbrain {
    server main.internal:8765;
    # Canary removed
}
EOF
nginx -s reload

# Stop canary
docker compose -f docker-compose.canary.yml down
```

### Database Rollback

**If migration causes issues:**

```bash
# 1. Switch traffic away first
# 2. Run Alembic downgrade
docker compose exec codingbrain-mcp alembic downgrade -1

# 3. Verify database state
docker compose exec postgres psql -U ${POSTGRES_USER} -c "SELECT version_num FROM alembic_version;"

# 4. Restore from backup if necessary
# See RUNBOOK-BACKUP-RESTORE.md
```

---

## Database Migrations

### Pre-Migration Checks

```bash
# 1. Create fresh backup
# See RUNBOOK-BACKUP-RESTORE.md

# 2. Verify backup
# Use the verification checklist in RUNBOOK-BACKUP-RESTORE.md

# 3. Check migration status
docker compose exec codingbrain-mcp alembic current
docker compose exec codingbrain-mcp alembic history
```

### Safe Migration Pattern

**For backward-compatible changes:**

```bash
# Apply migration to Green/Canary first
docker compose -f docker-compose.green.yml exec codingbrain-mcp alembic upgrade head

# Verify migration
docker compose -f docker-compose.green.yml exec codingbrain-mcp alembic current
```

**For breaking changes (two-phase):**

1. Phase 1: Deploy code that works with both old and new schema
2. Run migration
3. Phase 2: Deploy code that only works with new schema

```bash
# Phase 1: Dual-compatible code deployed

# Apply migration
docker compose exec codingbrain-mcp alembic upgrade head

# Phase 2: Clean up code deployed
```

### Migration Verification

```bash
# Verify row counts
docker compose exec postgres psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "
  SELECT
    (SELECT COUNT(*) FROM memories) as memories,
    (SELECT COUNT(*) FROM apps) as apps,
    (SELECT COUNT(*) FROM users) as users;
"

# Verify schema
docker compose exec postgres psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "\dt"
```

---

## Health Check Verification

### Health Endpoints

| Endpoint | Purpose | Expected Response |
|----------|---------|-------------------|
| `/health/live` | Liveness probe | `{"status": "ok"}` |
| `/health/ready` | Readiness probe | `{"status": "ok"|"degraded", "dependencies": {...}}` |
| `/health/deps` | Dependency health | `{"status": "...", "dependencies": {...}}` |

### Verification Script

```bash
#!/bin/bash
# verify_health.sh

HOST=${1:-http://localhost:8865}

echo "Checking liveness..."
curl -sf "${HOST}/health/live" || exit 1

echo "Checking readiness..."
READY=$(curl -sf "${HOST}/health/ready")
echo $READY | jq .

echo "Checking dependencies..."
DEPS=$(curl -sf "${HOST}/health/deps")
echo $DEPS | jq .

# Check all deps are healthy
UNHEALTHY=$(echo "$DEPS" | jq '[.dependencies[] | select(.status != "healthy")] | length')
if [ "$UNHEALTHY" -gt 0 ]; then
  echo "WARNING: $UNHEALTHY unhealthy dependencies"
  exit 1
fi

echo "All health checks passed!"
```

### Monitoring Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| Response time P95 | > 500ms | > 2000ms |
| Error rate | > 0.1% | > 1% |
| Memory usage | > 80% | > 95% |
| CPU usage | > 70% | > 90% |
| Connection pool | > 80% | > 95% |

---

## Post-Deployment Verification

### Functional Verification

```bash
# API smoke tests
API_BASE_URL=http://app.example.com ./scripts/smoke_test.sh

# Check key endpoints
curl http://app.example.com/v1/memories
curl http://app.example.com/v1/apps
curl http://app.example.com/health/deps
```

### Performance Verification

```bash
# Check latency metrics
curl -s 'http://metrics/query?q=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m]))'

# Check error rates
curl -s 'http://metrics/query?q=rate(http_requests_total{status=~"5.."}[5m])'
```

### Log Verification

```bash
# Check for errors in logs
docker compose logs --since 10m codingbrain-mcp | grep -i error

# Check for startup issues
docker compose logs codingbrain-mcp | grep -i "started\|ready\|listening"
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker compose logs codingbrain-mcp

# Check resource limits
docker stats codingbrain-mcp

# Check for port conflicts
lsof -i :8865
```

### Health Check Failing

```bash
# Check dependency connectivity
docker compose exec codingbrain-mcp python -c "
from app.settings import Settings
s = Settings()
print(f'Postgres: {s.database_url}')
print(f'Neo4j: {s.neo4j_url}')
"

# Test individual dependencies
docker compose exec postgres pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB} || echo "Postgres unreachable"
```

### High Latency After Deployment

```bash
# Check connection pools
docker compose exec postgres psql -U ${POSTGRES_USER} -c "SELECT * FROM pg_stat_activity;"

# Check for slow queries
docker compose exec postgres psql -U ${POSTGRES_USER} -c "
  SELECT query, calls, mean_time
  FROM pg_stat_statements
  ORDER BY mean_time DESC
  LIMIT 10;
"
```

### Memory Issues

```bash
# Check container memory
docker stats --no-stream codingbrain-mcp

# Check for memory leaks
docker compose exec codingbrain-mcp python -c "
import tracemalloc
tracemalloc.start()
# ... run some operations
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
"
```

---

## Emergency Procedures

### Full System Outage

1. **Triage**
   ```bash
   # Check what's running
   docker compose ps

   # Check logs for all services
   docker compose logs --tail 100
   ```

2. **Quick restore**
   ```bash
   # Restart all services
   docker compose down
   docker compose up -d
   ```

3. **If restart fails**
   ```bash
   # Rollback to known-good version
   docker compose pull codingbrain/mcp:${LAST_KNOWN_GOOD}
   docker compose up -d
   ```

### Database Corruption

1. **Stop application immediately**
   ```bash
   docker compose stop codingbrain-mcp codingbrain-ui
   ```

2. **Restore from backup**
   ```bash
   # See RUNBOOK-BACKUP-RESTORE.md for restore steps
   ```

3. **Restart application**
   ```bash
   docker compose start codingbrain-mcp codingbrain-ui
   ```

### Security Incident

1. **Isolate affected services**
   ```bash
   docker network disconnect codingbrain-net codingbrain-mcp
   ```

2. **Preserve evidence**
   ```bash
   docker logs codingbrain-mcp > incident_$(date +%Y%m%d_%H%M%S).log
   ```

3. **Follow incident response procedure**
   - Contact security team
   - Do not restart services until cleared
   - Preserve all logs and state

---

## Appendix: Deployment Automation

### Deployment Script Template

```bash
#!/bin/bash
# deploy.sh

set -e

VERSION=${1:?Version required}
STRATEGY=${2:-blue-green}  # blue-green or canary

echo "Deploying version ${VERSION} using ${STRATEGY} strategy"

# Pre-deployment checks
./scripts/pre-deploy-check.sh  # Placeholder; not included in this repo

# Run deployment
case $STRATEGY in
  blue-green)
    ./scripts/deploy-blue-green.sh $VERSION  # Placeholder; not included in this repo
    ;;
  canary)
    ./scripts/deploy-canary.sh $VERSION  # Placeholder; not included in this repo
    ;;
  *)
    echo "Unknown strategy: $STRATEGY"
    exit 1
    ;;
esac

# Post-deployment verification
./scripts/post-deploy-verify.sh  # Placeholder; not included in this repo

echo "Deployment complete!"
```

---

**Document Maintainer**: Platform Team
**Review Cycle**: Quarterly or after major infrastructure changes
