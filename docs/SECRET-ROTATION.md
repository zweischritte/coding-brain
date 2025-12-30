# Secret Rotation Procedures

This document describes the rotation schedules and procedures for all secrets used in Coding Brain.

## Rotation Schedule

| Secret | Rotation Period | Complexity Requirements | Last Rotated |
|--------|-----------------|------------------------|--------------|
| JWT_SECRET_KEY | 90 days | Min 32 characters, cryptographically random | YYYY-MM-DD |
| POSTGRES_PASSWORD | 180 days | Min 16 characters, mixed case + numbers | YYYY-MM-DD |
| NEO4J_PASSWORD | 180 days | Min 16 characters, mixed case + numbers | YYYY-MM-DD |
| OPENAI_API_KEY | 30-90 days | As provided by OpenAI | YYYY-MM-DD |
| OPENSEARCH_INITIAL_ADMIN_PASSWORD | 180 days | Upper, lower, number, special char | YYYY-MM-DD |

## Rotation Procedures

### JWT Secret Key (JWT_SECRET_KEY)

The JWT secret is used to sign authentication tokens. Rotation requires a rolling update to support both old and new keys during the transition.

#### Prerequisites
- Access to the `.env` file or secrets management system
- Ability to restart API containers
- Understanding that active sessions will be invalidated

#### Procedure

1. **Generate new key**:
   ```bash
   openssl rand -base64 48
   ```

2. **Plan the rotation window**:
   - Choose a low-traffic period
   - Inform users of potential re-authentication requirement

3. **Update the secret**:
   ```bash
   # Update JWT_SECRET_KEY in .env
   JWT_SECRET_KEY=<new-generated-key>
   ```

4. **Restart API services**:
   ```bash
   docker compose restart codingbrain-mcp
   ```

5. **Verify**:
   ```bash
   # Check that the service is healthy
   curl http://localhost:8865/health/live

   # Verify new tokens are issued correctly
   # (requires authentication test)
   ```

6. **Update rotation log**:
   - Record the rotation date in this document
   - Update any monitoring/alerting systems

#### Notes
- All existing tokens will become invalid immediately
- Users will need to re-authenticate
- For zero-downtime rotation, implement dual-key validation (future enhancement)

---

### PostgreSQL Password (POSTGRES_PASSWORD)

#### Prerequisites
- Access to PostgreSQL admin account
- Ability to update `.env` and restart services
- Brief maintenance window (1-2 minutes)

#### Procedure

1. **Generate new password**:
   ```bash
   openssl rand -base64 24 | tr -d '/+=' | head -c 24
   ```

2. **Connect to PostgreSQL and update password**:
   ```bash
   docker compose exec postgres psql -U codingbrain -d codingbrain
   ```
   ```sql
   ALTER USER codingbrain WITH PASSWORD 'new-password-here';
   ```

3. **Update `.env`**:
   ```bash
   POSTGRES_PASSWORD=new-password-here
   ```

4. **Restart dependent services**:
   ```bash
   docker compose restart codingbrain-mcp
   ```

5. **Verify connectivity**:
   ```bash
   docker compose exec codingbrain-mcp python -c "from app.database import engine; engine.connect(); print('Connected!')"
   ```

---

### Neo4j Password (NEO4J_PASSWORD)

#### Prerequisites
- Access to Neo4j admin account
- Ability to update `.env` and restart services

#### Procedure

1. **Generate new password**:
   ```bash
   openssl rand -base64 24 | tr -d '/+=' | head -c 24
   ```

2. **Connect to Neo4j and update password**:
   ```bash
   docker compose exec neo4j cypher-shell -u neo4j -p old-password
   ```
   ```cypher
   ALTER CURRENT USER SET PASSWORD FROM 'old-password' TO 'new-password';
   ```

3. **Update `.env`**:
   ```bash
   NEO4J_PASSWORD=new-password-here
   ```

4. **Restart dependent services**:
   ```bash
   docker compose restart codingbrain-mcp
   ```

5. **Verify connectivity**:
   ```bash
   curl http://localhost:8865/health/ready
   ```

---

### OpenAI API Key (OPENAI_API_KEY)

#### Prerequisites
- Access to OpenAI account
- Ability to update `.env` and restart services

#### Procedure

1. **Generate new key in OpenAI dashboard**:
   - Go to https://platform.openai.com/api-keys
   - Create a new API key
   - Copy the key (it won't be shown again)

2. **Update `.env`**:
   ```bash
   OPENAI_API_KEY=sk-proj-new-key-here
   ```

3. **Restart API services**:
   ```bash
   docker compose restart codingbrain-mcp
   ```

4. **Verify functionality**:
   ```bash
   # Test embedding generation or LLM call
   curl -X POST http://localhost:8865/api/v1/memories \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer <token>" \
     -d '{"messages": [{"role": "user", "content": "Test memory"}]}'
   ```

5. **Revoke old key**:
   - Return to OpenAI dashboard
   - Delete the old API key

---

### OpenSearch Admin Password (OPENSEARCH_INITIAL_ADMIN_PASSWORD)

#### Prerequisites
- Access to OpenSearch admin
- Understanding that this is primarily for initial setup
- Note: `openmemory/docker-compose.yml` disables OpenSearch security (`plugins.security.disabled=true`).
  In that default setup, the admin password is not enforced; only rotate it if you enable security.
  If security is disabled, skip step 2 and just update `.env` + restart.

#### Procedure

1. **Generate new password** (must meet complexity requirements):
   ```bash
   # Must include: uppercase, lowercase, number, special character
   openssl rand -base64 24 | tr -d '/+' | head -c 20
   # Manually add special character and ensure complexity
   ```

2. **Update OpenSearch internal users (only if security is enabled)**:
   ```bash
   docker compose exec opensearch bash
   cd /usr/share/opensearch/plugins/opensearch-security/tools
   ./securityadmin.sh -cd ../securityconfig/ -icl -nhnv \
     -cacert /usr/share/opensearch/config/root-ca.pem \
     -cert /usr/share/opensearch/config/admin.pem \
     -key /usr/share/opensearch/config/admin-key.pem
   ```

3. **Update `.env`**:
   ```bash
   OPENSEARCH_INITIAL_ADMIN_PASSWORD=NewPassword1!
   ```

4. **Restart services**:
   ```bash
   docker compose restart opensearch codingbrain-mcp
   ```

---

## Emergency Rotation

If a secret is compromised:

1. **Immediately generate a new secret** using the procedures above
2. **Revoke the compromised secret** at the source (API keys, database passwords)
3. **Update all environments** (development, staging, production)
4. **Audit access logs** for suspicious activity
5. **Notify security team** and document the incident

## Automation (Future)

Consider implementing:
- HashiCorp Vault for secret management
- AWS Secrets Manager or Azure Key Vault
- Automated rotation scripts with Kubernetes CronJobs
- Secret rotation monitoring and alerting

## Monitoring

Set up alerts for:
- Secrets approaching rotation deadline (7 days before)
- Failed authentication attempts (may indicate compromised secrets)
- Unusual API key usage patterns

---

*Last updated: 2025-12-27*
