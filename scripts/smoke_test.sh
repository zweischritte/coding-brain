#!/usr/bin/env bash
# =============================================================================
# Smoke Test Script for Coding Brain API
# =============================================================================
#
# This script verifies that the API stack is healthy by checking critical
# endpoints. It exits 0 if all checks pass, non-zero otherwise.
#
# Usage:
#   ./scripts/smoke_test.sh              # Uses default API_BASE_URL
#   API_BASE_URL=http://custom:8765 ./scripts/smoke_test.sh
#
# Endpoints tested:
#   - GET /health/live      - Basic liveness probe
#   - GET /health/deps      - Dependency health (postgres, neo4j, etc.)
#   - GET /mcp/health       - MCP server health
#   - GET /metrics          - Prometheus metrics endpoint
#
# Optional (if running inside Docker):
#   - alembic current       - Verify migration state
#
# =============================================================================

set -uo pipefail
# Note: we don't use 'set -e' because we want to continue after failed checks

# Configuration
# Default port 8865 matches docker-compose port mapping (8865:8765)
API_BASE_URL="${API_BASE_URL:-http://localhost:8865}"
TIMEOUT="${TIMEOUT:-10}"
VERBOSE="${VERBOSE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

log_info() {
    echo -e "${NC}[INFO] $1${NC}"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARNINGS++))
}

# Check if curl is available
check_curl() {
    if ! command -v curl &> /dev/null; then
        echo "Error: curl is required but not installed."
        exit 1
    fi
}

# Make HTTP request and check status code
# Args: $1=endpoint, $2=expected_status (default 200), $3=description
check_endpoint() {
    local endpoint="$1"
    local expected_status="${2:-200}"
    local description="${3:-$endpoint}"
    local url="${API_BASE_URL}${endpoint}"

    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Checking: $url"
    fi

    local response
    local http_code

    # Make request and capture both body and status code
    response=$(curl -s -w "\n%{http_code}" --max-time "$TIMEOUT" "$url" 2>/dev/null) || {
        log_fail "$description - Connection failed (timeout or refused)"
        return 1
    }

    # Extract status code (last line) and body (everything else)
    http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | sed '$d')

    if [[ "$http_code" == "$expected_status" ]]; then
        log_pass "$description (HTTP $http_code)"
        if [[ "$VERBOSE" == "true" ]] && [[ -n "$body" ]]; then
            echo "  Response: ${body:0:200}..."
        fi
        return 0
    else
        log_fail "$description - Expected HTTP $expected_status, got $http_code"
        if [[ -n "$body" ]]; then
            echo "  Response: ${body:0:200}"
        fi
        return 1
    fi
}

# Check endpoint accepting multiple status codes
# Args: $1=endpoint, $2=comma-separated-status-codes, $3=description
check_endpoint_accept_multiple() {
    local endpoint="$1"
    local accepted_codes="$2"
    local description="${3:-$endpoint}"
    local url="${API_BASE_URL}${endpoint}"

    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Checking: $url (accepting: $accepted_codes)"
    fi

    local response
    local http_code

    response=$(curl -s -w "\n%{http_code}" --max-time "$TIMEOUT" "$url" 2>/dev/null) || {
        log_fail "$description - Connection failed (timeout or refused)"
        return 1
    }

    http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | sed '$d')

    # Check if http_code is in the accepted list
    if echo "$accepted_codes" | tr ',' '\n' | grep -q "^${http_code}$"; then
        if [[ "$http_code" == "503" ]]; then
            log_pass "$description (HTTP $http_code - degraded but reachable)"
        else
            log_pass "$description (HTTP $http_code)"
        fi
        if [[ "$VERBOSE" == "true" ]] && [[ -n "$body" ]]; then
            echo "  Response: ${body:0:200}..."
        fi
        return 0
    else
        log_fail "$description - Expected HTTP [$accepted_codes], got $http_code"
        if [[ -n "$body" ]]; then
            echo "  Response: ${body:0:200}"
        fi
        return 1
    fi
}

# Check if endpoint returns specific content
# Args: $1=endpoint, $2=expected_content, $3=description
check_endpoint_content() {
    local endpoint="$1"
    local expected_content="$2"
    local description="${3:-$endpoint}"
    local url="${API_BASE_URL}${endpoint}"

    local response
    response=$(curl -s --max-time "$TIMEOUT" "$url" 2>/dev/null) || {
        log_fail "$description - Connection failed"
        return 1
    }

    if echo "$response" | grep -q "$expected_content"; then
        log_pass "$description (content verified)"
        return 0
    else
        log_fail "$description - Expected content '$expected_content' not found"
        return 1
    fi
}

# Check alembic migration status (optional, inside container)
check_alembic() {
    local container_name="${CONTAINER_NAME:-codingbrain-mcp}"

    if ! command -v docker &> /dev/null; then
        log_warn "Docker not available - skipping alembic check"
        return 0
    fi

    if ! docker ps --format '{{.Names}}' | grep -q "$container_name"; then
        log_warn "Container '$container_name' not running - skipping alembic check"
        return 0
    fi

    local result
    result=$(docker exec "$container_name" alembic current 2>&1) || {
        log_warn "alembic current failed: $result"
        return 0
    }

    if echo "$result" | grep -q "(head)"; then
        log_pass "Alembic migration at head"
    else
        log_warn "Alembic migration may not be at head: $result"
    fi
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

main() {
    echo "=============================================="
    echo "  Coding Brain API Smoke Test"
    echo "=============================================="
    echo "API Base URL: $API_BASE_URL"
    echo "Timeout: ${TIMEOUT}s"
    echo ""

    check_curl

    echo "--- Health Endpoints ---"
    check_endpoint "/health/live" 200 "Liveness probe (/health/live)"

    # /health/deps may return 200 (all healthy) or 503 (degraded) - both are acceptable
    check_endpoint_accept_multiple "/health/deps" "200,503" "Dependency health (/health/deps)"

    check_endpoint "/mcp/health" 200 "MCP server health (/mcp/health)"

    echo ""
    echo "--- Observability ---"
    check_endpoint_content "/metrics" "http_requests_total" "Prometheus metrics (/metrics)"

    echo ""
    echo "--- Optional Checks ---"
    check_alembic

    echo ""
    echo "=============================================="
    echo "  Results: $PASSED passed, $FAILED failed, $WARNINGS warnings"
    echo "=============================================="

    if [[ $FAILED -gt 0 ]]; then
        echo -e "${RED}SMOKE TEST FAILED${NC}"
        exit 1
    else
        echo -e "${GREEN}SMOKE TEST PASSED${NC}"
        exit 0
    fi
}

main "$@"
