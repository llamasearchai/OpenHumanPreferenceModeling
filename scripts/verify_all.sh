#!/bin/bash
# =============================================================================
# OpenHumanPreferenceModeling - Unified Verification Script
# =============================================================================
# This script performs a full verification of the codebase, including:
# 1. Frontend: Linting, Type Checking, Build
# 2. Backend: Unit Tests, Integration Tests
# 3. End-to-End: Full system validation via Playwright
#
# Usage: ./scripts/verify_all.sh
# =============================================================================

set -euo pipefail

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
VENV_DIR="$PROJECT_ROOT/.venv"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

fail() {
    log_error "$1"
    exit 1
}

# Cleanup handler
cleanup() {
    log_info "Cleaning up..."
    # Kill any background processes we started
    if [ -n "${BACKEND_PID:-}" ]; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [ -n "${FRONTEND_PID:-}" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# =============================================================================
# 1. Frontend Verification
# =============================================================================
verify_frontend() {
    log_info "Step 1: Verifying Frontend..."
    cd "$FRONTEND_DIR"

    if [ ! -d "node_modules" ]; then
        log_warn "Installing frontend dependencies..."
        pnpm install
    fi

    log_info "Running Frontend Lint..."
    pnpm lint || fail "Frontend lint failed"

    log_info "Running Frontend Type Check..."
    pnpm type-check || fail "Frontend type check failed"

    log_info "Running Frontend Build..."
    pnpm build || fail "Frontend build failed"

    log_success "Frontend verification passed!"
}

# =============================================================================
# 2. Backend Verification
# =============================================================================
verify_backend() {
    log_info "Step 2: Verifying Backend..."
    cd "$PROJECT_ROOT"

    if [ ! -d "$VENV_DIR" ]; then
        fail "Virtual environment not found at $VENV_DIR. Please set it up."
    fi

    log_info "Running Backend Tests (pytest)..."
    # Skip E2E tests here, they are run separately
    "$VENV_DIR/bin/pytest" --ignore=tests/e2e || fail "Backend unit tests failed"

    log_success "Backend verification passed!"
}

# =============================================================================
# 3. End-to-End Verification
# =============================================================================
verify_e2e() {
    log_info "Step 3: Verifying End-to-End Flows..."
    cd "$PROJECT_ROOT"

    # Start Backend in Dev Mode
    log_info "Starting Backend (Dev Mode)..."
    export OHPM_DEV_MODE=true
    "$VENV_DIR/bin/uvicorn" main:app --host 0.0.0.0 --port 8000 > /dev/null 2>&1 &
    BACKEND_PID=$!
    log_info "Backend PID: $BACKEND_PID"

    # Start Frontend
    log_info "Starting Frontend..."
    cd "$FRONTEND_DIR"
    pnpm dev > /dev/null 2>&1 &
    FRONTEND_PID=$!
    log_info "Frontend PID: $FRONTEND_PID"

    # Wait for services
    log_info "Waiting for services to spin up (10s)..."
    sleep 10

    # Run Playwright Tests
    log_info "Running E2E Tests..."
    cd "$PROJECT_ROOT"
    "$VENV_DIR/bin/pytest" tests/e2e/ || fail "End-to-End tests failed"

    log_success "End-to-End verification passed!"
}

# =============================================================================
# Main Execution
# =============================================================================
main() {
    log_info "Starting Unified Codebase Verification..."
    
    verify_frontend
    verify_backend
    verify_e2e

    echo ""
    log_success "============================================================"
    log_success "  ALL SYSTEMS GO: Codebase verified successfully."
    log_success "============================================================"
}

main
