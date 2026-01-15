#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="${ROOT_DIR}/frontend"

cd "${FRONTEND_DIR}"

echo "[INFO] Running frontend critique pipeline (unit + build + render audit)..."

pnpm lint
pnpm type-check
pnpm test
pnpm build
pnpm test:e2e:render-audit

echo "[SUCCESS] Frontend critique pipeline completed successfully."

