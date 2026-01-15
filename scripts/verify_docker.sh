#!/bin/bash
set -e

# Cleanup previous runs
echo "Cleaning up..."
docker-compose down -v --remove-orphans

# Build containers
echo "Building containers..."
docker-compose build

# Start services
echo "Starting services..."
docker-compose up -d

# Wait for health check
echo "Waiting for services to be ready..."
sleep 15
docker-compose ps

echo "Verifying backend health..."
curl -v --fail http://localhost:8000/api/health

echo "Verifying frontend..."
curl -v --fail http://localhost:8080/ | grep "<title>"

echo "Running E2E tests against Dockerized environment..."
# Run Playwright tests pointing to the dockerized frontend
# We use custom env vars to tell Playwright where to test
export BASE_URL=http://localhost:8080
export API_URL=http://localhost:8000

cd frontend
npx playwright test

echo "Verification COMPLETE!"
echo "To tear down: docker-compose down"
