#!/bin/bash

# Test script to verify API endpoints are working
# Run this before starting the Next.js frontend

API_URL="${1:-http://localhost:8000}"

echo "Testing API connection at: $API_URL"
echo ""

# Test health endpoint
echo "1. Testing /health endpoint..."
HEALTH=$(curl -s "$API_URL/health")
if [ $? -eq 0 ]; then
  echo "✓ Health check passed: $HEALTH"
else
  echo "✗ Health check failed"
  exit 1
fi
echo ""

# Test category-names endpoint
echo "2. Testing /category-names endpoint..."
CATEGORIES=$(curl -s "$API_URL/category-names" | head -c 100)
if [ $? -eq 0 ]; then
  echo "✓ Category names endpoint working (showing first 100 chars): $CATEGORIES..."
else
  echo "✗ Category names endpoint failed"
  exit 1
fi
echo ""

# Test classify endpoint
echo "3. Testing /classify endpoint..."
CLASSIFY=$(curl -s -X POST "$API_URL/classify" \
  -H "Content-Type: application/json" \
  -d '{"title": "Test product", "description": "Test description"}')
if [ $? -eq 0 ]; then
  echo "✓ Classify endpoint working: $CLASSIFY"
else
  echo "✗ Classify endpoint failed"
  exit 1
fi
echo ""

# Test testset endpoint
echo "4. Testing /testset endpoint..."
TESTSET=$(curl -s -I "$API_URL/testset" | head -1)
if [ $? -eq 0 ]; then
  echo "✓ Testset endpoint working: $TESTSET"
else
  echo "✗ Testset endpoint failed"
  exit 1
fi
echo ""

echo "All API endpoints are working! You can now start the Next.js frontend."
echo "Run: npm run dev"

