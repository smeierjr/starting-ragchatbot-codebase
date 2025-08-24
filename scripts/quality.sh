#!/bin/bash

# Complete code quality script for the RAG chatbot project
# Runs all quality checks in sequence

set -e

echo "🚀 Running complete code quality checks..."

# Format code
echo "1️⃣ Formatting code..."
./scripts/format.sh

# Lint code
echo "2️⃣ Linting code..."
./scripts/lint.sh

# Type check
echo "3️⃣ Type checking..."
./scripts/typecheck.sh

echo "🎉 All quality checks complete!"