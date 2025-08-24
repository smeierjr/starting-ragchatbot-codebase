#!/bin/bash

# Linting script for the RAG chatbot project
# Runs flake8 for code quality checks

set -e

echo "🔍 Running code linting..."

echo "📋 Checking code style with flake8..."
uv run flake8 backend/ main.py

echo "✅ Linting complete!"