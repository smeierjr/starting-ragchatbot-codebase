#!/bin/bash

# Type checking script for the RAG chatbot project
# Runs mypy for static type analysis

set -e

echo "🔎 Running type checking..."

echo "📊 Checking types with mypy..."
uv run mypy backend/ main.py

echo "✅ Type checking complete!"