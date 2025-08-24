#!/bin/bash

# Code formatting script for the RAG chatbot project
# Runs black and isort on the codebase

set -e

echo "🔧 Running code formatting..."

echo "📝 Formatting with black..."
uv run black backend/ main.py

echo "🔄 Sorting imports with isort..."
uv run isort backend/ main.py

echo "✅ Code formatting complete!"