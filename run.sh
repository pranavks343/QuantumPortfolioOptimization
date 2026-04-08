#!/bin/bash
set -e

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt -q

# Copy .env from example if .env doesn't exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "⚠️  .env file created. Please add your API keys to .env before continuing."
    echo "    Required: OPENAI_API_KEY or ANTHROPIC_API_KEY"
    echo ""
    exit 1
fi

# Launch the app
echo "Starting app..."
streamlit run app.py
