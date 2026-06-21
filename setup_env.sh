#!/usr/bin/env bash
# FlashEdge — Environment Setup Script
# Usage: bash setup_env.sh

set -e

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"

echo "============================================"
echo "  FlashEdge Environment Setup"
echo "============================================"
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
$PYTHON --version
PYTHON_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "       Using Python $PYTHON_VERSION"
echo ""

# Create virtual environment
echo "[2/5] Creating virtual environment at $VENV_DIR..."
if [ -d "$VENV_DIR" ]; then
    echo "       Virtual environment already exists. Skipping creation."
else
    $PYTHON -m venv "$VENV_DIR"
    echo "       Done."
fi
echo ""

# Activate virtual environment
echo "[3/5] Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "       Activated: $(which python)"
echo ""

# Upgrade pip
echo "[4/5] Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet
echo "       Done."
echo ""

# Install FlashEdge with all dependencies
echo "[5/5] Installing FlashEdge in editable mode..."
pip install -e ".[all,dev]" --quiet
echo "       Done."
echo ""

echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Activate the environment with:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  Verify installation:"
echo "    flashedge version"
echo "============================================"
