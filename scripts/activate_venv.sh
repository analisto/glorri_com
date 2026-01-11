#!/bin/bash
# Activate the ASR virtual environment

echo "======================================================================"
echo "Activating ASR Virtual Environment"
echo "======================================================================"

# Activate venv
source venv_asr/bin/activate

echo "✓ Virtual environment activated: venv_asr"
echo ""
echo "Python: $(which python)"
echo "Version: $(python --version)"
echo ""
echo "To deactivate, type: deactivate"
echo "======================================================================"
