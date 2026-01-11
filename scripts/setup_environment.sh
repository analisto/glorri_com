#!/bin/bash

# Setup Environment for Azerbaijani ASR Training
# This script installs all necessary dependencies

set -e  # Exit on error

echo "======================================================================"
echo "Azerbaijani ASR - Environment Setup"
echo "======================================================================"

# Check Python version
echo -e "\n1. Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $python_version"

required_version="3.10"
if [[ $(echo -e "$python_version\n$required_version" | sort -V | head -n1) != "$required_version" ]]; then
    echo "   ⚠️  Warning: Python 3.10+ recommended (you have $python_version)"
else
    echo "   ✓ Python version OK"
fi

# Install pip packages
echo -e "\n2. Installing Python dependencies..."
echo "   This may take a few minutes..."

pip install --upgrade pip > /dev/null 2>&1

# Core ML packages
echo "   - Installing core ML packages..."
pip install torch>=2.9.0 torchaudio>=2.9.0 \
    transformers>=4.57.0 datasets>=4.4.0 \
    accelerate>=1.12.0 evaluate>=0.4.0 \
    --quiet || echo "   ⚠️  Some packages may need manual installation"

# Audio processing
echo "   - Installing audio processing packages..."
pip install librosa>=0.11.0 soundfile>=0.13.0 \
    jiwer>=4.0.0 torchcodec \
    --quiet || echo "   ⚠️  Audio packages may need manual installation"

# Training utilities
echo "   - Installing training utilities..."
pip install tensorboard>=2.20.0 safetensors>=0.7.0 \
    --quiet || echo "   ⚠️  Utility packages may need manual installation"

# Jupyter and visualization
echo "   - Installing Jupyter and visualization packages..."
pip install ipykernel>=7.1.0 jupyter \
    pandas matplotlib seaborn \
    --quiet || echo "   ⚠️  Visualization packages may need manual installation"

echo "   ✓ Dependencies installed"

# Create project directories
echo -e "\n3. Creating project directories..."
mkdir -p data
mkdir -p charts
mkdir -p outputs
mkdir -p artifacts
mkdir -p models
echo "   ✓ Directories created"

# Check for GPU
echo -e "\n4. Checking for GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'   ✓ GPU detected: {torch.cuda.get_device_name(0)}')
    print(f'   ✓ CUDA version: {torch.version.cuda}')
elif torch.backends.mps.is_available():
    print('   ✓ Apple Silicon (MPS) detected')
else:
    print('   ℹ️  No GPU detected - will use CPU (training will be slower)')
"

# Download model (optional)
echo -e "\n5. Pre-download Whisper model? (optional)"
read -p "   Download openai/whisper-small now? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   Downloading model..."
    python3 scripts/download_model.py --model openai/whisper-small
else
    echo "   Skipping model download (will download during training)"
fi

# Summary
echo -e "\n======================================================================"
echo "Setup Complete!"
echo "======================================================================"
echo -e "\nNext steps:"
echo "  1. Review the production notebook:"
echo "     jupyter notebook asr_training_production.ipynb"
echo ""
echo "  2. Or download the dataset:"
echo "     python scripts/download_data.py"
echo ""
echo "  3. Or run sample training:"
echo "     python train_sample.py"
echo ""
echo "======================================================================"
