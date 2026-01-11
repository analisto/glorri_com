# Scripts Directory

Utility scripts for downloading and setting up the ASR training environment.

## Scripts

### 1. download_data.py
Downloads the Azerbaijani ASR dataset from Hugging Face.

**Usage:**
```bash
# Download full dataset (38.5 GB)
python scripts/download_data.py

# Use streaming mode (downloads as needed)
python scripts/download_data.py --streaming
```

**Features:**
- Retry logic for network issues
- SSL bypass for corporate networks
- Streaming mode support
- Progress tracking

### 2. download_model.py
Pre-downloads Whisper models to avoid issues during training.

**Usage:**
```bash
# Download whisper-small (default)
python scripts/download_model.py

# Download specific model
python scripts/download_model.py --model openai/whisper-tiny
python scripts/download_model.py --model openai/whisper-base
python scripts/download_model.py --model openai/whisper-medium

# Specify cache directory
python scripts/download_model.py --cache-dir ./my_models
```

**Available Models:**
- `whisper-tiny` - 39M params (~150 MB)
- `whisper-base` - 74M params (~290 MB)
- `whisper-small` - 244M params (~970 MB) **[recommended]**
- `whisper-medium` - 769M params (~3 GB)
- `whisper-large-v2` - 1.5B params (~6 GB)

### 3. download_dependencies.py
Downloads both dataset and model in one go.

**Usage:**
```bash
# Download everything with defaults
python scripts/download_dependencies.py

# Download specific model with streaming dataset
python scripts/download_dependencies.py --model tiny --dataset streaming

# Skip model download
python scripts/download_dependencies.py --skip-model

# Skip dataset download
python scripts/download_dependencies.py --skip-dataset
```

**Options:**
- `--model` - Whisper model (tiny/base/small/medium)
- `--dataset` - full or streaming
- `--skip-model` - Skip model download
- `--skip-dataset` - Skip dataset download

### 4. setup_environment.sh
Complete environment setup script (bash).

**Usage:**
```bash
# Make executable
chmod +x scripts/setup_environment.sh

# Run setup
./scripts/setup_environment.sh
```

**What it does:**
1. Checks Python version
2. Installs all pip dependencies
3. Creates project directories
4. Detects GPU/CPU
5. Optionally downloads model

## Quick Start Workflow

### Option 1: Full Setup
```bash
# 1. Setup environment
./scripts/setup_environment.sh

# 2. Download everything
python scripts/download_dependencies.py --model small

# 3. Start training
jupyter notebook asr_training_production.ipynb
```

### Option 2: Quick Testing
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Use streaming (no download needed)
python train_sample.py

# Or run notebook in sample mode
jupyter notebook asr_training_production.ipynb
```

### Option 3: Minimal Downloads
```bash
# Just download model, use streaming for data
python scripts/download_model.py --model openai/whisper-small
python train_sample.py
```

## Troubleshooting

### SSL/Network Errors
All scripts include SSL bypass for corporate networks. If you still get errors:
```bash
# Set environment variables
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_SSL_VERIFY=1

# Then run scripts
python scripts/download_data.py
```

### Download Interrupted
Scripts include retry logic. If download fails partway:
```bash
# Run again - it will resume from where it left off
python scripts/download_data.py
```

### Disk Space
- Full dataset: 38.5 GB
- Whisper-small model: ~1 GB
- Total: ~40 GB minimum

Check available space:
```bash
df -h .
```

### Memory Issues
If download fails due to memory:
```bash
# Use streaming mode instead
python scripts/download_data.py --streaming
```

## File Organization

After running setup scripts:
```
automatic_speech_recognition/
├── scripts/
│   ├── download_data.py          # Dataset downloader
│   ├── download_model.py          # Model downloader
│   ├── download_dependencies.py   # Combined downloader
│   ├── setup_environment.sh       # Environment setup
│   └── README.md                  # This file
├── data/                          # Dataset cache
├── models/                        # Model cache
├── charts/                        # Visualizations
├── outputs/                       # Metrics
└── artifacts/                     # Trained models
```

## Dependencies

Required packages (installed by setup_environment.sh):
- torch >= 2.9.0
- transformers >= 4.57.0
- datasets >= 4.4.0
- librosa >= 0.11.0
- evaluate >= 0.4.0
- pandas, matplotlib, seaborn
- jupyter

## Support

For issues:
1. Check the main README_PRODUCTION.md
2. Review error messages carefully
3. Try with smaller model (whisper-tiny)
4. Use streaming mode for data

---

**Last Updated**: January 11, 2026
