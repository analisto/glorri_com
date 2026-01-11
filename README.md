# Azerbaijani Automatic Speech Recognition (ASR)

Production-ready end-to-end pipeline for training Whisper-based ASR models for Azerbaijani language.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-LocalDoc%2Fazerbaijani__asr-orange.svg)](https://huggingface.co/datasets/LocalDoc/azerbaijani_asr)

## 🚀 Quick Start

### Option 1: Complete Setup (Recommended)
```bash
# 1. Run environment setup
./scripts/setup_environment.sh

# 2. Download dependencies
python scripts/download_dependencies.py --model small

# 3. Start training
jupyter notebook asr_training_production.ipynb
```

### Option 2: Fast Testing (No Downloads)
```bash
# Install dependencies
pip install -r requirements.txt
pip install evaluate seaborn torchcodec

# Run with streaming (no download needed)
jupyter notebook asr_training_production.ipynb
# Keep SAMPLE_MODE=True and run all cells
```

## 📁 Project Structure

```
automatic_speech_recognition/
├── 📓 asr_training_production.ipynb    # Main production notebook
├── 📓 azerbaijani_asr_training.ipynb   # Original training notebook
├── 🐍 train_sample.py                  # Standalone training script
├── 📋 requirements.txt                 # Python dependencies
├── 📖 README.md                        # This file
├── 📖 README_PRODUCTION.md             # Detailed production guide
│
├── scripts/                            # 🛠️ Utility scripts
│   ├── download_data.py                # Dataset downloader
│   ├── download_model.py               # Model downloader
│   ├── download_dependencies.py        # Combined downloader
│   ├── setup_environment.sh            # Environment setup
│   └── README.md                       # Scripts documentation
│
├── charts/                             # 📊 Generated visualizations
│   ├── duration_distribution.png
│   ├── text_length_distribution.png
│   ├── training_loss_curve.png
│   ├── validation_wer_curve.png
│   └── results_summary.png
│
├── outputs/                            # 📈 Metrics and results
│   ├── *_config.json                   # Experiment configurations
│   ├── *_device_info.json              # Hardware information
│   ├── *_training_history.csv          # Training logs
│   ├── *_eval_history.csv              # Evaluation logs
│   ├── *_validation_results.json       # Validation metrics
│   ├── *_test_results.json             # Test metrics
│   └── *_sample_predictions.csv        # Example predictions
│
├── artifacts/                          # 💾 Trained models
│   └── {experiment_name}_final/
│       ├── config.json                 # Model configuration
│       ├── model.safetensors           # Model weights
│       ├── preprocessor_config.json    # Preprocessing config
│       ├── tokenizer_config.json       # Tokenizer config
│       ├── experiment_metadata.json    # Complete experiment info
│       └── README.md                   # Model documentation
│
├── data/                               # 💿 Dataset cache
│   └── dataset_cache/                  # Downloaded dataset
│
└── models/                             # 🤖 Model cache
    └── [huggingface model cache]       # Downloaded Whisper models
```

## ✨ Features

### Production Notebook
- ✅ **Complete ML Pipeline** (14 stages from data to deployment)
- ✅ **Industry Best Practices** (reproducibility, logging, versioning)
- ✅ **Automated Artifact Management** (charts, metrics, models)
- ✅ **Comprehensive Evaluation** (WER, sample predictions, visualizations)
- ✅ **Hardware Auto-Detection** (CPU/GPU/MPS)
- ✅ **Streaming Mode Support** (no download required for testing)
- ✅ **Train/Val/Test Splits** (80/10/10, no data leakage)
- ✅ **Fixed Random Seeds** (fully reproducible results)

### Scripts
- 🔽 **download_data.py** - Dataset downloader with retry logic
- 🔽 **download_model.py** - Pre-download Whisper models
- 🔽 **download_dependencies.py** - Download everything at once
- ⚙️ **setup_environment.sh** - Complete environment setup

## 📊 Dataset

**Source**: [LocalDoc/azerbaijani_asr](https://huggingface.co/datasets/LocalDoc/azerbaijani_asr)

| Metric | Value |
|--------|-------|
| Samples | 351,019 |
| Duration | ~334 hours |
| Size | 38.5 GB |
| Format | WAV (16kHz) |
| Language | Azerbaijani |
| License | CC-BY-NC-4.0 |

**Duration Distribution:**
- 0-2 sec: 36.1%
- 2-5 sec: 47.2%
- 5-10 sec: 14.6%
- 10-20 sec: 2.0%
- 20+ sec: 0.1%

## 🤖 Supported Models

| Model | Parameters | Size | Use Case |
|-------|-----------|------|----------|
| whisper-tiny | 39M | ~150 MB | Fast testing |
| whisper-base | 74M | ~290 MB | CPU training |
| **whisper-small** | **244M** | **~970 MB** | **Recommended** |
| whisper-medium | 769M | ~3 GB | GPU training |
| whisper-large-v2 | 1.5B | ~6 GB | Best accuracy |

## 📖 Usage

### 1. Setup Environment
```bash
# Option A: Automated setup
./scripts/setup_environment.sh

# Option B: Manual setup
pip install -r requirements.txt
pip install evaluate seaborn torchcodec
mkdir -p data charts outputs artifacts models
```

### 2. Download Resources
```bash
# Option A: Download everything
python scripts/download_dependencies.py --model small

# Option B: Dataset only
python scripts/download_data.py

# Option C: Model only
python scripts/download_model.py --model openai/whisper-small

# Option D: Use streaming (no downloads)
# Just run the notebook with SAMPLE_MODE=True
```

### 3. Train Model
```bash
# Option A: Production notebook (recommended)
jupyter notebook asr_training_production.ipynb
# Set SAMPLE_MODE=True for testing or False for full training

# Option B: Standalone script
python train_sample.py
```

### 4. Use Trained Model
```python
from transformers import pipeline

# Load model
pipe = pipeline(
    "automatic-speech-recognition",
    model="./artifacts/{experiment_name}_final"
)

# Transcribe audio
result = pipe("audio.wav")
print(result["text"])
```

## 🎯 Training Modes

### Sample Mode (Testing)
- **Samples**: 500
- **Duration**: ~10-20 minutes
- **Hardware**: CPU OK
- **Purpose**: Quick testing, development
- **Config**: `SAMPLE_MODE=True`

### Full Mode (Production)
- **Samples**: 351,019
- **Duration**: Several hours
- **Hardware**: GPU recommended
- **Purpose**: Production model
- **Config**: `SAMPLE_MODE=False`

## 📈 Expected Results

| Metric | Sample Mode | Full Mode |
|--------|-------------|-----------|
| WER (Validation) | 25-35% | 15-25% |
| WER (Test) | 25-35% | 15-25% |
| Training Time | 10-20 min | 3-8 hours |
| Hardware | CPU | GPU |

Lower WER = Better (0% = perfect transcription)

## 🔧 Configuration

Edit notebook Cell 1 or modify `CONFIG` dict:

```python
CONFIG = {
    # Mode
    "sample_mode": True,              # True=testing, False=production
    "sample_size": 500,                # Samples in sample mode

    # Model
    "model_name": "openai/whisper-small",
    "language": "azerbaijani",

    # Training
    "batch_size": 8,
    "num_epochs": 3,
    "learning_rate": 1e-5,

    # Reproducibility
    "random_seed": 42,

    # Splits
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
}
```

## 📊 Generated Outputs

After training, you'll find:

### Charts (`/charts`)
- Duration/text distributions
- Training loss curves
- Validation WER curves
- Results summary dashboard

### Metrics (`/outputs`)
- Configuration JSONs
- Training/eval history (CSV)
- Validation/test results
- Sample predictions
- Data validation reports

### Models (`/artifacts`)
- Complete trained model
- Preprocessor & tokenizer
- Experiment metadata
- Model README

## 💻 Hardware Requirements

| Mode | CPU | RAM | GPU | Disk |
|------|-----|-----|-----|------|
| Sample | ✅ Any | 8GB | ❌ Not needed | 5GB |
| Full | ⚠️ Slow | 16GB | ✅ 8GB+ VRAM | 50GB |

**Supported Devices:**
- CUDA GPUs (NVIDIA)
- Apple Silicon (MPS)
- CPU (slow for full training)

## 🐛 Troubleshooting

### Network/SSL Issues
```bash
# Scripts include SSL bypass for corporate networks
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_SSL_VERIFY=1
```

### Out of Memory
```python
# Reduce batch size
CONFIG["batch_size"] = 4  # or 2

# Or use smaller model
CONFIG["model_name"] = "openai/whisper-tiny"
```

### Slow Training
- Use sample mode for testing
- Enable GPU if available
- Use smaller model (whisper-tiny)
- Reduce sample_size

See [docs/README_PRODUCTION.md](docs/README_PRODUCTION.md) for detailed troubleshooting.

## 📚 Documentation

- **[docs/](docs/)** - Complete documentation index
- **[docs/README_PRODUCTION.md](docs/README_PRODUCTION.md)** - Detailed production guide
- **[docs/SCRIPTS.md](docs/SCRIPTS.md)** - Scripts documentation
- **[Notebook](asr_training_production.ipynb)** - Inline documentation

## 🔬 Reproducibility

All experiments are fully reproducible:
- ✅ Fixed random seeds
- ✅ Version-controlled configurations
- ✅ Complete environment logs
- ✅ Deterministic data splits

To reproduce:
1. Use same `random_seed`
2. Use same configuration
3. Follow same data preprocessing steps

## 📄 License

- **Code**: MIT License
- **Dataset**: CC-BY-NC-4.0 (non-commercial use only)
- **Model**: OpenAI Whisper License

## 🙏 Acknowledgments

- **Dataset**: [LocalDoc/azerbaijani_asr](https://huggingface.co/datasets/LocalDoc/azerbaijani_asr)
- **Model**: [OpenAI Whisper](https://github.com/openai/whisper)
- **Framework**: [Hugging Face Transformers](https://github.com/huggingface/transformers)

## 📞 Support

For issues or questions:
1. Check the documentation (README_PRODUCTION.md, scripts/README.md)
2. Review troubleshooting sections
3. Check original dataset/model repositories

## 🗺️ Roadmap

- [ ] Add data augmentation
- [ ] Support for other Whisper variants
- [ ] Model quantization for deployment
- [ ] Real-time inference support
- [ ] Multi-GPU training
- [ ] Distributed training support

---

**Version**: 1.0
**Last Updated**: January 11, 2026
**Python**: 3.10+
**Status**: Production Ready ✅
