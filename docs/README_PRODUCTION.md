# Azerbaijani ASR Production Training Pipeline

## Overview
Complete production-ready pipeline for training an Automatic Speech Recognition model for Azerbaijani language using Whisper architecture.

## Project Structure
```
automatic_speech_recognition/
├── asr_training_production.ipynb    # Main production notebook
├── download_data.py                 # Dataset downloader
├── train_sample.py                  # Standalone training script
├── requirements.txt                 # Python dependencies
├── charts/                          # Generated visualizations (auto-created)
├── outputs/                         # Metrics and evaluation results (auto-created)
├── artifacts/                       # Trained models and configs (auto-created)
└── data/                           # Dataset cache (auto-created)
```

## Features

### Production Notebook (`asr_training_production.ipynb`)
✓ **Complete ML Pipeline** with 14 stages:
1. Environment setup and configuration
2. Hardware detection (CPU/GPU/MPS)
3. Data loading with streaming support
4. Data validation and schema checks
5. Exploratory data analysis with visualizations
6. Train/validation/test splits (80/10/10)
7. Model loading (Whisper)
8. Data preprocessing
9. Model training
10. Comprehensive evaluation
11. Training visualizations
12. Model persistence
13. Inference testing
14. Final summary report

✓ **Industry Best Practices**:
- Centralized configuration
- Fixed random seeds for reproducibility
- No data leakage between splits
- Comprehensive logging and metrics
- Modular, documented code
- Clean cell organization

✓ **Automated Artifact Management**:
- All charts saved to `/charts`
- All metrics/tables saved to `/outputs`
- All models/configs saved to `/artifacts`
- Complete experiment metadata

✓ **Generated Outputs**:
- Duration distribution charts
- Text length distribution charts
- Training loss curves
- Validation WER curves
- Results summary visualization
- Training/evaluation history (CSV)
- Sample predictions
- Model README
- Complete experiment metadata (JSON)

## Quick Start

### Option 1: Run Production Notebook (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt
pip install evaluate seaborn  # Additional notebook dependencies

# Start Jupyter
jupyter notebook asr_training_production.ipynb

# Run all cells (Ctrl+A, then Shift+Enter)
```

### Option 2: Run Standalone Script
```bash
# Download dataset first
python download_data.py

# Run training
python train_sample.py
```

## Configuration

### Toggle Training Mode
Edit Cell 1 in `asr_training_production.ipynb`:

```python
CONFIG = {
    "sample_mode": True,   # Set to False for full training
    "sample_size": 500,    # Samples for testing
    ...
}
```

**Sample Mode** (default):
- 500 samples
- 1 epoch
- Quick testing (~10-20 minutes)
- Perfect for development

**Full Training Mode**:
- All 351K samples
- 3 epochs
- Production model (~several hours)
- Requires GPU for reasonable speed

### Key Hyperparameters
```python
"batch_size": 8,
"learning_rate": 1e-5,
"num_epochs": 3,
"warmup_steps": 500,
"gradient_accumulation_steps": 2,
"random_seed": 42,  # For reproducibility
```

## Dataset

**Source**: LocalDoc/azerbaijani_asr
**Size**: 351,019 samples, ~334 hours
**Format**: WAV (16kHz)
**License**: CC-BY-NC-4.0 (non-commercial)
**Language**: Azerbaijani

**Duration Distribution**:
- 0-2 sec: 36.1%
- 2-5 sec: 47.2%
- 5-10 sec: 14.6%
- 10-20 sec: 2.0%
- 20+ sec: 0.1%

## Model

**Base Model**: openai/whisper-small
**Parameters**: ~244M
**Architecture**: Encoder-decoder transformer
**Task**: Transcription
**Language**: Azerbaijani

**Alternative Models** (edit CONFIG["model_name"]):
- `openai/whisper-tiny` (39M params, fastest)
- `openai/whisper-base` (74M params)
- `openai/whisper-small` (244M params, recommended)
- `openai/whisper-medium` (769M params, requires GPU)

## Generated Artifacts

### Charts (`/charts`)
- `duration_distribution.png` - Audio duration analysis
- `text_length_distribution.png` - Transcription length analysis
- `training_loss_curve.png` - Training loss over time
- `validation_wer_curve.png` - Validation WER over time
- `training_overview.png` - Combined training metrics
- `results_summary.png` - Final results visualization

### Outputs (`/outputs`)
- `{experiment_name}_config.json` - Complete configuration
- `{experiment_name}_device_info.json` - Hardware info
- `{experiment_name}_validation.json` - Data validation results
- `{experiment_name}_split_info.json` - Split statistics
- `{experiment_name}_model_info.json` - Model metadata
- `{experiment_name}_training_history.csv` - Training logs
- `{experiment_name}_eval_history.csv` - Evaluation logs
- `{experiment_name}_validation_results.json` - Val metrics
- `{experiment_name}_test_results.json` - Test metrics
- `{experiment_name}_sample_predictions.csv` - Example predictions
- `{experiment_name}_data_summary.csv` - Data statistics

### Model Artifacts (`/artifacts`)
- `{experiment_name}_final/` - Complete trained model
  - `config.json` - Model configuration
  - `preprocessor_config.json` - Audio preprocessing
  - `tokenizer_config.json` - Tokenizer settings
  - `model.safetensors` - Model weights
  - `experiment_metadata.json` - Full experiment info
  - `README.md` - Model documentation

## Usage After Training

### Load and Use Trained Model
```python
from transformers import pipeline

# Load the model
pipe = pipeline(
    "automatic-speech-recognition",
    model="./artifacts/{experiment_name}_final"
)

# Transcribe audio
result = pipe("path/to/audio.wav")
print(result["text"])

# Or with audio array
import librosa
audio, sr = librosa.load("audio.wav", sr=16000)
result = pipe(audio)
print(result["text"])
```

### Evaluation Metrics
- **WER (Word Error Rate)**: Lower is better (0% = perfect)
- Typical results: 15-25% WER on validation set
- Final test set WER provided in outputs

## Reproducibility

All experiments are fully reproducible:
- Fixed random seed: 42
- Environment info saved
- Complete configuration logged
- Git commit hash (if in repo)
- All hyperparameters documented

To reproduce an experiment:
1. Load the config JSON from `/outputs`
2. Set same random seed
3. Use same data splits
4. Run with same hyperparameters

## Troubleshooting

### Network/SSL Issues
The notebook includes SSL bypass for corporate networks. If downloads fail:
```python
# In the SSL configuration cell, ensure:
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
```

### Out of Memory
If training fails with OOM:
- Reduce `batch_size` (try 4 or 2)
- Increase `gradient_accumulation_steps`
- Use smaller model (whisper-tiny or whisper-base)
- Enable `fp16=True` if using GPU

### Slow Training
- Ensure GPU is detected (check Cell 2 output)
- Reduce `sample_size` for testing
- Use `whisper-tiny` for faster iterations

## Requirements

**Python**: 3.10+
**Key Dependencies**:
- torch >= 2.9.0
- transformers >= 4.57.0
- datasets >= 4.4.0
- evaluate >= 0.4.0
- librosa >= 0.11.0
- pandas, matplotlib, seaborn
- jupyter

**Hardware**:
- **Minimum**: 8GB RAM, CPU (very slow)
- **Recommended**: 16GB RAM, GPU with 8GB+ VRAM
- **Sample Mode**: Works on CPU
- **Full Training**: Requires GPU

## License

Code: MIT
Dataset: CC-BY-NC-4.0 (non-commercial use only)
Model: OpenAI Whisper license

## Citation

```bibtex
@dataset{azerbaijani_asr,
  title={Azerbaijani ASR Dataset},
  author={LocalDoc},
  year={2024},
  doi={10.57967/hf/6048},
  url={https://huggingface.co/datasets/LocalDoc/azerbaijani_asr}
}
```

## Next Steps

1. **Review Notebook**: Open `asr_training_production.ipynb` in Jupyter
2. **Run Sample Training**: Keep `SAMPLE_MODE=True` and run all cells
3. **Review Outputs**: Check `/charts` and `/outputs` directories
4. **Full Training**: Set `SAMPLE_MODE=False` and retrain
5. **Deploy Model**: Use the model from `/artifacts` for inference

## Support

For issues with:
- **Dataset**: https://huggingface.co/datasets/LocalDoc/azerbaijani_asr
- **Whisper Model**: https://github.com/openai/whisper
- **Transformers**: https://github.com/huggingface/transformers

---

**Created**: January 2026
**Last Updated**: January 11, 2026
**Version**: 1.0
