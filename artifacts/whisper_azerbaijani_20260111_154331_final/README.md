# Azerbaijani ASR Model

## Experiment: whisper_azerbaijani_20260111_154331

### Model Information
- Base Model: openai/whisper-tiny
- Language: azerbaijani
- Task: transcribe
- Parameters: 37,760,640

### Performance
- Validation WER: 59.70%
- Test WER: 59.28%

### Training Details
- Training Samples: 400
- Validation Samples: 50
- Test Samples: 50
- Epochs: 2.00
- Training Time: 5.01 hours
- Device: Apple Silicon (MPS)

### Usage

```python
from transformers import pipeline

# Load the model
pipe = pipeline(
    "automatic-speech-recognition",
    model="artifacts/whisper_azerbaijani_20260111_154331_final"
)

# Transcribe audio
result = pipe("path/to/audio.wav")
print(result["text"])
```

### Files
- `config.json` - Model configuration
- `preprocessor_config.json` - Audio preprocessing config
- `tokenizer_config.json` - Tokenizer configuration
- `model.safetensors` - Model weights
- `experiment_metadata.json` - Complete experiment details

### Reproducibility
- Random Seed: 42
- All configuration and results saved in experiment_metadata.json

Generated on: 2026-01-11 20:46:11
