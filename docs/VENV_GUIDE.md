# Virtual Environment Guide

## ✅ What's Set Up

A dedicated virtual environment `venv_asr` has been created with all dependencies:
- ✅ torch 2.9.1 (with MPS support for Apple Silicon)
- ✅ transformers 4.57.3
- ✅ datasets 4.4.2
- ✅ librosa, soundfile (audio processing)
- ✅ evaluate, tensorboard (training/evaluation)
- ✅ jupyter, ipykernel (notebook support)
- ✅ pandas, matplotlib, seaborn (visualization)
- ✅ torchcodec (audio decoding)

## 🚀 How to Use

### Option 1: Quick Activation
```bash
# Activate the venv
source activate_venv.sh

# Now you can run any Python script
python asr_training_run.py
```

### Option 2: Manual Activation
```bash
# Activate
source venv_asr/bin/activate

# Verify
which python
python --version

# Run scripts
python asr_training_run.py
python scripts/download_model.py

# Deactivate when done
deactivate
```

### Option 3: Run Jupyter in venv
```bash
# Activate venv
source venv_asr/bin/activate

# Start Jupyter
jupyter notebook

# Open: asr_training_production.ipynb
# The notebook will use the venv's packages
```

## 📦 Installed Packages

Core ML:
- torch 2.9.1
- torchaudio 2.9.1
- transformers 4.57.3
- datasets 4.4.2
- accelerate 1.12.0
- evaluate 0.4.6

Audio:
- librosa 0.11.0
- soundfile 0.13.1
- torchcodec 0.9.1

Training:
- tensorboard 2.20.0
- safetensors 0.7.0

Visualization:
- matplotlib 3.10.8
- seaborn 0.13.2
- pandas 2.3.3

Jupyter:
- ipykernel 7.1.0
- jupyter-client 8.8.0

## 🎯 Quick Commands

```bash
# Activate and run training
source venv_asr/bin/activate && python asr_training_run.py

# Activate and download model
source venv_asr/bin/activate && python scripts/download_model.py

# Activate and start Jupyter
source venv_asr/bin/activate && jupyter notebook

# Activate and run standalone script
source venv_asr/bin/activate && python train_sample.py
```

## 🔧 Troubleshooting

### Venv Not Activating
```bash
# Make sure you're in the project directory
cd /Users/ismatsamadov/automatic_speech_recognition

# Then activate
source venv_asr/bin/activate
```

### Wrong Python Being Used
```bash
# After activation, verify:
which python
# Should show: .../venv_asr/bin/python

# If not, try:
deactivate
source venv_asr/bin/activate
```

### Missing Packages
```bash
# Activate venv first
source venv_asr/bin/activate

# Then install missing package
pip install package_name
```

### Recreate Venv
```bash
# Remove old venv
rm -rf venv_asr

# Recreate
python3 -m venv venv_asr
source venv_asr/bin/activate
pip install -r requirements.txt
pip install torchcodec seaborn matplotlib
```

## 📊 Current Training

Training is **currently running** in the venv with:
- Model: whisper-tiny (faster, easier to download)
- Samples: 500
- Device: Apple Silicon (MPS)

Monitor progress in outputs:
```bash
# Watch charts being generated
ls -lh charts/

# Check metrics
cat outputs/whisper_azerbaijani_*/validation.json
```

## 🎓 Best Practices

1. **Always activate venv before running scripts**
   ```bash
   source venv_asr/bin/activate
   ```

2. **Check which Python you're using**
   ```bash
   which python  # Should be in venv_asr/
   ```

3. **Install packages in venv only**
   ```bash
   # Activate first!
   source venv_asr/bin/activate
   pip install new_package
   ```

4. **Use requirements.txt for reproducibility**
   ```bash
   # Export current packages
   pip freeze > requirements_venv.txt
   ```

## 🔄 Updating Packages

```bash
# Activate venv
source venv_asr/bin/activate

# Update specific package
pip install --upgrade transformers

# Update all packages
pip install --upgrade -r requirements.txt
```

## 📝 Add Venv to Jupyter

```bash
# Activate venv
source venv_asr/bin/activate

# Install ipykernel (already installed)
# Add kernel to Jupyter
python -m ipykernel install --user --name=asr_venv --display-name="ASR (venv_asr)"

# Now in Jupyter:
# Kernel → Change Kernel → ASR (venv_asr)
```

---

**Created**: January 11, 2026
**Python**: 3.10
**Virtual Environment**: venv_asr
**Status**: ✅ Ready to Use
