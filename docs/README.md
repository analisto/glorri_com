# Documentation Index

Complete documentation for the Azerbaijani ASR Training Pipeline.

## 📚 Documentation Files

### 1. [Production Guide](README_PRODUCTION.md)
**Complete end-to-end production guide**

Topics covered:
- Project structure and overview
- Complete ML pipeline (14 stages)
- Best practices implementation
- Generated artifacts and outputs
- Configuration options
- Dataset details
- Model information
- Usage instructions
- Troubleshooting guide

**Audience**: Data scientists, ML engineers
**When to read**: Before starting production training

---

### 2. [Scripts Documentation](SCRIPTS.md)
**Utility scripts reference**

Topics covered:
- download_data.py - Dataset downloader
- download_model.py - Model downloader
- download_dependencies.py - Combined downloader
- setup_environment.sh - Environment setup
- Usage examples
- Troubleshooting downloads

**Audience**: DevOps, ML engineers
**When to read**: During environment setup

---

## 🗂️ Quick Reference

### Getting Started
1. Start with main [README.md](../README.md) in project root
2. Review [Production Guide](README_PRODUCTION.md) for details
3. Check [Scripts Documentation](SCRIPTS.md) for download utilities

### Common Tasks

**Setup Environment**
```bash
./scripts/setup_environment.sh
```
See: [Scripts Documentation](SCRIPTS.md#4-setup_environmentsh)

**Download Resources**
```bash
python scripts/download_dependencies.py
```
See: [Scripts Documentation](SCRIPTS.md#3-download_dependenciespy)

**Train Model**
```bash
jupyter notebook asr_training_production.ipynb
```
See: [Production Guide](README_PRODUCTION.md#quick-start)

**Use Trained Model**
```python
from transformers import pipeline
pipe = pipeline("automatic-speech-recognition", model="./artifacts/{name}_final")
```
See: [Production Guide](README_PRODUCTION.md#usage-after-training)

## 📖 Documentation Structure

```
docs/
├── README.md                  # This file (documentation index)
├── README_PRODUCTION.md       # Complete production guide
└── SCRIPTS.md                 # Scripts reference
```

## 🔍 Find Information

| Topic | Document | Section |
|-------|----------|---------|
| **Setup** |
| Environment setup | [SCRIPTS.md](SCRIPTS.md) | setup_environment.sh |
| Dependencies | [SCRIPTS.md](SCRIPTS.md) | download_dependencies.py |
| **Training** |
| Quick start | [README_PRODUCTION.md](README_PRODUCTION.md) | Quick Start |
| Configuration | [README_PRODUCTION.md](README_PRODUCTION.md) | Configuration |
| Sample vs Full | [README_PRODUCTION.md](README_PRODUCTION.md) | Training Modes |
| **Dataset** |
| Dataset info | [README_PRODUCTION.md](README_PRODUCTION.md) | Dataset |
| Download | [SCRIPTS.md](SCRIPTS.md) | download_data.py |
| Validation | [README_PRODUCTION.md](README_PRODUCTION.md) | Data Validation |
| **Model** |
| Model selection | [README_PRODUCTION.md](README_PRODUCTION.md) | Model |
| Download | [SCRIPTS.md](SCRIPTS.md) | download_model.py |
| Usage | [README_PRODUCTION.md](README_PRODUCTION.md) | Usage After Training |
| **Results** |
| Generated files | [README_PRODUCTION.md](README_PRODUCTION.md) | Generated Artifacts |
| Metrics | [README_PRODUCTION.md](README_PRODUCTION.md) | Evaluation Metrics |
| Charts | [README_PRODUCTION.md](README_PRODUCTION.md) | Training Visualizations |
| **Troubleshooting** |
| Common issues | [README_PRODUCTION.md](README_PRODUCTION.md) | Troubleshooting |
| Download issues | [SCRIPTS.md](SCRIPTS.md) | Troubleshooting |
| Network/SSL | [README_PRODUCTION.md](README_PRODUCTION.md) | Network Issues |

## 🎯 By Role

### Data Scientist / ML Engineer
1. [README_PRODUCTION.md](README_PRODUCTION.md) - Full pipeline guide
2. Main [README.md](../README.md) - Quick reference
3. [SCRIPTS.md](SCRIPTS.md) - Setup tools

### DevOps / Infrastructure
1. [SCRIPTS.md](SCRIPTS.md) - Setup and downloads
2. Main [README.md](../README.md) - Requirements
3. [README_PRODUCTION.md](README_PRODUCTION.md) - Hardware specs

### Research / Academic
1. [README_PRODUCTION.md](README_PRODUCTION.md) - Reproducibility
2. Main [README.md](../README.md) - Dataset & model info
3. [SCRIPTS.md](SCRIPTS.md) - Download utilities

## 📝 Additional Resources

### External Documentation
- [Hugging Face Datasets](https://huggingface.co/docs/datasets) - Dataset library
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) - Model library
- [OpenAI Whisper](https://github.com/openai/whisper) - Whisper documentation
- [LocalDoc Dataset](https://huggingface.co/datasets/LocalDoc/azerbaijani_asr) - Dataset page

### Code Examples
All notebooks contain inline documentation:
- `asr_training_production.ipynb` - Production notebook with markdown cells
- `azerbaijani_asr_training.ipynb` - Original training notebook

## 🔄 Updates

Documentation is version-controlled with the code.

**Current Version**: 1.0
**Last Updated**: January 11, 2026

To see changes:
```bash
git log -- docs/
```

## 💡 Contributing

To improve documentation:
1. Edit relevant markdown files
2. Update this index if adding new docs
3. Keep formatting consistent
4. Test all code examples

---

**Need Help?**
- Review main [README.md](../README.md) first
- Check [Production Guide](README_PRODUCTION.md) for details
- See [Scripts Documentation](SCRIPTS.md) for utilities
- Search for keywords in documentation
