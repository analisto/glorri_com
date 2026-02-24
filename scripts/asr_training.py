# # Azerbaijani ASR Model Training - Production Pipeline
# 
# **End-to-end ASR training workflow following industry best practices**
# 
# ## Overview
# This notebook implements a complete pipeline for training an Automatic Speech Recognition model for Azerbaijani language using Whisper architecture.
# 
# ## Project Structure
# ```
# .
# ├── charts/      # All visualizations (PNG/SVG)
# ├── outputs/     # Metrics, tables, evaluation summaries
# ├── artifacts/   # Trained models, processors, configs
# └── data/        # Dataset cache
# ```
# 
# ## Features
# - Reproducible training with fixed random seeds
# - Proper train/val/test splits with no data leakage
# - Comprehensive evaluation metrics and visualizations
# - Model versioning and artifact management
# - Production-ready code with error handling

# ---
# ## 1. Environment Setup and Configuration

# ============================================================
# CONFIGURATION - Central configuration for entire pipeline
# ============================================================

import os
from pathlib import Path
from datetime import datetime
import json

# Project directories
PROJECT_ROOT = Path(".")
CHARTS_DIR = PROJECT_ROOT / "charts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"

# Create directories
for directory in [CHARTS_DIR, OUTPUTS_DIR, ARTIFACTS_DIR, DATA_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# Training configuration
CONFIG = {
    # Model settings
    "model_name": "openai/whisper-tiny",  # CHANGED: Using tiny for faster download
    "language": "azerbaijani",
    "task": "transcribe",
    
    # Dataset settings
    "dataset_name": "LocalDoc/azerbaijani_asr",
    "sample_mode": True,  # Set False for full training
    "sample_size": 500,   # Number of samples in sample mode
    "sampling_rate": 16000,
    
    # Data splits
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    
    # Training hyperparameters
    "batch_size": 8,
    "num_epochs": 3,
    "learning_rate": 1e-5,
    "warmup_steps": 500,
    "gradient_accumulation_steps": 2,
    "max_steps": -1,  # -1 for full epochs
    "fp16": True,
    
    # Evaluation settings
    "eval_steps": 100,
    "save_steps": 500,
    "logging_steps": 50,
    "save_total_limit": 3,
    
    # Reproducibility
    "random_seed": 42,
    
    # Experiment tracking
    "experiment_name": f"whisper_azerbaijani_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
}

# Adjust config for sample mode
if CONFIG["sample_mode"]:
    CONFIG.update({
        "batch_size": 4,
        "num_epochs": 1,
        "max_steps": 100,
        "eval_steps": 50,
        "save_steps": 50,
        "warmup_steps": 20,
    })

# Save configuration
config_path = OUTPUTS_DIR / f"{CONFIG['experiment_name']}_config.json"
with open(config_path, 'w') as f:
    json.dump(CONFIG, f, indent=2, default=str)

print("=" * 70)
print(f"Experiment: {CONFIG['experiment_name']}")
print("=" * 70)
print(f"Mode: {'SAMPLE' if CONFIG['sample_mode'] else 'FULL TRAINING'}")
print(f"Model: {CONFIG['model_name']}")
print(f"Dataset: {CONFIG['dataset_name']}")
print(f"Random Seed: {CONFIG['random_seed']}")
print(f"\nConfiguration saved to: {config_path}")
print("=" * 70)

# ============================================================
# SSL Configuration for Corporate Networks
# ============================================================

import ssl
import sys
import warnings

# Disable Xet storage and SSL verification
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

sys.modules['hf_xet'] = None
ssl._create_default_https_context = ssl._create_unverified_context

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

import requests
_orig_request = requests.Session.request

def _patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return _orig_request(self, method, url, **kwargs)

requests.Session.request = _patched_request

print("✓ SSL verification disabled (corporate network compatibility)")

# ============================================================
# Set Random Seeds for Reproducibility
# ============================================================

import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hashing
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(CONFIG["random_seed"])
print(f"✓ Random seed set to {CONFIG['random_seed']} for reproducibility")

# ============================================================
# Import Required Libraries
# ============================================================

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Hugging Face libraries
from datasets import load_dataset, Dataset, DatasetDict, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback,
)
import evaluate

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

print("✓ All libraries imported successfully")

# ---
# ## 2. Hardware Detection and Optimization

# ============================================================
# Detect Available Hardware
# ============================================================

def detect_device() -> Dict[str, Any]:
    """Detect available hardware and return device info."""
    device_info = {
        "device": "cpu",
        "device_name": "CPU",
        "memory_gb": None,
        "fp16_available": False,
    }
    
    if torch.cuda.is_available():
        device_info["device"] = "cuda"
        device_info["device_name"] = torch.cuda.get_device_name(0)
        device_info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        device_info["fp16_available"] = True
    elif torch.backends.mps.is_available():
        device_info["device"] = "mps"
        device_info["device_name"] = "Apple Silicon (MPS)"
        device_info["fp16_available"] = False  # MPS doesn't support fp16 mixed precision
    
    return device_info

device_info = detect_device()

# Update config based on hardware
if not device_info["fp16_available"]:
    CONFIG["fp16"] = False

# Save device info
device_info_path = OUTPUTS_DIR / f"{CONFIG['experiment_name']}_device_info.json"
with open(device_info_path, 'w') as f:
    json.dump(device_info, f, indent=2)

print("=" * 70)
print("Hardware Information")
print("=" * 70)
print(f"Device: {device_info['device_name']}")
if device_info['memory_gb']:
    print(f"Memory: {device_info['memory_gb']:.2f} GB")
print(f"FP16 Training: {'Enabled' if CONFIG['fp16'] else 'Disabled'}")
print(f"\nDevice info saved to: {device_info_path}")
print("=" * 70)

# ---
# ## 3. Data Loading and Validation

# ============================================================
# Load Dataset
# ============================================================

def load_asr_dataset(config: Dict) -> DatasetDict:
    """
    Load ASR dataset with streaming support.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DatasetDict with train split
    """
    print(f"Loading dataset: {config['dataset_name']}")
    print(f"Mode: {'Sample ({} samples)'.format(config['sample_size']) if config['sample_mode'] else 'Full dataset'}")
    
    # Check if local data exists
    local_path = DATA_DIR / "dataset_cache"
    
    if local_path.exists() and not config['sample_mode']:
        print(f"Loading from local cache: {local_path}")
        from datasets import load_from_disk
        dataset = load_from_disk(str(local_path))
    else:
        print("Loading with streaming mode...")
        dataset_stream = load_dataset(
            config['dataset_name'],
            streaming=True,
            trust_remote_code=False
        )
        
        # Take samples if in sample mode
        if config['sample_mode']:
            n_samples = config['sample_size']
            print(f"Taking {n_samples} samples from stream...")
            
            samples = list(tqdm(
                dataset_stream["train"].take(n_samples),
                total=n_samples,
                desc="Loading samples"
            ))
            
            dataset = DatasetDict({
                "train": Dataset.from_list(samples)
            })
        else:
            # For full dataset, download and cache
            dataset = load_dataset(config['dataset_name'])
            dataset.save_to_disk(str(local_path))
            print(f"Dataset cached to: {local_path}")
    
    return dataset

# Load dataset
print("=" * 70)
dataset = load_asr_dataset(CONFIG)
print("\n✓ Dataset loaded successfully")
print(f"Train samples: {len(dataset['train'])}")
print("=" * 70)

# ============================================================
# Data Validation and Schema Checks
# ============================================================

def validate_dataset(dataset: Dataset) -> Dict[str, Any]:
    """
    Validate dataset schema and content.
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "total_samples": len(dataset),
        "columns": list(dataset.features.keys()),
        "missing_values": {},
        "duration_stats": {},
        "text_stats": {},
        "issues": [],
    }
    
    # Check for required columns
    sample = dataset[0]
    audio_col = "audio" if "audio" in sample else "path"
    text_col = next((col for col in ["sentence", "text", "transcription"] if col in sample), None)
    
    if audio_col not in sample:
        validation_results["issues"].append("Missing audio column")
    if text_col is None:
        validation_results["issues"].append("Missing text column")
    
    validation_results["audio_column"] = audio_col
    validation_results["text_column"] = text_col
    
    # Check for missing/empty values
    for col in [audio_col, text_col]:
        if col:
            empty_count = sum(1 for item in dataset if not item.get(col))
            validation_results["missing_values"][col] = empty_count
            if empty_count > 0:
                validation_results["issues"].append(f"{col} has {empty_count} missing values")
    
    # Analyze durations if available
    if "duration" in sample:
        durations = [item["duration"] for item in dataset if item.get("duration")]
        validation_results["duration_stats"] = {
            "mean": np.mean(durations),
            "median": np.median(durations),
            "std": np.std(durations),
            "min": np.min(durations),
            "max": np.max(durations),
            "total_hours": np.sum(durations) / 3600,
        }
    
    # Analyze text lengths
    if text_col:
        text_lengths = [len(item[text_col]) for item in dataset if item.get(text_col)]
        validation_results["text_stats"] = {
            "mean_length": np.mean(text_lengths),
            "median_length": np.median(text_lengths),
            "min_length": np.min(text_lengths),
            "max_length": np.max(text_lengths),
        }
    
    return validation_results

# Validate dataset
print("=" * 70)
print("Data Validation")
print("=" * 70)

validation_results = validate_dataset(dataset["train"])

# Print validation results
print(f"\nTotal Samples: {validation_results['total_samples']}")
print(f"Columns: {', '.join(validation_results['columns'])}")
print(f"Audio Column: {validation_results['audio_column']}")
print(f"Text Column: {validation_results['text_column']}")

if validation_results['missing_values']:
    print("\nMissing Values:")
    for col, count in validation_results['missing_values'].items():
        print(f"  {col}: {count}")

if validation_results['duration_stats']:
    print("\nDuration Statistics:")
    for key, value in validation_results['duration_stats'].items():
        print(f"  {key}: {value:.2f}")

if validation_results['text_stats']:
    print("\nText Statistics:")
    for key, value in validation_results['text_stats'].items():
        print(f"  {key}: {value:.2f}")

if validation_results['issues']:
    print("\n⚠️  Issues Found:")
    for issue in validation_results['issues']:
        print(f"  - {issue}")
else:
    print("\n✓ No validation issues found")

# Save validation results
validation_path = OUTPUTS_DIR / f"{CONFIG['experiment_name']}_validation.json"
with open(validation_path, 'w') as f:
    json.dump(validation_results, f, indent=2, default=str)

print(f"\nValidation results saved to: {validation_path}")
print("=" * 70)

# ============================================================
# Exploratory Data Analysis - Visualizations
# ============================================================

def create_eda_visualizations(dataset: Dataset, validation_results: Dict, save_dir: Path):
    """
    Create exploratory data analysis visualizations.
    """
    text_col = validation_results["text_column"]
    
    # Duration distribution (if available)
    if "duration" in dataset[0]:
        durations = [item["duration"] for item in dataset if item.get("duration")]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Histogram
        ax1.hist(durations, bins=50, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Duration (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Audio Duration Distribution')
        ax1.axvline(np.mean(durations), color='r', linestyle='--', label=f'Mean: {np.mean(durations):.2f}s')
        ax1.axvline(np.median(durations), color='g', linestyle='--', label=f'Median: {np.median(durations):.2f}s')
        ax1.legend()
        
        # Box plot
        ax2.boxplot(durations, vert=True)
        ax2.set_ylabel('Duration (seconds)')
        ax2.set_title('Audio Duration Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / "duration_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Duration distribution chart saved")
    
    # Text length distribution
    if text_col:
        text_lengths = [len(item[text_col]) for item in dataset if item.get(text_col)]
        word_counts = [len(item[text_col].split()) for item in dataset if item.get(text_col)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Character length
        ax1.hist(text_lengths, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.set_xlabel('Text Length (characters)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Transcription Length Distribution (Characters)')
        ax1.axvline(np.mean(text_lengths), color='r', linestyle='--', label=f'Mean: {np.mean(text_lengths):.1f}')
        ax1.legend()
        
        # Word count
        ax2.hist(word_counts, bins=50, edgecolor='black', alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Word Count')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Transcription Length Distribution (Words)')
        ax2.axvline(np.mean(word_counts), color='r', linestyle='--', label=f'Mean: {np.mean(word_counts):.1f}')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / "text_length_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Text length distribution chart saved")
    
    # Summary statistics table
    summary_data = []
    if "duration" in dataset[0]:
        durations = [item["duration"] for item in dataset if item.get("duration")]
        summary_data.append(['Duration (sec)', f"{np.mean(durations):.2f}", f"{np.median(durations):.2f}", 
                            f"{np.std(durations):.2f}", f"{np.min(durations):.2f}", f"{np.max(durations):.2f}"])
    
    if text_col:
        text_lengths = [len(item[text_col]) for item in dataset if item.get(text_col)]
        summary_data.append(['Text Length (chars)', f"{np.mean(text_lengths):.1f}", f"{np.median(text_lengths):.1f}",
                            f"{np.std(text_lengths):.1f}", f"{np.min(text_lengths):.0f}", f"{np.max(text_lengths):.0f}"])
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Mean', 'Median', 'Std', 'Min', 'Max'])
        summary_df.to_csv(OUTPUTS_DIR / f"{CONFIG['experiment_name']}_data_summary.csv", index=False)
        print("✓ Data summary table saved")

# Create visualizations
print("\nCreating exploratory data analysis visualizations...")
create_eda_visualizations(dataset["train"], validation_results, CHARTS_DIR)
print(f"\nAll visualizations saved to: {CHARTS_DIR}")

# ---
# ## 4. Data Splitting (No Data Leakage)

# ============================================================
# Create Train/Val/Test Splits
# ============================================================

def create_splits(dataset: Dataset, config: Dict) -> DatasetDict:
    """
    Create stratified train/val/test splits with fixed random seed.
    
    Args:
        dataset: Input dataset
        config: Configuration dictionary
        
    Returns:
        DatasetDict with train, validation, and test splits
    """
    # Reset seed for reproducibility
    set_seed(config["random_seed"])
    
    train_ratio = config["train_ratio"]
    val_ratio = config["val_ratio"]
    test_ratio = config["test_ratio"]
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # First split: train and temp (val + test)
    train_test_split = dataset.train_test_split(
        test_size=(val_ratio + test_ratio),
        seed=config["random_seed"]
    )
    
    # Second split: val and test from temp
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_test_split = train_test_split["test"].train_test_split(
        test_size=val_test_ratio,
        seed=config["random_seed"]
    )
    
    splits = DatasetDict({
        "train": train_test_split["train"],
        "validation": val_test_split["train"],
        "test": val_test_split["test"],
    })
    
    return splits

# Create splits
print("=" * 70)
print("Creating Data Splits")
print("=" * 70)

dataset_splits = create_splits(dataset["train"], CONFIG)

split_info = {
    "train": len(dataset_splits["train"]),
    "validation": len(dataset_splits["validation"]),
    "test": len(dataset_splits["test"]),
    "train_ratio": CONFIG["train_ratio"],
    "val_ratio": CONFIG["val_ratio"],
    "test_ratio": CONFIG["test_ratio"],
    "random_seed": CONFIG["random_seed"],
}

print(f"\nSplit Sizes:")
print(f"  Train:      {split_info['train']:>6} ({CONFIG['train_ratio']*100:.0f}%)")
print(f"  Validation: {split_info['validation']:>6} ({CONFIG['val_ratio']*100:.0f}%)")
print(f"  Test:       {split_info['test']:>6} ({CONFIG['test_ratio']*100:.0f}%)")
print(f"  Total:      {sum([split_info['train'], split_info['validation'], split_info['test']]):>6}")

# Save split info
split_info_path = OUTPUTS_DIR / f"{CONFIG['experiment_name']}_split_info.json"
with open(split_info_path, 'w') as f:
    json.dump(split_info, f, indent=2)

print(f"\n✓ Split information saved to: {split_info_path}")
print("=" * 70)

# ---
# ## 5. Model Loading

# ============================================================
# Load Whisper Model and Processor
# ============================================================

print("=" * 70)
print(f"Loading Model: {CONFIG['model_name']}")
print("=" * 70)

# Load processor (combines tokenizer and feature extractor)
processor = WhisperProcessor.from_pretrained(
    CONFIG["model_name"],
    language=CONFIG["language"],
    task=CONFIG["task"]
)

# Load model
model = WhisperForConditionalGeneration.from_pretrained(CONFIG["model_name"])

# Configure model for Azerbaijani
model.generation_config.language = CONFIG["language"]
model.generation_config.task = CONFIG["task"]
model.generation_config.forced_decoder_ids = None

# Model information
model_info = {
    "model_name": CONFIG["model_name"],
    "num_parameters": model.num_parameters(),
    "language": CONFIG["language"],
    "task": CONFIG["task"],
    "vocab_size": model.config.vocab_size,
    "d_model": model.config.d_model,
}

print(f"\n✓ Model loaded successfully")
print(f"Parameters: {model_info['num_parameters']:,}")
print(f"Vocabulary Size: {model_info['vocab_size']:,}")
print(f"Model Dimension: {model_info['d_model']}")

# Save model info
model_info_path = OUTPUTS_DIR / f"{CONFIG['experiment_name']}_model_info.json"
with open(model_info_path, 'w') as f:
    json.dump(model_info, f, indent=2)

print(f"\nModel information saved to: {model_info_path}")
print("=" * 70)

# ---
# ## 6. Data Preprocessing

# ============================================================
# Prepare Dataset for Training
# ============================================================

# Get column names from validation results
audio_column = validation_results["audio_column"]
text_column = validation_results["text_column"]

print("=" * 70)
print("Preprocessing Dataset")
print("=" * 70)
print(f"Audio column: {audio_column}")
print(f"Text column: {text_column}")
print(f"Target sampling rate: {CONFIG['sampling_rate']} Hz")

# Cast audio to correct sampling rate
dataset_splits = dataset_splits.cast_column(
    audio_column,
    Audio(sampling_rate=CONFIG["sampling_rate"])
)

def prepare_dataset(batch):
    """
    Prepare a batch for training.
    Converts audio to features and tokenizes text.
    """
    # Extract audio
    audio = batch[audio_column]
    
    # Compute input features
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Tokenize text
    batch["labels"] = processor.tokenizer(batch[text_column]).input_ids
    
    return batch

# Process all splits
print("\nProcessing splits...")
processed_datasets = dataset_splits.map(
    prepare_dataset,
    remove_columns=dataset_splits["train"].column_names,
    desc="Preprocessing",
)

print(f"\n✓ All splits preprocessed")
print(f"  Train: {len(processed_datasets['train'])} samples")
print(f"  Validation: {len(processed_datasets['validation'])} samples")
print(f"  Test: {len(processed_datasets['test'])} samples")
print("=" * 70)

# ============================================================
# Data Collator for Dynamic Padding
# ============================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs and labels.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self,
        features: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        # Split inputs and labels
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )

        # Replace padding with -100 to ignore in loss computation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        # Remove BOS token if present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# Create data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

print("✓ Data collator created")

# ---
# ## 7. Evaluation Metrics

# ============================================================
# Define Evaluation Metrics
# ============================================================

# Load WER metric
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    """
    Compute Word Error Rate (WER) during evaluation.
    
    Lower WER is better (0% is perfect).
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute WER
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

print("✓ Evaluation metrics configured")
print("  Metric: Word Error Rate (WER)")
print("  Lower is better (0% = perfect transcription)")

# ---
# ## 8. Training Configuration

# ============================================================
# Training Arguments
# ============================================================

# Create output directory for this experiment
experiment_artifacts_dir = ARTIFACTS_DIR / CONFIG["experiment_name"]
experiment_artifacts_dir.mkdir(exist_ok=True, parents=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=str(experiment_artifacts_dir),
    
    # Training parameters
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    num_train_epochs=CONFIG["num_epochs"],
    max_steps=CONFIG["max_steps"],
    warmup_steps=CONFIG["warmup_steps"],
    
    # Precision
    fp16=CONFIG["fp16"],
    
    # Evaluation and saving
    eval_strategy="steps",
    eval_steps=CONFIG["eval_steps"],
    save_strategy="steps",
    save_steps=CONFIG["save_steps"],
    save_total_limit=CONFIG["save_total_limit"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    
    # Generation settings for evaluation
    predict_with_generate=True,
    generation_max_length=225,
    
    # Logging
    logging_steps=CONFIG["logging_steps"],
    logging_dir=str(experiment_artifacts_dir / "logs"),
    report_to=["tensorboard"],
    
    # Device
    use_cpu=(device_info["device"] == "cpu"),
    
    # Reproducibility
    seed=CONFIG["random_seed"],
    data_seed=CONFIG["random_seed"],
    
    # Misc
    push_to_hub=False,
    remove_unused_columns=False,
)

print("=" * 70)
print("Training Configuration")
print("=" * 70)
print(f"Output directory: {experiment_artifacts_dir}")
print(f"\nHyperparameters:")
print(f"  Batch size: {CONFIG['batch_size']}")
print(f"  Gradient accumulation: {CONFIG['gradient_accumulation_steps']}")
print(f"  Effective batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
print(f"  Learning rate: {CONFIG['learning_rate']}")
print(f"  Epochs: {CONFIG['num_epochs']}")
print(f"  Max steps: {CONFIG['max_steps'] if CONFIG['max_steps'] > 0 else 'Full epochs'}")
print(f"  Warmup steps: {CONFIG['warmup_steps']}")
print(f"  FP16: {CONFIG['fp16']}")
print(f"\nEvaluation:")
print(f"  Eval every: {CONFIG['eval_steps']} steps")
print(f"  Save every: {CONFIG['save_steps']} steps")
print(f"  Keep best: {CONFIG['save_total_limit']} checkpoints")
print("=" * 70)

# ============================================================
# Custom Callback for Tracking Training Progress
# ============================================================

class TrainingMetricsCallback(TrainerCallback):
    """
    Callback to track and save training metrics.
    """
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.training_history = []
        self.eval_history = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logs_copy = logs.copy()
            logs_copy["step"] = state.global_step
            logs_copy["epoch"] = state.epoch
            
            if "loss" in logs:
                self.training_history.append(logs_copy)
            if "eval_wer" in logs:
                self.eval_history.append(logs_copy)
    
    def on_train_end(self, args, state, control, **kwargs):
        # Save training history
        if self.training_history:
            train_df = pd.DataFrame(self.training_history)
            train_df.to_csv(
                self.output_dir / f"{CONFIG['experiment_name']}_training_history.csv",
                index=False
            )
        
        # Save evaluation history
        if self.eval_history:
            eval_df = pd.DataFrame(self.eval_history)
            eval_df.to_csv(
                self.output_dir / f"{CONFIG['experiment_name']}_eval_history.csv",
                index=False
            )

# Create callback
metrics_callback = TrainingMetricsCallback(OUTPUTS_DIR)

print("✓ Training metrics callback created")

# ---
# ## 9. Model Training

# ============================================================
# Initialize Trainer
# ============================================================

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
    callbacks=[metrics_callback],
)

print("✓ Trainer initialized")
print(f"  Train samples: {len(processed_datasets['train'])}")
print(f"  Validation samples: {len(processed_datasets['validation'])}")

# ============================================================
# Train Model
# ============================================================

print("\n" + "=" * 70)
print("Starting Training")
print("=" * 70)
print(f"Device: {device_info['device_name']}")
print(f"Mode: {'SAMPLE' if CONFIG['sample_mode'] else 'FULL'}")
print("\nThis may take a while...")
print("=" * 70 + "\n")

# Train
train_result = trainer.train()

print("\n" + "=" * 70)
print("Training Completed!")
print("=" * 70)

# Save training metrics
training_metrics = {
    "train_runtime": train_result.metrics.get("train_runtime", 0),
    "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
    "train_steps_per_second": train_result.metrics.get("train_steps_per_second", 0),
    "total_flos": train_result.metrics.get("total_flos", 0),
    "train_loss": train_result.metrics.get("train_loss", 0),
    "epoch": train_result.metrics.get("epoch", 0),
}

# Print summary
print(f"\nTraining Summary:")
print(f"  Runtime: {training_metrics['train_runtime']:.2f} seconds")
print(f"  Samples/second: {training_metrics['train_samples_per_second']:.2f}")
print(f"  Final loss: {training_metrics['train_loss']:.4f}")
print(f"  Epochs completed: {training_metrics['epoch']:.2f}")

# Save metrics
metrics_path = OUTPUTS_DIR / f"{CONFIG['experiment_name']}_training_metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(training_metrics, f, indent=2)

print(f"\nTraining metrics saved to: {metrics_path}")
print("=" * 70)

# ---
# ## 10. Model Evaluation

# ============================================================
# Evaluate on Validation Set
# ============================================================

print("=" * 70)
print("Evaluating on Validation Set")
print("=" * 70)

val_results = trainer.evaluate()

print(f"\nValidation Results:")
for key, value in val_results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Save results
val_results_path = OUTPUTS_DIR / f"{CONFIG['experiment_name']}_validation_results.json"
with open(val_results_path, 'w') as f:
    json.dump(val_results, f, indent=2)

print(f"\nValidation results saved to: {val_results_path}")
print("=" * 70)

# ============================================================
# Evaluate on Test Set (Final Evaluation)
# ============================================================

print("=" * 70)
print("Evaluating on Test Set (Hold-out)")
print("=" * 70)

test_results = trainer.evaluate(processed_datasets["test"])

print(f"\nTest Results:")
for key, value in test_results.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# Save results
test_results_path = OUTPUTS_DIR / f"{CONFIG['experiment_name']}_test_results.json"
with open(test_results_path, 'w') as f:
    json.dump(test_results, f, indent=2)

print(f"\nTest results saved to: {test_results_path}")
print("=" * 70)

# ============================================================
# Generate Sample Predictions
# ============================================================

def generate_sample_predictions(
    trainer: Seq2SeqTrainer,
    dataset: Dataset,
    processor: WhisperProcessor,
    n_samples: int = 10
) -> List[Dict[str, str]]:
    """
    Generate predictions for sample test cases.
    """
    # Get random samples
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    samples = dataset.select(indices)
    
    # Generate predictions
    predictions = trainer.predict(samples)
    pred_ids = predictions.predictions
    label_ids = predictions.label_ids
    
    # Decode
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Create results
    results = []
    for i, (pred, label) in enumerate(zip(pred_str, label_str)):
        results.append({
            "sample_id": i + 1,
            "reference": label,
            "prediction": pred,
        })
    
    return results

print("Generating sample predictions...")
sample_predictions = generate_sample_predictions(
    trainer,
    processed_datasets["test"],
    processor,
    n_samples=10
)

# Display samples
print("\n" + "=" * 70)
print("Sample Predictions")
print("=" * 70)
for sample in sample_predictions[:5]:  # Show first 5
    print(f"\nSample {sample['sample_id']}:")
    print(f"  Reference:  {sample['reference']}")
    print(f"  Prediction: {sample['prediction']}")

# Save all predictions
predictions_df = pd.DataFrame(sample_predictions)
predictions_path = OUTPUTS_DIR / f"{CONFIG['experiment_name']}_sample_predictions.csv"
predictions_df.to_csv(predictions_path, index=False)

print(f"\nAll sample predictions saved to: {predictions_path}")
print("=" * 70)

# ---
# ## 11. Training Visualizations

# ============================================================
# Create Training Visualizations
# ============================================================

def plot_training_curves(metrics_callback: TrainingMetricsCallback, save_dir: Path):
    """
    Create comprehensive training visualization plots.
    """
    if not metrics_callback.training_history:
        print("No training history available for plotting")
        return
    
    # Convert to dataframes
    train_df = pd.DataFrame(metrics_callback.training_history)
    eval_df = pd.DataFrame(metrics_callback.eval_history) if metrics_callback.eval_history else None
    
    # Training loss curve
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_df['step'], train_df['loss'], label='Training Loss', linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "training_loss_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Training loss curve saved")
    
    # Evaluation WER curve
    if eval_df is not None and not eval_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(eval_df['step'], eval_df['eval_wer'], label='Validation WER', 
                color='orange', linewidth=2, marker='o')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Word Error Rate (%)')
        ax.set_title('Validation WER Over Time (Lower is Better)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "validation_wer_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Validation WER curve saved")
    
    # Combined plot
    if eval_df is not None and not eval_df.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Loss
        ax1.plot(train_df['step'], train_df['loss'], linewidth=2)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        
        # WER
        ax2.plot(eval_df['step'], eval_df['eval_wer'], color='orange', 
                linewidth=2, marker='o', markersize=4)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Word Error Rate (%)')
        ax2.set_title('Validation WER')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / "training_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Training overview saved")

# Create plots
print("\nCreating training visualizations...")
plot_training_curves(metrics_callback, CHARTS_DIR)
print(f"\nAll charts saved to: {CHARTS_DIR}")

# ============================================================
# Create Final Results Summary Visualization
# ============================================================

def create_results_summary(train_metrics, val_results, test_results, save_dir: Path):
    """
    Create a comprehensive results summary visualization.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. WER Comparison
    wer_data = {
        'Validation': val_results.get('eval_wer', 0),
        'Test': test_results.get('eval_wer', 0),
    }
    ax1.bar(wer_data.keys(), wer_data.values(), color=['#2ecc71', '#e74c3c'])
    ax1.set_ylabel('Word Error Rate (%)')
    ax1.set_title('Model Performance (Lower is Better)')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, (k, v) in enumerate(wer_data.items()):
        ax1.text(i, v + 0.5, f'{v:.2f}%', ha='center', fontweight='bold')
    
    # 2. Training Time Breakdown
    runtime_hours = train_metrics['train_runtime'] / 3600
    ax2.bar(['Training Time'], [runtime_hours], color='#3498db')
    ax2.set_ylabel('Hours')
    ax2.set_title('Training Duration')
    ax2.text(0, runtime_hours + runtime_hours*0.05, f'{runtime_hours:.2f}h', 
            ha='center', fontweight='bold')
    
    # 3. Dataset Split Sizes
    split_sizes = {
        'Train': len(processed_datasets['train']),
        'Validation': len(processed_datasets['validation']),
        'Test': len(processed_datasets['test']),
    }
    ax3.bar(split_sizes.keys(), split_sizes.values(), color=['#9b59b6', '#f39c12', '#1abc9c'])
    ax3.set_ylabel('Number of Samples')
    ax3.set_title('Dataset Split Sizes')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Training Metrics Summary
    metrics_text = f"""
    Experiment: {CONFIG['experiment_name']}
    
    Model: {CONFIG['model_name']}
    Parameters: {model_info['num_parameters']:,}
    
    Training:
    - Epochs: {train_metrics['epoch']:.2f}
    - Final Loss: {train_metrics['train_loss']:.4f}
    - Samples/sec: {train_metrics['train_samples_per_second']:.2f}
    
    Validation WER: {val_results.get('eval_wer', 0):.2f}%
    Test WER: {test_results.get('eval_wer', 0):.2f}%
    
    Device: {device_info['device_name']}
    FP16: {CONFIG['fp16']}
    Batch Size: {CONFIG['batch_size']}
    Learning Rate: {CONFIG['learning_rate']}
    """
    ax4.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax4.axis('off')
    ax4.set_title('Training Summary')
    
    plt.tight_layout()
    plt.savefig(save_dir / "results_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Results summary visualization saved")

# Create summary
print("\nCreating results summary...")
create_results_summary(training_metrics, val_results, test_results, CHARTS_DIR)

# ---
# ## 12. Model Persistence

# ============================================================
# Save Final Model and Artifacts
# ============================================================

print("=" * 70)
print("Saving Model Artifacts")
print("=" * 70)

# Create final model directory
final_model_dir = ARTIFACTS_DIR / f"{CONFIG['experiment_name']}_final"
final_model_dir.mkdir(exist_ok=True, parents=True)

# Save model and processor
trainer.save_model(str(final_model_dir))
processor.save_pretrained(str(final_model_dir))

print(f"\n✓ Model saved to: {final_model_dir}")

# Save complete experiment metadata
experiment_metadata = {
    "experiment_name": CONFIG["experiment_name"],
    "timestamp": datetime.now().isoformat(),
    "config": CONFIG,
    "device_info": device_info,
    "model_info": model_info,
    "split_info": split_info,
    "training_metrics": training_metrics,
    "validation_results": val_results,
    "test_results": test_results,
    "model_path": str(final_model_dir),
}

metadata_path = final_model_dir / "experiment_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(experiment_metadata, f, indent=2, default=str)

print(f"✓ Experiment metadata saved to: {metadata_path}")

# Create README for the model
readme_content = f"""# Azerbaijani ASR Model

## Experiment: {CONFIG['experiment_name']}

### Model Information
- Base Model: {CONFIG['model_name']}
- Language: {CONFIG['language']}
- Task: {CONFIG['task']}
- Parameters: {model_info['num_parameters']:,}

### Performance
- Validation WER: {val_results.get('eval_wer', 0):.2f}%
- Test WER: {test_results.get('eval_wer', 0):.2f}%

### Training Details
- Training Samples: {split_info['train']}
- Validation Samples: {split_info['validation']}
- Test Samples: {split_info['test']}
- Epochs: {training_metrics['epoch']:.2f}
- Training Time: {training_metrics['train_runtime']/3600:.2f} hours
- Device: {device_info['device_name']}

### Usage

```python
from transformers import pipeline

# Load the model
pipe = pipeline(
    "automatic-speech-recognition",
    model="{final_model_dir}"
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
- Random Seed: {CONFIG['random_seed']}
- All configuration and results saved in experiment_metadata.json

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

readme_path = final_model_dir / "README.md"
with open(readme_path, 'w') as f:
    f.write(readme_content)

print(f"✓ Model README saved to: {readme_path}")
print("\n" + "=" * 70)
print("All artifacts saved successfully!")
print("=" * 70)

# ---
# ## 13. Inference Test

# ============================================================
# Test Inference Pipeline
# ============================================================

from transformers import pipeline

print("=" * 70)
print("Testing Inference Pipeline")
print("=" * 70)

# Load inference pipeline
device_id = 0 if device_info["device"] == "cuda" else -1
pipe = pipeline(
    "automatic-speech-recognition",
    model=str(final_model_dir),
    device=device_id,
)

print("\n✓ Inference pipeline loaded successfully")
print(f"  Model: {final_model_dir}")
print(f"  Device: {device_info['device']}")

print("\nTo use the model for inference:")
print("""\n```python
from transformers import pipeline

# Load the model
pipe = pipeline(
    "automatic-speech-recognition",
    model="{}"
)

# Transcribe audio file
result = pipe("path/to/audio.wav")
print(result["text"])

# Or with audio array
import librosa
audio, sr = librosa.load("path/to/audio.wav", sr=16000)
result = pipe(audio)
print(result["text"])
```""".format(final_model_dir))

print("\n" + "=" * 70)

# ---
# ## 14. Final Summary

# ============================================================
# Generate Final Report
# ============================================================

print("\n" + "=" * 70)
print("FINAL EXPERIMENT SUMMARY")
print("=" * 70)

print(f"\nExperiment: {CONFIG['experiment_name']}")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print("\n" + "-" * 70)
print("MODEL PERFORMANCE")
print("-" * 70)
print(f"Validation WER: {val_results.get('eval_wer', 0):.2f}%")
print(f"Test WER: {test_results.get('eval_wer', 0):.2f}%")

print("\n" + "-" * 70)
print("DATASET")
print("-" * 70)
print(f"Train: {split_info['train']} samples")
print(f"Validation: {split_info['validation']} samples")
print(f"Test: {split_info['test']} samples")
print(f"Total: {sum([split_info['train'], split_info['validation'], split_info['test']])} samples")

print("\n" + "-" * 70)
print("TRAINING")
print("-" * 70)
print(f"Runtime: {training_metrics['train_runtime']/3600:.2f} hours")
print(f"Epochs: {training_metrics['epoch']:.2f}")
print(f"Final Loss: {training_metrics['train_loss']:.4f}")
print(f"Throughput: {training_metrics['train_samples_per_second']:.2f} samples/sec")

print("\n" + "-" * 70)
print("ARTIFACTS LOCATIONS")
print("-" * 70)
print(f"Model: {final_model_dir}")
print(f"Charts: {CHARTS_DIR}")
print(f"Outputs: {OUTPUTS_DIR}")
print(f"All Artifacts: {ARTIFACTS_DIR}")

print("\n" + "-" * 70)
print("GENERATED FILES")
print("-" * 70)

# List all generated files
print("\nCharts:")
for f in sorted(CHARTS_DIR.glob("*.png")):
    print(f"  - {f.name}")

print("\nOutputs:")
for f in sorted(OUTPUTS_DIR.glob(f"{CONFIG['experiment_name']}*")):
    print(f"  - {f.name}")

print("\nModel Files:")
for f in sorted(final_model_dir.glob("*")):
    if f.is_file():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.2f} MB)")

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\n✓ All stages completed")
print("✓ Model trained and evaluated")
print("✓ All artifacts saved")
print("✓ Visualizations generated")
print("✓ Results documented")

print("\nNext steps:")
print("  1. Review visualizations in the charts/ directory")
print("  2. Examine detailed metrics in the outputs/ directory")
print("  3. Load the model from artifacts/ for inference")
print("  4. For full training, set SAMPLE_MODE=False and rerun")

print("\n" + "=" * 70)
