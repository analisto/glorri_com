"""
Download All Dependencies

Downloads both dataset and model to prepare for offline training.
Useful for environments with intermittent connectivity.

Usage:
    python scripts/download_dependencies.py
    python scripts/download_dependencies.py --model whisper-tiny --samples 1000
"""

import argparse
import os
import sys

# Import download functions
sys.path.insert(0, os.path.dirname(__file__))
from download_data import download_dataset, download_streaming
from download_model import download_model


def main():
    parser = argparse.ArgumentParser(
        description="Download all dependencies for ASR training"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="whisper-small",
        help="Whisper model to download (tiny/base/small/medium)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="full",
        choices=["full", "streaming"],
        help="Download full dataset or use streaming mode"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples for streaming mode"
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model download"
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip dataset download"
    )

    args = parser.parse_args()

    # Map short names to full model names
    model_map = {
        "tiny": "openai/whisper-tiny",
        "base": "openai/whisper-base",
        "small": "openai/whisper-small",
        "medium": "openai/whisper-medium",
        "large": "openai/whisper-large-v2",
    }

    model_name = model_map.get(args.model, f"openai/whisper-{args.model}")

    print("=" * 70)
    print("Downloading All Dependencies")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Dataset: {args.dataset}")
    print("=" * 70)

    success = True

    # Download dataset
    if not args.skip_dataset:
        print("\n\n")
        print("=" * 70)
        print("STEP 1: Download Dataset")
        print("=" * 70)

        if args.dataset == "full":
            dataset_success = download_dataset()
        else:
            dataset_success = download_streaming()

        if not dataset_success:
            print("\n❌ Dataset download failed")
            success = False
    else:
        print("\n\nSkipping dataset download...")

    # Download model
    if not args.skip_model:
        print("\n\n")
        print("=" * 70)
        print("STEP 2: Download Model")
        print("=" * 70)

        model_success = download_model(model_name)

        if not model_success:
            print("\n❌ Model download failed")
            success = False
    else:
        print("\n\nSkipping model download...")

    # Summary
    print("\n\n")
    print("=" * 70)
    if success:
        print("✓ All Dependencies Downloaded Successfully!")
        print("=" * 70)
        print("\nYou're ready to start training:")
        print("  - Run the notebook: jupyter notebook asr_training_production.ipynb")
        print("  - Or run the script: python train_sample.py")
    else:
        print("✗ Some Downloads Failed")
        print("=" * 70)
        print("\nPlease check the errors above and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
