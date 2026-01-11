"""
Download Whisper Model

Pre-downloads the Whisper model to avoid issues during training.
Useful for environments with network restrictions.

Usage:
    python scripts/download_model.py --model openai/whisper-small
"""

import argparse
import os
import ssl
import sys

# SSL Configuration
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

sys.modules['hf_xet'] = None
ssl._create_default_https_context = ssl._create_unverified_context

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

import requests
_orig_request = requests.Session.request

def _patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return _orig_request(self, method, url, **kwargs)

requests.Session.request = _patched_request


def download_model(model_name: str, cache_dir: str = "./models"):
    """Download Whisper model and processor."""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    print("=" * 70)
    print(f"Downloading Whisper Model: {model_name}")
    print("=" * 70)

    os.makedirs(cache_dir, exist_ok=True)

    try:
        # Download processor
        print("\n1. Downloading processor (tokenizer + feature extractor)...")
        processor = WhisperProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            force_download=False
        )
        print("   ✓ Processor downloaded")

        # Download model
        print(f"\n2. Downloading model ({model_name})...")
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            force_download=False
        )
        print("   ✓ Model downloaded")

        # Print info
        print("\n" + "=" * 70)
        print("Download Complete!")
        print("=" * 70)
        print(f"Model: {model_name}")
        print(f"Parameters: {model.num_parameters():,}")
        print(f"Cache directory: {cache_dir}")
        print("=" * 70)

        return True

    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Try with force_download=True")
        print("3. Use a smaller model (whisper-tiny or whisper-base)")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Whisper model")
    parser.add_argument(
        "--model",
        type=str,
        default="openai/whisper-small",
        choices=[
            "openai/whisper-tiny",
            "openai/whisper-base",
            "openai/whisper-small",
            "openai/whisper-medium",
            "openai/whisper-large-v2",
        ],
        help="Whisper model to download"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./models",
        help="Directory to cache the model"
    )

    args = parser.parse_args()

    success = download_model(args.model, args.cache_dir)

    if success:
        print("\n✓ Ready for training!")
    else:
        print("\n✗ Download failed")
        sys.exit(1)
