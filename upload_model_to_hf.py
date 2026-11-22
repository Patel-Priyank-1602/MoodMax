"""
MoodMax - Upload Fine-Tuned Emotion Model to Hugging Face Model Hub.

Usage:
    python upload_model_to_hf.py --repo-id <username>/moodmax-emotion-distilbert

Requirements:
    pip install huggingface_hub
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Upload MoodMax model to Hugging Face Model Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Target Hugging Face repository ID (e.g., 'your-username/moodmax-emotion-distilbert')"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face User Access Token (with write permission). Defaults to $HF_TOKEN"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Path(__file__).parent / "models" / "emotion-distilbert-multi"),
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the Hugging Face repository private (default is public)"
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        print(f"❌ Error: Model directory not found at: {model_dir}")
        sys.exit(1)

    required_files = ["config.json", "model.safetensors"]
    for req in required_files:
        if not (model_dir / req).exists():
            print(f"❌ Error: Required model file '{req}' missing from {model_dir}")
            sys.exit(1)

    token = args.token
    if not token:
        print("🔑 Hugging Face Token required.")
        print("Get your write token at: https://huggingface.co/settings/tokens")
        token = input("Enter your Hugging Face Write Token: ").strip()

    if not token:
        print("❌ Error: No token provided.")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("📦 Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import HfApi, login

    print(f"\n🚀 Authenticating with Hugging Face...")
    login(token=token, add_to_git_credential=False)
    api = HfApi()

    repo_id = args.repo_id
    if "/" not in repo_id:
        user = api.whoami(token=token)["name"]
        repo_id = f"{user}/{repo_id}"

    print(f"📁 Target Repository: https://huggingface.co/{repo_id}")
    print(f"📦 Source Directory:  {model_dir}")

    # 1. Create Model Repo if not exists
    print(f"\n1️⃣  Creating repository on Hugging Face (if not already existing)...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
            token=token
        )
        print("   ✓ Repository ready.")
    except Exception as e:
        print(f"   ⚠️ Repo creation warning: {e}")

    # 2. Upload model folder
    print(f"\n2️⃣  Uploading model files (~540MB safetensors + configs)...")
    print("   (This might take a couple of minutes depending on your internet upload speed)")

    # Exclude checkpoints folder, raw val probs to save upload bandwidth
    ignore_patterns = ["checkpoints/**", "*.csv", "*.npy", "__pycache__/**"]

    upload_url = api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=ignore_patterns,
        token=token,
        commit_message="Upload MoodMax fine-tuned multilingual emotion model"
    )

    print(f"\n✅ Upload Successful!")
    print(f"🎉 Model is live at: https://huggingface.co/{repo_id}")
    print(f"\n─────────────────────────────────────────────────────────────")
    print(f"Next Steps for Deployment:")
    print(f"1. On Render backend dashboard -> Environment Variables:")
    print(f"   HF_MODEL_ID = {repo_id}")
    print(f"   USE_HF_INFERENCE_API = true   (Recommended for Render 512MB RAM free tier)")
    print(f"   HF_TOKEN = <your_token>       (Optional if repo is public)")
    print(f"─────────────────────────────────────────────────────────────\n")

if __name__ == "__main__":
    main()
