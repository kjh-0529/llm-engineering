"""Upload the trained adapter and MODEL_CARD.md to the Hugging Face Hub.

    export HF_TOKEN=...          # on Kaggle: from Add-ons -> Secrets
    python upload_adapter.py

The token is read from the HF_TOKEN environment variable by huggingface_hub;
it is never written to disk or printed.
"""
import argparse
import os
import shutil

from huggingface_hub import HfApi

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="kjh0529/qwen3-1.7b-ko-tech-qlora")
ap.add_argument("--adapter", default=os.path.join(HERE, "runs", "qlora"))
args = ap.parse_args()

if not os.path.isfile(os.path.join(args.adapter, "adapter_config.json")):
    raise SystemExit(f"no adapter at {args.adapter}: did train_qlora.py finish?")
shutil.copy(os.path.join(HERE, "MODEL_CARD.md"), os.path.join(args.adapter, "README.md"))

api = HfApi()
api.create_repo(args.repo, exist_ok=True)
api.upload_folder(folder_path=args.adapter, repo_id=args.repo,
                  commit_message="Upload QLoRA adapter and model card")
print(f"uploaded: https://huggingface.co/{args.repo}")
