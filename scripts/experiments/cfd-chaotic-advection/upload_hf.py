#!/usr/bin/env python3
"""
Upload CFD chaotic advection Lyapunov sweep data to Hugging Face.

Creates/updates cahlen/cfd-chaotic-advection with CSVs, logs, and metadata.

Usage:
  python3 scripts/experiments/cfd-chaotic-advection/upload_hf.py
"""

import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "cahlen/cfd-chaotic-advection"
SCRIPT_DIR = Path(__file__).parent
RESULTS = SCRIPT_DIR / "results"

SWEEPS = [
    ("deep_sweep", "lyapunov_k2048_ic8192_iter50000.csv", "run_k2048_ic8192_iter50000.log"),
    ("standard_sweep", "lyapunov_k512_ic4096_iter20000.csv", "run_k512_ic4096_iter20000.log"),
    ("smoke_test", "lyapunov_k64_ic512_iter5000.csv", "run_k64_ic512_iter5000.log"),
]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        for config, csv_name, log_name in SWEEPS:
            csv_src = RESULTS / csv_name
            log_src = RESULTS / log_name
            if not csv_src.exists():
                raise FileNotFoundError(f"Missing {csv_src}")
            dest = staging / "data" / config
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_src, dest / csv_name)
            if log_src.exists():
                log_dest = staging / "logs"
                log_dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(log_src, log_dest / log_name)

        metadata = {
            "experiment": "cfd-chaotic-advection",
            "dataset_repo": f"https://huggingface.co/datasets/{REPO_ID}",
            "kernel_repo": "https://huggingface.co/cahlen/bigcompute-cuda-kernels",
            "code": "https://github.com/cahlen/idontknow/tree/main/scripts/experiments/cfd-chaotic-advection",
            "sweeps": {
                name: {"csv": csv, "log": log}
                for name, csv, log in SWEEPS
            },
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        shutil.copy2(SCRIPT_DIR / "HF_README.md", staging / "README.md")

        print(f"Creating dataset repo: {REPO_ID}")
        create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

        api = HfApi()
        print("Uploading dataset files...")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Add Chirikov standard map Lyapunov spectrum (RTX 5090 certifying sweeps)",
        )

    print(f"\nDone! Dataset: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
