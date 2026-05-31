#!/usr/bin/env python3
"""
Upload CFD chaotic advection Lyapunov sweep data to Hugging Face.

Creates/updates cahlen/cfd-chaotic-advection with CSVs, logs, validation
artifacts, and metadata. All experiment/finding data must live here and be
linked from bigcompute.science.

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

CONVERGENCE_CSVS = [
    "lyapunov_k2_ic65536_iter5000.csv",
    "lyapunov_k2_ic65536_iter10000.csv",
    "lyapunov_k2_ic65536_iter20000.csv",
    "lyapunov_k2_ic65536_iter50000.csv",
    "lyapunov_k2_ic65536_iter100000.csv",
]

VALIDATION_JSON = [
    "convergence_k5_gpu.json",
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

        val_dir = staging / "validation"
        val_dir.mkdir(parents=True, exist_ok=True)
        for name in CONVERGENCE_CSVS:
            src = RESULTS / name
            if src.exists():
                shutil.copy2(src, val_dir / name)
        for name in VALIDATION_JSON:
            src = RESULTS / name
            if src.exists():
                shutil.copy2(src, val_dir / name)

        metadata = {
            "experiment": "cfd-chaotic-advection",
            "finding": "cfd-standard-map-chaos-onset",
            "experiment_url": "https://bigcompute.science/experiments/cfd-chaotic-advection/",
            "finding_url": "https://bigcompute.science/findings/cfd-standard-map-chaos-onset/",
            "dataset_repo": f"https://huggingface.co/datasets/{REPO_ID}",
            "kernel_repo": "https://huggingface.co/cahlen/bigcompute-cuda-kernels",
            "code": "https://github.com/cahlen/idontknow/tree/main/scripts/experiments/cfd-chaotic-advection",
            "sweeps": {
                name: {"csv": csv, "log": log}
                for name, csv, log in SWEEPS
            },
            "validation": {
                "convergence_k5_gpu": "validation/convergence_k5_gpu.json",
                "convergence_csvs": CONVERGENCE_CSVS,
                "script": "validate_claims.py",
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
            commit_message="Add validation artifacts and bidirectional bigcompute links",
        )

    print(f"\nDone! Dataset: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
