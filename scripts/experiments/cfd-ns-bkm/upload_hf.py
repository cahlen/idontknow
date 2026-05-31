#!/usr/bin/env python3
"""Upload CFD NS BKM data to Hugging Face (cahlen/cfd-ns-bkm)."""

import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "cahlen/cfd-ns-bkm"
SCRIPT_DIR = Path(__file__).parent
RESULTS = SCRIPT_DIR / "results"

RUNS = [
    ("smoke_taylor_green", "bkm_n256_nu1e-03_steps2000.csv", "run_n256_nu0.001_steps2000_taylor-green.log"),
    ("standard_random", "bkm_n512_nu1e-04_steps5000.csv", "run_n512_nu0.0001_steps5000_random.log"),
]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        for config, csv_name, log_name in RUNS:
            csv_src = RESULTS / csv_name
            if not csv_src.exists():
                print(f"Skipping missing {csv_src}")
                continue
            dest = staging / "data" / config
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(csv_src, dest / csv_name)
            log_src = RESULTS / log_name
            if log_src.exists():
                log_dest = staging / "logs"
                log_dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(log_src, log_dest / log_name)

        readme = SCRIPT_DIR / "HF_README.md"
        if readme.exists():
            shutil.copy2(readme, staging / "README.md")

        metadata = {
            "experiment": "cfd-ns-bkm",
            "experiment_url": "https://bigcompute.science/experiments/cfd-ns-bkm/",
            "dataset_repo": f"https://huggingface.co/datasets/{REPO_ID}",
            "kernel": "https://github.com/cahlen/idontknow/blob/main/scripts/experiments/cfd-ns-bkm/ns2d_bkm.cu",
            "code": "https://github.com/cahlen/idontknow/tree/main/scripts/experiments/cfd-ns-bkm",
            "finding": "https://bigcompute.science/findings/cfd-ns-bkm-diagnostic/",
            "related_dataset": "https://huggingface.co/datasets/cahlen/cfd-ns3d-bkm",
            "runs": {name: {"csv": csv, "log": log} for name, csv, log in RUNS},
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2))

        create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
        api = HfApi()
        api.upload_folder(
            folder_path=str(staging),
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="CFD NS BKM diagnostic sweeps",
        )
        print(f"Uploaded to https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
