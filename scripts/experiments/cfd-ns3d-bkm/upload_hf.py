#!/usr/bin/env python3
"""Upload 3D NS BKM data to Hugging Face (cahlen/cfd-ns3d-bkm)."""

import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "cahlen/cfd-ns3d-bkm"
SCRIPT_DIR = Path(__file__).parent
RESULTS = SCRIPT_DIR / "results"

RUNS = [
    ("smoke_taylor_green", "bkm3d_n64_nu1e-02_steps200.csv", "run_n64_nu0.01_steps200_taylor-green.log"),
    ("standard_random", "bkm3d_n128_nu1e-03_steps1000.csv", "run_n128_nu0.001_steps1000_random.log"),
    ("blowup_search", "bkm3d_n256_nu1e-04_steps500.csv", "run_n256_nu0.0001_steps500_random.log"),
    ("blowup_search_long", "bkm3d_n256_nu1e-04_steps2000.csv", "run_n256_nu0.0001_steps2000_random.log"),
    ("taylor_green_256", "bkm3d_n256_nu1e-03_steps1000.csv", "run_n256_nu0.001_steps1000_taylor-green.log"),
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
            "experiment": "cfd-ns3d-bkm",
            "experiment_url": "https://bigcompute.science/experiments/cfd-ns3d-bkm/",
            "dataset_repo": f"https://huggingface.co/datasets/{REPO_ID}",
            "kernel": "https://github.com/cahlen/idontknow/blob/main/scripts/experiments/cfd-ns3d-bkm/ns3d_bkm.cu",
            "code": "https://github.com/cahlen/idontknow/tree/main/scripts/experiments/cfd-ns3d-bkm",
            "finding": "https://bigcompute.science/findings/cfd-ns3d-bkm-infrastructure/",
            "related_dataset": "https://huggingface.co/datasets/cahlen/cfd-ns-bkm",
            "runs": {name: {"csv": csv, "log": log} for name, csv, log in RUNS},
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2))

        create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
        api = HfApi()
        api.upload_folder(
            folder_path=str(staging),
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="CFD 3D NS BKM certifying runs",
        )
        print(f"Uploaded to https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
