#!/usr/bin/env python3
"""
Upload class number raw data to Hugging Face.

Creates cahlen/class-numbers-real-quadratic dataset with Parquet files.
Reads binary (uint64 d, int32 h) pairs from raw_gpu*.bin files.

Usage:
  python3 scripts/experiments/class-numbers/upload_hf.py
"""

import struct
import os
import glob
import sys

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, create_repo


REPO_ID = "cahlen/class-numbers-real-quadratic"
RANGE_NAME = "1e9_to_1e10"
BIN_DIR = "data/class-numbers"
PARQUET_DIR = f"data/class-numbers/parquet/{RANGE_NAME}"
CHUNK_SIZE = 50_000_000  # 50M records per parquet file


def read_binary_chunk(f, n=CHUNK_SIZE):
    """Read up to n (d, h) pairs from an open binary file."""
    disc = []
    cls = []
    for _ in range(n):
        buf = f.read(12)
        if len(buf) < 12:
            break
        d = struct.unpack('Q', buf[:8])[0]
        h = struct.unpack('i', buf[8:12])[0]
        if h > 0:
            disc.append(d)
            cls.append(h)
    return disc, cls


def main():
    # Find binary files for this range
    pattern = os.path.join(BIN_DIR, "raw_gpu*_1*_*.bin")
    bin_files = sorted(glob.glob(pattern))
    # Filter to only [1e9, 1e10] range files
    bin_files = [f for f in bin_files if "1000000000" in f or "2125000000" in f
                 or "3250000000" in f or "4375000000" in f or "5500000000" in f
                 or "6625000000" in f or "7750000000" in f or "8875000000" in f]

    if not bin_files:
        print("No binary files found!")
        sys.exit(1)

    total_bytes = sum(os.path.getsize(f) for f in bin_files)
    total_records = total_bytes // 12
    print(f"Found {len(bin_files)} binary files")
    print(f"Total: {total_records:,} records ({total_bytes / 1e9:.1f} GB)")

    for f in bin_files:
        n = os.path.getsize(f) // 12
        print(f"  {os.path.basename(f)}: {n:,} records")

    # Convert to Parquet
    os.makedirs(PARQUET_DIR, exist_ok=True)
    part = 0
    global_count = 0

    for bin_file in bin_files:
        print(f"\nReading {os.path.basename(bin_file)}...")
        with open(bin_file, 'rb') as f:
            while True:
                disc, cls = read_binary_chunk(f)
                if not disc:
                    break

                table = pa.table({
                    'discriminant': pa.array(disc, type=pa.uint64()),
                    'class_number': pa.array(cls, type=pa.int32())
                })

                out_path = os.path.join(PARQUET_DIR, f"part-{part:05d}.parquet")
                pq.write_table(table, out_path, compression='zstd')
                global_count += len(disc)
                print(f"  Wrote {out_path}: {len(disc):,} records "
                      f"({global_count:,} total, "
                      f"{100*global_count/total_records:.1f}%)")
                part += 1

    print(f"\n{'='*60}")
    print(f"Conversion complete: {global_count:,} records in {part} parquet files")
    print(f"Parquet dir: {PARQUET_DIR}")

    # Check parquet sizes
    parquet_size = sum(os.path.getsize(os.path.join(PARQUET_DIR, f))
                       for f in os.listdir(PARQUET_DIR) if f.endswith('.parquet'))
    print(f"Total parquet size: {parquet_size / 1e9:.2f} GB "
          f"(compression ratio: {total_bytes / parquet_size:.1f}x)")

    # Create HF repo
    print(f"\nCreating dataset repo: {REPO_ID}")
    create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

    api = HfApi()

    # Upload README
    readme_path = "scripts/experiments/class-numbers/HF_README.md"
    print(f"Uploading README...")
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    # Upload parquet files
    print(f"Uploading {part} parquet files to data/{RANGE_NAME}/...")
    api.upload_folder(
        folder_path=PARQUET_DIR,
        path_in_repo=f"data/{RANGE_NAME}",
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    print(f"\nDone! Dataset: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
