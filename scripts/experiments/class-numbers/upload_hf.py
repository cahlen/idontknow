#!/usr/bin/env python3
"""
Upload class number raw data to Hugging Face as a dataset.

Dataset: cahlen/class-numbers-real-quadratic
Format: Parquet (converted from binary for efficient access)
Schema: {discriminant: uint64, class_number: int32}

Usage:
  python3 upload_hf.py data/class-numbers/raw_gpu*.bin
"""

import sys
import struct
import os
import glob

def read_binary(path):
    """Read (d, h) pairs from binary file."""
    records = []
    with open(path, 'rb') as f:
        data = f.read()
    n = len(data) // 12
    for i in range(n):
        d = struct.unpack('Q', data[i*12:i*12+8])[0]
        h = struct.unpack('i', data[i*12+8:i*12+12])[0]
        records.append((d, h))
    return records

def main():
    bin_files = sorted(glob.glob("data/class-numbers/raw_gpu*.bin"))
    if not bin_files:
        print("No binary files found in data/class-numbers/")
        sys.exit(1)

    print(f"Found {len(bin_files)} binary files")
    total_records = 0
    for f in bin_files:
        size = os.path.getsize(f)
        n = size // 12
        total_records += n
        print(f"  {f}: {n:,} records ({size/1e9:.1f} GB)")

    print(f"\nTotal: {total_records:,} records")
    print(f"Converting to Parquet for Hugging Face upload...")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Process in chunks to avoid memory issues
        chunk_size = 10_000_000  # 10M records per parquet file
        parquet_dir = "data/class-numbers/parquet"
        os.makedirs(parquet_dir, exist_ok=True)

        part = 0
        buffer_d = []
        buffer_h = []

        for bin_file in bin_files:
            print(f"  Reading {bin_file}...")
            records = read_binary(bin_file)
            for d, h in records:
                buffer_d.append(d)
                buffer_h.append(h)
                if len(buffer_d) >= chunk_size:
                    table = pa.table({
                        'discriminant': pa.array(buffer_d, type=pa.uint64()),
                        'class_number': pa.array(buffer_h, type=pa.int32())
                    })
                    out_path = f"{parquet_dir}/part-{part:05d}.parquet"
                    pq.write_table(table, out_path, compression='zstd')
                    print(f"    Wrote {out_path} ({len(buffer_d):,} records)")
                    buffer_d = []
                    buffer_h = []
                    part += 1

        # Flush remaining
        if buffer_d:
            table = pa.table({
                'discriminant': pa.array(buffer_d, type=pa.uint64()),
                'class_number': pa.array(buffer_h, type=pa.int32())
            })
            out_path = f"{parquet_dir}/part-{part:05d}.parquet"
            pq.write_table(table, out_path, compression='zstd')
            print(f"    Wrote {out_path} ({len(buffer_d):,} records)")

        print(f"\n{part+1} parquet files written to {parquet_dir}/")
        print(f"\nTo upload to Hugging Face:")
        print(f"  huggingface-cli upload cahlen/class-numbers-real-quadratic {parquet_dir} data/ --repo-type dataset")

    except ImportError:
        print("pyarrow not installed. Install with: pip3 install pyarrow")
        print("Alternatively, upload the raw .bin files directly.")


if __name__ == "__main__":
    main()
