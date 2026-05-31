#!/usr/bin/env python3
"""Summarize Kerr ν-sweep CSVs vs random baseline."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

RUNS = [
    ("kerr", "1e-03", "bkm3d_n256_nu1e-03_steps1000_kerr.csv"),
    ("random", "1e-03", "bkm3d_n256_nu1e-03_steps1000_random.csv"),
    ("kerr", "1e-04", "bkm3d_n256_nu1e-04_steps2000_kerr.csv"),
    ("random", "1e-04", "bkm3d_n256_nu1e-04_steps2000.csv"),
    ("kerr", "1e-05", "bkm3d_n256_nu1e-05_steps2000_kerr.csv"),
]


def load(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items() if k != "step"})
    return rows


def main() -> int:
    print(f"{'IC':<8} {'nu':<8} {'t_final':>8} {'max|ω|':>10} {'BKM':>10} {'rows':>5}")
    print("-" * 55)
    missing = 0
    for ic, nu, name in RUNS:
        path = RESULTS / name
        if not path.exists():
            print(f"{ic:<8} {nu:<8} {'MISSING':>8}")
            missing += 1
            continue
        rows = load(path)
        last = rows[-1]
        mx = max(r["max_vorticity"] for r in rows)
        print(
            f"{ic:<8} {nu:<8} {last['time']:8.3f} {mx:10.4f} {last['bkm_cumulative']:10.4f} {len(rows):5d}"
        )
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
