#!/usr/bin/env python3
"""Overlay Kerr vs random BKM diagnostics for Phase 5 comparison."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "time": float(row["time"]),
                    "max_vorticity": float(row["max_vorticity"]),
                    "bkm": float(row["bkm_cumulative"]),
                }
            )
    return rows


def svg_dual(kerr: list[dict], random: list[dict], *, width: int = 920, height: int = 420) -> str:
    margin = dict(l=70, r=70, t=40, b=55)
    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]
    t_max = max(kerr[-1]["time"], random[-1]["time"])
    w_max = max(max(r["max_vorticity"] for r in kerr + random), 1.0) * 1.05
    b_max = max(kerr[-1]["bkm"], random[-1]["bkm"], 1.0) * 1.05

    def x_px(t: float) -> float:
        return margin["l"] + t / t_max * plot_w

    def y_w(v: float) -> float:
        return margin["t"] + plot_h - v / w_max * plot_h

    def y_b(v: float) -> float:
        return margin["t"] + plot_h - v / b_max * plot_h

    def poly(rows: list[dict], key: str, yfn) -> str:
        return " ".join(f"{x_px(r['time']):.1f},{yfn(r[key]):.1f}" for r in rows)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        f'<text x="{width/2:.0f}" y="24" fill="#e2e8f0" font-family="system-ui,sans-serif" font-size="16" text-anchor="middle">Kerr vs random — 256³, ν=10⁻⁴</text>',
        f'<polyline fill="none" stroke="#f97316" stroke-width="2" points="{poly(kerr, "bkm", y_b)}"/>',
        f'<polyline fill="none" stroke="#38bdf8" stroke-width="2" points="{poly(random, "bkm", y_b)}"/>',
        f'<text x="{width-16}" y="{margin["t"]+14}" fill="#f97316" font-size="11" text-anchor="end">Kerr BKM={kerr[-1]["bkm"]:.2f}</text>',
        f'<text x="{width-16}" y="{margin["t"]+28}" fill="#38bdf8" font-size="11" text-anchor="end">Random BKM={random[-1]["bkm"]:.2f}</text>',
        "</svg>",
    ]
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("kerr_csv", type=Path)
    p.add_argument("random_csv", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg_dual(load(args.kerr_csv), load(args.random_csv)))


if __name__ == "__main__":
    main()
