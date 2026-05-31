#!/usr/bin/env python3
"""Plot BKM diagnostic CSV → SVG for bigcompute.science."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "step": float(row["step"]),
                    "time": float(row["time"]),
                    "max_vorticity": float(row["max_vorticity"]),
                    "enstrophy": float(row["enstrophy"]),
                    "bkm": float(row["bkm_cumulative"]),
                }
            )
    return rows


def svg_dual_axis(rows: list[dict[str, float]], *, width: int = 920, height: int = 420) -> str:
    margin = dict(l=70, r=70, t=40, b=55)
    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]

    times = [r["time"] for r in rows]
    max_w = [r["max_vorticity"] for r in rows]
    bkm = [r["bkm"] for r in rows]
    t_min, t_max = min(times), max(times)
    w_max = max(max_w) * 1.05 or 1.0
    b_max = max(bkm) * 1.05 or 1.0

    def x_px(t: float) -> float:
        return margin["l"] + (t - t_min) / max(t_max - t_min, 1e-12) * plot_w

    def y_w(v: float) -> float:
        return margin["t"] + plot_h - (v / w_max) * plot_h

    def y_b(v: float) -> float:
        return margin["t"] + plot_h - (v / b_max) * plot_h

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="2D NS max vorticity and BKM integral vs time">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        f'<text x="{width/2:.1f}" y="24" fill="#e2e8f0" font-family="system-ui,sans-serif" '
        f'font-size="16" text-anchor="middle">2D NS — ||ω||∞ and BKM integral</text>',
    ]

    x0, y0 = margin["l"], margin["t"] + plot_h
    x1 = margin["l"] + plot_w
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#64748b"/>')
    parts.append(f'<line x1="{x0}" y1="{margin["t"]}" x2="{x0}" y2="{y0}" stroke="#64748b"/>')

    w_pts = " ".join(f"{x_px(times[i]):.1f},{y_w(max_w[i]):.1f}" for i in range(len(rows)))
    b_pts = " ".join(f"{x_px(times[i]):.1f},{y_b(bkm[i]):.1f}" for i in range(len(rows)))
    parts.append(f'<polyline fill="none" stroke="#38bdf8" stroke-width="2" points="{w_pts}"/>')
    parts.append(f'<polyline fill="none" stroke="#f472b6" stroke-width="2" points="{b_pts}"/>')

    parts.append(
        f'<text x="{margin["l"]}" y="{height-12}" fill="#94a3b8" font-size="12" '
        f'font-family="system-ui,sans-serif">time</text>'
    )
    parts.append(
        f'<text x="12" y="{margin["t"]+plot_h/2:.0f}" fill="#38bdf8" font-size="12" '
        f'font-family="system-ui,sans-serif" transform="rotate(-90 12 {margin["t"]+plot_h/2:.0f})">'
        f'max |ω|</text>'
    )
    parts.append(
        f'<text x="{width-12}" y="{margin["t"]+plot_h/2:.0f}" fill="#f472b6" font-size="12" '
        f'font-family="system-ui,sans-serif" text-anchor="end" '
        f'transform="rotate(90 {width-12} {margin["t"]+plot_h/2:.0f})">∫||ω||∞ dt</text>'
    )
    parts.append(
        f'<text x="{margin["l"]+8}" y="{margin["t"]+16}" fill="#38bdf8" font-size="11" '
        f'font-family="system-ui,sans-serif">||ω||∞</text>'
    )
    parts.append(
        f'<text x="{margin["l"]+8}" y="{margin["t"]+32}" fill="#f472b6" font-size="11" '
        f'font-family="system-ui,sans-serif">BKM</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()
    rows = load_csv(args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg_dual_axis(rows))


if __name__ == "__main__":
    main()
