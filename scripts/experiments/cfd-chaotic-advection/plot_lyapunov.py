#!/usr/bin/env python3
"""Plot Λ(K) from standard_map_lyapunov CSV → SVG for bigcompute.science."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def load_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "K": float(row["K"]),
                    "mean": float(row["mean_lyapunov"]),
                    "std": float(row["std_lyapunov"]),
                    "frac": float(row["fraction_positive"]),
                }
            )
    return rows


def svg_line_chart(
    rows: list[dict[str, float]],
    *,
    width: int = 920,
    height: int = 420,
    k_crit: float = 0.971635406,
) -> str:
    margin = dict(l=70, r=30, t=40, b=55)
    plot_w = width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]

    ks = [r["K"] for r in rows]
    means = [r["mean"] for r in rows]
    stds = [r["std"] for r in rows]
    k_min, k_max = min(ks), max(ks)
    y_max = max(means[i] + stds[i] for i in range(len(rows)))
    y_max = max(y_max, 0.05) * 1.08

    def x_px(k: float) -> float:
        return margin["l"] + (k - k_min) / (k_max - k_min) * plot_w

    def y_px(v: float) -> float:
        return margin["t"] + plot_h - (v / y_max) * plot_h

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Standard map Lyapunov exponent vs K">',
        '<rect width="100%" height="100%" fill="#0b1020"/>',
        f'<text x="{width/2:.1f}" y="24" fill="#e2e8f0" font-family="system-ui,sans-serif" '
        f'font-size="16" text-anchor="middle">Chirikov standard map — mean Λ(K)</text>',
    ]

    # axes
    x0, y0 = margin["l"], margin["t"] + plot_h
    x1 = margin["l"] + plot_w
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#64748b" stroke-width="1"/>')
    parts.append(f'<line x1="{x0}" y1="{margin["t"]}" x2="{x0}" y2="{y0}" stroke="#64748b" stroke-width="1"/>')

    # K_crit vertical
    xc = x_px(k_crit)
    parts.append(
        f'<line x1="{xc:.2f}" y1="{margin["t"]}" x2="{xc:.2f}" y2="{y0}" '
        f'stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="6 4"/>'
    )
    parts.append(
        f'<text x="{xc+6:.1f}" y="{margin["t"]+16}" fill="#fbbf24" font-size="12" '
        f'font-family="system-ui,sans-serif">K_crit ≈ {k_crit:.3f}</text>'
    )

    # std band polygon
    upper = " ".join(f"{x_px(ks[i]):.2f},{y_px(means[i]+stds[i]):.2f}" for i in range(len(rows)))
    lower = " ".join(
        f"{x_px(ks[i]):.2f},{y_px(max(0.0, means[i]-stds[i])):.2f}" for i in reversed(range(len(rows)))
    )
    parts.append(f'<polygon points="{upper} {lower}" fill="#38bdf8" fill-opacity="0.18"/>')

    # mean line
    path = "M " + " L ".join(f"{x_px(ks[i]):.2f},{y_px(means[i]):.2f}" for i in range(len(rows)))
    parts.append(f'<path d="{path}" fill="none" stroke="#38bdf8" stroke-width="2.5"/>')

    # labels
    parts.append(
        f'<text x="{width/2:.1f}" y="{height-18}" fill="#94a3b8" font-size="13" '
        f'text-anchor="middle" font-family="system-ui,sans-serif">K (coupling)</text>'
    )
    parts.append(
        f'<text x="18" y="{margin["t"]+plot_h/2:.1f}" fill="#94a3b8" font-size="13" '
        f'font-family="system-ui,sans-serif" transform="rotate(-90 18 {margin["t"]+plot_h/2:.1f})">'
        f'mean Λ(K)</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    args = p.parse_args()
    rows = load_csv(args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg_line_chart(rows))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
