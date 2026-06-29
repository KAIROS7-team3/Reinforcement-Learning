#!/usr/bin/env python3
"""YOLO training results.csv / results.png를 개별 그래프 PNG로 분리합니다."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


def slugify(name: str) -> str:
    slug = re.sub(r"[^\w]+", "_", name.strip().lower())
    return slug.strip("_")


def read_results_csv(csv_path: Path) -> tuple[list[str], list[dict[str, float]]]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header in {csv_path}")
        rows: list[dict[str, float]] = []
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if key is None:
                    continue
                key = key.strip()
                parsed[key] = float(value)
            rows.append(parsed)
        return [name.strip() for name in reader.fieldnames], rows


def column_values(rows: list[dict[str, float]], col: str) -> list[float]:
    return [row[col] for row in rows]


def plot_series(
    epochs: list[float],
    series: dict[str, list[float]],
    title: str,
    out_path: Path,
    ylabel: str = "value",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for label, values in series.items():
        ax.plot(epochs, values, linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(series) > 1:
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def split_from_csv(run_dir: Path, output_dir: Path | None) -> list[Path]:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found: {csv_path}")

    out_dir = output_dir or (run_dir / "results_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    columns, rows = read_results_csv(csv_path)
    if "epoch" not in columns:
        raise ValueError("results.csv must contain an 'epoch' column")

    epochs = column_values(rows, "epoch")
    saved: list[Path] = []

    paired_groups = [
        ("train/box_loss", "val/box_loss", "Box Loss"),
        ("train/seg_loss", "val/seg_loss", "Seg Loss"),
        ("train/cls_loss", "val/cls_loss", "Cls Loss"),
        ("train/dfl_loss", "val/dfl_loss", "DFL Loss"),
        ("train/sem_loss", "val/sem_loss", "Sem Loss"),
    ]
    for train_col, val_col, title in paired_groups:
        cols = [c for c in (train_col, val_col) if c in columns and sum(abs(v) for v in column_values(rows, c)) > 0]
        if not cols:
            continue
        series = {c.split("/", 1)[-1]: column_values(rows, c) for c in cols}
        out_path = out_dir / f"{slugify(title)}.png"
        plot_series(epochs, series, title, out_path, ylabel="loss")
        saved.append(out_path)

    metric_groups = [
        (["metrics/precision(B)", "metrics/precision(M)"], "Precision"),
        (["metrics/recall(B)", "metrics/recall(M)"], "Recall"),
        (["metrics/mAP50(B)", "metrics/mAP50(M)"], "mAP50"),
        (["metrics/mAP50-95(B)", "metrics/mAP50-95(M)"], "mAP50-95"),
    ]
    for cols, title in metric_groups:
        present = [c for c in cols if c in columns]
        if not present:
            continue
        series = {c.split("/", 1)[-1]: column_values(rows, c) for c in present}
        out_path = out_dir / f"{slugify(title)}.png"
        plot_series(epochs, series, title, out_path, ylabel="score")
        saved.append(out_path)

    lr_cols = [c for c in columns if c.startswith("lr/")]
    if lr_cols:
        series = {c.split("/", 1)[-1]: column_values(rows, c) for c in lr_cols}
        out_path = out_dir / "learning_rate.png"
        plot_series(epochs, series, "Learning Rate", out_path, ylabel="lr")
        saved.append(out_path)

    if "time" in columns:
        out_path = out_dir / "elapsed_time.png"
        plot_series(epochs, {"time": column_values(rows, "time")}, "Elapsed Time", out_path, ylabel="seconds")
        saved.append(out_path)

    used = {col for group in paired_groups for col in group[:2]}
    used.update(col for group, _ in metric_groups for col in group)
    used.update(lr_cols)
    used.update({"epoch", "time"})

    for col in columns:
        if col in used:
            continue
        out_path = out_dir / f"{slugify(col)}.png"
        plot_series(epochs, {col: column_values(rows, col)}, col, out_path)
        saved.append(out_path)

    return saved


def split_from_png(run_dir: Path, output_dir: Path | None, rows: int = 2, cols: int = 5) -> list[Path]:
    from PIL import Image

    png_path = run_dir / "results.png"
    if not png_path.exists():
        raise FileNotFoundError(f"results.png not found: {png_path}")

    out_dir = output_dir or (run_dir / "results_plots_cropped")
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(png_path)
    width, height = image.size
    tile_w = width // cols
    tile_h = height // rows

    saved: list[Path] = []
    idx = 0
    for row in range(rows):
        for col in range(cols):
            left = col * tile_w
            upper = row * tile_h
            right = left + tile_w if col < cols - 1 else width
            lower = upper + tile_h if row < rows - 1 else height
            tile = image.crop((left, upper, right, lower))
            out_path = out_dir / f"results_tile_{idx:02d}_r{row}_c{col}.png"
            tile.save(out_path)
            saved.append(out_path)
            idx += 1
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split YOLO results into individual plot PNGs")
    parser.add_argument("run_dir", type=Path, help="YOLO run directory containing results.csv")
    parser.add_argument("--output-dir", type=Path, help="Output directory (default: <run_dir>/results_plots)")
    parser.add_argument("--from-png", action="store_true", help="Also crop composite results.png into grid tiles")
    parser.add_argument("--rows", type=int, default=2, help="results.png grid rows")
    parser.add_argument("--cols", type=int, default=5, help="results.png grid cols")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()

    saved = split_from_csv(run_dir, args.output_dir)
    print(f"✓ Saved {len(saved)} plot(s) from results.csv")
    for path in saved:
        print(f"  {path}")

    if args.from_png:
        cropped = split_from_png(run_dir, None, rows=args.rows, cols=args.cols)
        print(f"✓ Saved {len(cropped)} cropped tile(s) from results.png")
        for path in cropped:
            print(f"  {path}")


if __name__ == "__main__":
    main()
