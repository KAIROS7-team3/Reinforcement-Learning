#!/usr/bin/env python3
"""Plot Isaac Lab teleop demo HDF5 for quality review.

Layouts:
  compare-joint (default) — 2-column scrollable dashboard, all demos overlaid
  per-episode           — 4-panel summary per demo, stacked in one window

Default: single scrollable Tk window (vertical scroll + matplotlib toolbar).
Prints per-channel spread stats (mean/max/end σ across demos).
Use --save to write individual PNGs to --out-dir instead.

Examples:
    python3 scripts/plot_demo_hdf5.py \\
        --dataset ./data/demos/return_tool/dataset.hdf5

    python3 scripts/plot_demo_hdf5.py --save --out-dir ./plots/compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)] + ["rh_r1"]
OBS_IDX = {
    "joints": slice(0, 7),
    "object": slice(7, 10),
    "target": slice(10, 13),
}

OBJECT_NAMES = ["object_x", "object_y", "object_z"]
COMPARE_ROW_HEIGHT_IN = 2.55


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot demo HDF5 trajectories.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/demos/return_tool/dataset.hdf5",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./data/demos/return_tool/plots/compare",
        help="PNG output directory (only with --save).",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="compare-joint",
        choices=("compare-joint", "per-episode"),
    )
    parser.add_argument("--episode", type=str, default="all", help="demo_0 or all")
    parser.add_argument(
        "--show-action",
        action="store_true",
        help="Overlay dashed action lines (compare-joint only)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write PNG files to --out-dir instead of opening the GUI.",
    )
    parser.add_argument(
        "--stats-grid",
        type=int,
        default=101,
        help="Progress samples for cross-demo std (compare-joint).",
    )
    parser.add_argument(
        "--no-std-band",
        action="store_true",
        help="Skip mean±1σ overlay on compare plots (still print stats).",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print spread stats only (no GUI / PNG).",
    )
    return parser.parse_args()


def _list_episodes(f: h5py.File) -> list[str]:
    return sorted(k for k in f["data"].keys() if k.startswith("demo_"))


def _episode_dt(f: h5py.File) -> float:
    dt = 1.0 / 60.0
    if "env_args" in f["data"].attrs:
        try:
            sim = json.loads(f["data"].attrs["env_args"]).get("sim_args", {})
            dt = float(sim.get("dt", dt))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return dt


def _load_all(f: h5py.File, episodes: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    out: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for ep in episodes:
        grp = f["data"][ep]
        obs = np.asarray(grp["obs"][:], dtype=np.float64)
        act = np.asarray(grp["actions"][:], dtype=np.float64)
        n = min(obs.shape[0], act.shape[0])
        out[ep] = (obs[:n], act[:n], n)
    return out


def _progress_axis(n: int) -> np.ndarray:
    if n <= 1:
        return np.array([0.0])
    return np.linspace(0.0, 1.0, n)


def _resample_progress(y: np.ndarray, n_grid: int) -> np.ndarray:
    if y.size == 0:
        return np.full(n_grid, np.nan)
    if y.shape[0] == 1:
        return np.full(n_grid, float(y[0]))
    src_x = np.linspace(0.0, 1.0, y.shape[0])
    dst_x = np.linspace(0.0, 1.0, n_grid)
    return np.interp(dst_x, src_x, y)


def _stack_obs_channel(
    episodes_data: dict[str, tuple[np.ndarray, np.ndarray, int]],
    channel: int,
    *,
    n_grid: int,
) -> np.ndarray:
    rows = [
        _resample_progress(obs[:, channel], n_grid)
        for obs, _act, _n in episodes_data.values()
    ]
    return np.vstack(rows)


def _summarize_stack(stack: np.ndarray) -> dict[str, np.ndarray | float]:
    """stack: (n_demos, n_grid) → mean/std curves + scalar spread metrics."""
    if stack.shape[0] < 2:
        zeros = np.zeros(stack.shape[1])
        return {
            "mean_curve": np.mean(stack, axis=0),
            "std_curve": zeros,
            "mean_std": 0.0,
            "max_std": 0.0,
            "start_std": 0.0,
            "end_std": 0.0,
        }
    std_curve = np.std(stack, axis=0, ddof=1)
    return {
        "mean_curve": np.mean(stack, axis=0),
        "std_curve": std_curve,
        "mean_std": float(np.mean(std_curve)),
        "max_std": float(np.max(std_curve)),
        "start_std": float(std_curve[0]),
        "end_std": float(std_curve[-1]),
    }


def _compute_compare_stats(
    episodes_data: dict[str, tuple[np.ndarray, np.ndarray, int]],
    *,
    n_grid: int,
) -> tuple[dict[str, dict], np.ndarray, dict[str, float]]:
    grid = np.linspace(0.0, 1.0, n_grid)
    stats: dict[str, dict] = {}

    for j, name in enumerate(JOINT_NAMES):
        stats[name] = _summarize_stack(_stack_obs_channel(episodes_data, j, n_grid=n_grid))

    for k, name in enumerate(OBJECT_NAMES):
        stats[name] = _summarize_stack(
            _stack_obs_channel(episodes_data, OBS_IDX["object"].start + k, n_grid=n_grid)
        )

    end_xy = np.array(
        [
            [obs[-1, OBS_IDX["object"].start], obs[-1, OBS_IDX["object"].start + 1]]
            for obs, _act, _n in episodes_data.values()
        ]
    )
    endpoint = {
        "object_xy_end_x_std": float(np.std(end_xy[:, 0], ddof=1)) if len(end_xy) > 1 else 0.0,
        "object_xy_end_y_std": float(np.std(end_xy[:, 1], ddof=1)) if len(end_xy) > 1 else 0.0,
        "object_xy_end_xy_std": float(np.linalg.norm(np.std(end_xy, axis=0, ddof=1)))
        if len(end_xy) > 1
        else 0.0,
    }
    return stats, grid, endpoint


def _print_compare_stats(
    stats: dict[str, dict],
    endpoint: dict[str, float],
    *,
    n_demos: int,
) -> None:
    print(f"[INFO] Demo spread — {n_demos} demos, progress-aligned σ (lower = more uniform)")
    print(f"{'signal':<12} {'mean_σ':>8} {'max_σ':>8} {'start_σ':>8} {'end_σ':>8}  unit")
    print("-" * 58)
    for name in JOINT_NAMES:
        s = stats[name]
        print(
            f"{name:<12} {s['mean_std']:8.4f} {s['max_std']:8.4f} "
            f"{s['start_std']:8.4f} {s['end_std']:8.4f}  rad"
        )
    for name in OBJECT_NAMES:
        s = stats[name]
        print(
            f"{name:<12} {s['mean_std']:8.4f} {s['max_std']:8.4f} "
            f"{s['start_std']:8.4f} {s['end_std']:8.4f}  m"
        )
    print("-" * 58)
    print(
        f"object end XY spread: σx={endpoint['object_xy_end_x_std']:.4f} m, "
        f"σy={endpoint['object_xy_end_y_std']:.4f} m "
        f"(combined={endpoint['object_xy_end_xy_std']:.4f} m)"
    )


def _overlay_std_band(ax, grid: np.ndarray, summary: dict) -> None:
    mean = summary["mean_curve"]
    std = summary["std_curve"]
    ax.fill_between(
        grid,
        mean - std,
        mean + std,
        color="0.15",
        alpha=0.18,
        linewidth=0.0,
        zorder=2,
    )
    ax.plot(grid, mean, color="0.15", linewidth=1.5, linestyle="-", alpha=0.85, zorder=3)


def _spread_title(base: str, summary: dict | None) -> str:
    if summary is None:
        return base
    return (
        f"{base}  |  meanσ={summary['mean_std']:.3f} "
        f"maxσ={summary['max_std']:.3f} endσ={summary['end_std']:.3f}"
    )


def _demo_colors(cmap, n: int) -> np.ndarray:
    return cmap(np.linspace(0, 1, max(n, 1)))


def _draw_joint_compare(
    ax,
    episodes_data: dict[str, tuple[np.ndarray, np.ndarray, int]],
    j: int,
    name: str,
    colors: np.ndarray,
    *,
    show_action: bool,
    grid: np.ndarray | None = None,
    stats: dict | None = None,
    show_std_band: bool = True,
) -> None:
    for (ep, (obs, act, n)), color in zip(episodes_data.items(), colors):
        prog = _progress_axis(n)
        ax.plot(prog, obs[:, j], color=color, linewidth=1.4, label=f"{ep} ({n} st)", zorder=1)
        if show_action and j < act.shape[1]:
            ax.plot(prog, act[:, j], color=color, linestyle="--", alpha=0.4, linewidth=0.9, zorder=1)
    if show_std_band and grid is not None and stats is not None:
        _overlay_std_band(ax, grid, stats)
    ax.set_ylabel("rad")
    subtitle = "obs" + (", act--" if show_action else "") + (", gray=mean±1σ" if show_std_band and stats else "")
    ax.set_title(_spread_title(f"{name} ({subtitle})", stats), fontsize=10, loc="left")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    if j == 0:
        ax.legend(fontsize=7, ncol=min(5, len(episodes_data)), loc="upper right")


def _draw_object_axis(
    ax,
    episodes_data: dict[str, tuple[np.ndarray, np.ndarray, int]],
    k: int,
    name: str,
    colors: np.ndarray,
    *,
    grid: np.ndarray | None = None,
    stats: dict | None = None,
    show_std_band: bool = True,
) -> None:
    for (ep, (obs, _act, n)), color in zip(episodes_data.items(), colors):
        prog = _progress_axis(n)
        ax.plot(prog, obs[:, OBS_IDX["object"].start + k], color=color, linewidth=1.4, label=ep, zorder=1)
    if show_std_band and grid is not None and stats is not None:
        _overlay_std_band(ax, grid, stats)
    ax.set_ylabel("m")
    subtitle = "world" + (", gray=mean±1σ" if show_std_band and stats else "")
    ax.set_title(_spread_title(f"{name} ({subtitle})", stats), fontsize=10, loc="left")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)


def _draw_object_xy(
    ax,
    episodes_data: dict[str, tuple[np.ndarray, np.ndarray, int]],
    colors: np.ndarray,
) -> None:
    for (ep, (obs, _act, n)), color in zip(episodes_data.items(), colors):
        obj = obs[:, OBS_IDX["object"]]
        tgt = obs[:, OBS_IDX["target"]]
        ax.plot(obj[:, 0], obj[:, 1], color=color, linewidth=1.3, label=f"{ep} ({n} st)")
        ax.scatter(obj[-1, 0], obj[-1, 1], color=color, s=40, marker="x", zorder=5)
        ax.scatter(tgt[0, 0], tgt[0, 1], color=color, s=50, marker="*", alpha=0.45, zorder=5)
    ax.set_xlabel("object x (m)")
    ax.set_ylabel("object y (m)")
    ax.set_title("object XY (×=end, ★=target)", fontsize=10, loc="left")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def _build_compare_dashboard(
    plt,
    episodes_data: dict[str, tuple[np.ndarray, np.ndarray, int]],
    *,
    show_action: bool,
    n_grid: int = 101,
    show_std_band: bool = True,
):
    from matplotlib.gridspec import GridSpec

    stats, grid, endpoint = _compute_compare_stats(episodes_data, n_grid=n_grid)
    if len(episodes_data) >= 2:
        _print_compare_stats(stats, endpoint, n_demos=len(episodes_data))

    # 7 joints + 3 object axes in 5×2 grid, object XY full-width (tall) on row 6
    n_rows = 6
    xy_row_ratio = 3.5
    height_ratios = [1.0] * 5 + [xy_row_ratio]
    fig = plt.figure(figsize=(16, COMPARE_ROW_HEIGHT_IN * (5 + xy_row_ratio)))
    gs = GridSpec(
        n_rows,
        2,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.42,
        wspace=0.28,
    )
    colors = _demo_colors(plt.cm.tab10, len(episodes_data))

    progress_axes = []
    slot = 0
    for j, name in enumerate(JOINT_NAMES):
        r, c = divmod(slot, 2)
        ax = fig.add_subplot(gs[r, c])
        _draw_joint_compare(
            ax, episodes_data, j, name, colors,
            show_action=show_action,
            grid=grid,
            stats=stats[name] if len(episodes_data) >= 2 else None,
            show_std_band=show_std_band,
        )
        progress_axes.append((r, ax))
        slot += 1

    for k, name in enumerate(OBJECT_NAMES):
        r, c = divmod(slot, 2)
        ax = fig.add_subplot(gs[r, c])
        _draw_object_axis(
            ax, episodes_data, k, name, colors,
            grid=grid,
            stats=stats[name] if len(episodes_data) >= 2 else None,
            show_std_band=show_std_band,
        )
        progress_axes.append((r, ax))
        slot += 1

    ax = fig.add_subplot(gs[n_rows - 1, :])
    _draw_object_xy(ax, episodes_data, colors)
    if len(episodes_data) >= 2:
        ax.set_title(
            _spread_title("object XY (×=end, ★=target)", None)
            + f"  |  end spread σx={endpoint['object_xy_end_x_std']:.3f} "
            f"σy={endpoint['object_xy_end_y_std']:.3f} m",
            fontsize=10,
            loc="left",
        )

    last_progress_row = n_rows - 2
    for r, ax in progress_axes:
        if r < last_progress_row:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("episode progress (0 → 1)")

    n_eps = len(episodes_data)
    fig.suptitle(
        f"Demo quality compare — {n_eps} episode(s)",
        fontsize=13,
        y=0.998,
    )
    return fig


def _draw_episode_panels(
    axes,
    ep: str,
    obs: np.ndarray,
    actions: np.ndarray,
    dt: float,
) -> None:
    steps = np.arange(obs.shape[0])
    time_s = steps * dt
    joints_obs = obs[:, OBS_IDX["joints"]]
    joints_act = actions[:, :7]
    obj = obs[:, OBS_IDX["object"]]
    tgt = obs[:, OBS_IDX["target"]]

    ax = axes[0, 0]
    for j, name in enumerate(JOINT_NAMES):
        ax.plot(time_s, joints_obs[:, j], label=f"{name} obs", linewidth=1.1)
        ax.plot(time_s, joints_act[:, j], "--", alpha=0.6, linewidth=0.8)
    ax.set_ylabel("rad")
    ax.set_title("Joints (solid=obs, dashed=act)", fontsize=9, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=2, loc="upper right")

    ax = axes[0, 1]
    ax.plot(time_s, joints_obs[:, 6], "b-", label="rh_r1 obs", linewidth=1.4)
    ax.plot(time_s, joints_act[:, 6], "r--", label="rh_r1 act", linewidth=1.1)
    ax.set_title("Gripper", fontsize=9, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(obj[:, 0], obj[:, 1], "b-", linewidth=1.4, label="path")
    ax.scatter(obj[0, 0], obj[0, 1], c="green", s=45, zorder=5, label="start")
    ax.scatter(obj[-1, 0], obj[-1, 1], c="red", s=45, zorder=5, label="end")
    ax.scatter(tgt[0, 0], tgt[0, 1], c="orange", marker="*", s=90, zorder=5, label="target")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Object XY", fontsize=9, loc="left")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(time_s, obj[:, 2], "b-", label="object z", linewidth=1.4)
    ax.plot(time_s, tgt[:, 2], "orange", linestyle="--", label="target z", linewidth=1.1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("z (m)")
    ax.set_title("Height", fontsize=9, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    axes[0, 0].figure.text(
        0.01,
        axes[0, 0].get_position().y1 + 0.012,
        f"{ep}  ({obs.shape[0]} steps, dt={dt * 1000:.1f} ms)",
        fontsize=11,
        fontweight="bold",
        transform=axes[0, 0].figure.transFigure,
    )


def _build_episode_dashboard(
    plt,
    episodes: list[str],
    data: dict[str, tuple[np.ndarray, np.ndarray, int]],
    dt: float,
):
    from matplotlib.gridspec import GridSpec

    n_eps = len(episodes)
    block_h = 5.2
    fig = plt.figure(figsize=(14, block_h * n_eps))
    outer = GridSpec(n_eps, 1, figure=fig, height_ratios=[1.0] * n_eps, hspace=0.55)

    for i, ep in enumerate(episodes):
        obs, act, _n = data[ep]
        inner = GridSpec.from_subplot_spec(2, 2, subplot_spec=outer[i], hspace=0.32, wspace=0.28)
        axes = np.empty((2, 2), dtype=object)
        for r in range(2):
            for c in range(2):
                axes[r, c] = fig.add_subplot(inner[r, c])
        _draw_episode_panels(axes, ep, obs, act, dt)

    fig.suptitle(f"Per-episode summary — {n_eps} demo(s)", fontsize=13, y=0.998)
    return fig


def _save_compare_individual(
    plt,
    episodes_data: dict[str, tuple[np.ndarray, np.ndarray, int]],
    out_dir: Path,
    *,
    show_action: bool,
    n_grid: int = 101,
    show_std_band: bool = True,
) -> int:
    stats, grid, endpoint = _compute_compare_stats(episodes_data, n_grid=n_grid)
    if len(episodes_data) >= 2:
        _print_compare_stats(stats, endpoint, n_demos=len(episodes_data))

    colors = _demo_colors(plt.cm.tab10, len(episodes_data))
    count = 0

    for j, name in enumerate(JOINT_NAMES):
        fig, ax = plt.subplots(figsize=(11, 5))
        _draw_joint_compare(
            ax, episodes_data, j, name, colors,
            show_action=show_action,
            grid=grid,
            stats=stats[name] if len(episodes_data) >= 2 else None,
            show_std_band=show_std_band,
        )
        ax.set_xlabel("episode progress (0 → 1)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"compare_{name}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path.resolve()}")
        count += 1

    for k, name in enumerate(OBJECT_NAMES):
        fig, ax = plt.subplots(figsize=(11, 5))
        _draw_object_axis(
            ax, episodes_data, k, name, colors,
            grid=grid,
            stats=stats[name] if len(episodes_data) >= 2 else None,
            show_std_band=show_std_band,
        )
        ax.set_xlabel("episode progress (0 → 1)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"compare_{name}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path.resolve()}")
        count += 1

    fig, ax = plt.subplots(figsize=(9, 8))
    _draw_object_xy(ax, episodes_data, colors)
    fig.tight_layout()
    path = out_dir / "compare_object_xy.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.resolve()}")
    return count + 1


def _save_episode_individual(
    plt,
    episodes: list[str],
    data: dict[str, tuple[np.ndarray, np.ndarray, int]],
    dt: float,
    out_dir: Path,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ep in episodes:
        obs, act, _n = data[ep]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        _draw_episode_panels(axes, ep, obs, act, dt)
        fig.tight_layout()
        path = out_dir / f"{ep}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path.resolve()}")
    return len(episodes)


def _show_scrollable(fig, *, title: str) -> None:
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    root = tk.Tk()
    root.title(title)
    root.minsize(880, 560)
    root.geometry("1220x860")

    toolbar_host = tk.Frame(root)
    toolbar_host.pack(side=tk.TOP, fill=tk.X)

    body = tk.Frame(root)
    body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(body, highlightthickness=0)
    vsb = tk.Scrollbar(body, orient=tk.VERTICAL, command=canvas.yview)
    inner = tk.Frame(canvas)

    inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _on_inner_configure(_event: tk.Event) -> None:
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas_configure(event: tk.Event) -> None:
        canvas.itemconfigure(inner_id, width=event.width)

    inner.bind("<Configure>", _on_inner_configure)
    canvas.bind("<Configure>", _on_canvas_configure)

    mpl_canvas = FigureCanvasTkAgg(fig, master=inner)
    mpl_canvas.draw()
    mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    NavigationToolbar2Tk(mpl_canvas, toolbar_host)

    canvas.configure(yscrollcommand=vsb.set)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    vsb.pack(side=tk.RIGHT, fill=tk.Y)

    def _on_scroll(event: tk.Event) -> None:
        if getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
            canvas.yview_scroll(3, "units")
        elif getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            canvas.yview_scroll(-3, "units")

    for widget in (canvas, mpl_canvas.get_tk_widget(), inner):
        widget.bind("<MouseWheel>", _on_scroll)
        widget.bind("<Button-4>", _on_scroll)
        widget.bind("<Button-5>", _on_scroll)

    root.mainloop()
    import matplotlib.pyplot as plt

    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    import matplotlib

    if args.save:
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    dataset = Path(args.dataset)
    if not dataset.is_file():
        print(f"[ERROR] not found: {dataset}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.save else None

    with h5py.File(dataset, "r") as f:
        episodes = _list_episodes(f)
        if not episodes:
            print("[ERROR] no demo_* in file", file=sys.stderr)
            sys.exit(1)
        targets = episodes if args.episode == "all" else [args.episode]
        for ep in targets:
            if ep not in f["data"]:
                print(f"[ERROR] missing {ep}", file=sys.stderr)
                sys.exit(1)

        data = _load_all(f, targets)

        if args.layout == "compare-joint":
            show_std = not args.no_std_band
            if args.stats_only:
                if len(data) >= 2:
                    stats, _grid, endpoint = _compute_compare_stats(data, n_grid=args.stats_grid)
                    _print_compare_stats(stats, endpoint, n_demos=len(data))
                return
            if args.save:
                assert out_dir is not None
                out_dir.mkdir(parents=True, exist_ok=True)
                n = _save_compare_individual(
                    plt, data, out_dir,
                    show_action=args.show_action,
                    n_grid=args.stats_grid,
                    show_std_band=show_std,
                )
                print(f"[INFO] {n} compare plots → {out_dir.resolve()}")
            else:
                fig = _build_compare_dashboard(
                    plt, data,
                    show_action=args.show_action,
                    n_grid=args.stats_grid,
                    show_std_band=show_std,
                )
                print("[INFO] Opening scrollable compare dashboard (close window to exit).")
                _show_scrollable(fig, title=f"Demo compare — {dataset.name}")
        else:
            dt = _episode_dt(f)
            if args.save:
                assert out_dir is not None
                n = _save_episode_individual(plt, targets, data, dt, out_dir)
                print(f"[INFO] {n} episode plot(s) → {out_dir.resolve()}")
            else:
                fig = _build_episode_dashboard(plt, targets, data, dt)
                print("[INFO] Opening scrollable episode dashboard (close window to exit).")
                _show_scrollable(fig, title=f"Demo episodes — {dataset.name}")


if __name__ == "__main__":
    main(_parse_args())
