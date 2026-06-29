#!/usr/bin/env python3
"""Inspect Isaac Lab teleop demo HDF5 (obs/actions per frame).

Examples:
    python3 scripts/inspect_demo_hdf5.py \\
        --dataset ./data/demos/return_tool/dataset.hdf5

    # All frames → CSV (open in LibreOffice / Excel)
    python3 scripts/inspect_demo_hdf5.py \\
        --dataset ./data/demos/return_tool/dataset.hdf5 \\
        --episode demo_0 --export-csv /tmp/demo_0.csv

    # First / last 5 rows in terminal
    python3 scripts/inspect_demo_hdf5.py --dataset ... --head 5 --tail 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import h5py

ACTION_NAMES = [f"joint_{i}" for i in range(1, 7)] + ["rh_r1"]
OBS_NAMES_19 = (
    [f"joint_{i}" for i in range(1, 7)]
    + ["rh_r1"]
    + ["object_x", "object_y", "object_z"]
    + ["target_x", "target_y", "target_z"]
    + [f"tool_id_{i}" for i in range(6)]
)


def _obs_column_names(dim: int) -> list[str]:
    if dim == len(OBS_NAMES_19):
        return list(OBS_NAMES_19)
    return [f"obs_{i}" for i in range(dim)]


def _list_episodes(f: h5py.File) -> list[str]:
    if "data" not in f:
        return []
    return sorted(k for k in f["data"].keys() if k.startswith("demo_"))


def _print_tree(f: h5py.File, episode: str) -> None:
    print(f"file: {f.filename}")
    if "data" in f:
        attrs = dict(f["data"].attrs)
        if attrs:
            print(f"data attrs: {attrs}")
    grp = f["data"][episode]
    print(f"\n=== {episode} ===")
    for key in grp.keys():
        obj = grp[key]
        if isinstance(obj, h5py.Dataset):
            print(f"  {key}: shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"  {key}/  (group)")


def _frame_rows(ep_grp: h5py.Group, names_obs: list[str], names_act: list[str]) -> list[dict]:
    obs = ep_grp["obs"][:]
    actions = ep_grp["actions"][:]
    n = min(obs.shape[0], actions.shape[0])
    rows: list[dict] = []
    for i in range(n):
        row: dict = {"step": i}
        for j, name in enumerate(names_obs):
            row[name] = float(obs[i, j])
        for j, name in enumerate(names_act):
            row[f"act_{name}"] = float(actions[i, j])
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Isaac Lab demo HDF5 frames.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--episode", type=str, default="all", help="demo_0 or all")
    parser.add_argument("--head", type=int, default=0, help="Print first N rows (0=off)")
    parser.add_argument("--tail", type=int, default=0, help="Print last N rows (0=off)")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Print every frame to stdout (404 lines for long demos)",
    )
    parser.add_argument("--export-csv", type=str, default="", help="Write all frames to CSV")
    args = parser.parse_args()

    path = Path(args.dataset)
    if not path.is_file():
        print(f"[ERROR] not found: {path}", file=sys.stderr)
        sys.exit(1)

    with h5py.File(path, "r") as f:
        episodes = _list_episodes(f)
        if not episodes:
            print("[ERROR] no demo_* episodes under /data", file=sys.stderr)
            sys.exit(1)

        targets = episodes if args.episode == "all" else [args.episode]
        for ep in targets:
            if ep not in f["data"]:
                print(f"[ERROR] missing episode: {ep}", file=sys.stderr)
                sys.exit(1)

        for ep in targets:
            _print_tree(f, ep)
            grp = f["data"][ep]
            names_obs = _obs_column_names(int(grp["obs"].shape[1]))
            names_act = ACTION_NAMES if grp["actions"].shape[1] == 7 else [
                f"act_{i}" for i in range(grp["actions"].shape[1])
            ]
            rows = _frame_rows(grp, names_obs, names_act)
            print(f"\nframes: {len(rows)}")
            print("obs layout (19D ReturnTool): joint_1..6, rh_r1, object_xyz, target_xyz, tool_id_0..5")
            print("actions: joint_1..6, rh_r1 (rad)")

            if args.export_csv:
                out = Path(args.export_csv)
                if len(targets) > 1:
                    out = out.with_stem(f"{out.stem}_{ep}")
                fieldnames = list(rows[0].keys()) if rows else []
                with out.open("w", newline="", encoding="utf-8") as cf:
                    writer = csv.DictWriter(cf, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"exported → {out.resolve()}")

            if args.all:
                fieldnames = list(rows[0].keys()) if rows else []
                writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            else:
                if args.head > 0:
                    print(f"\n--- head {args.head} ---")
                    for row in rows[: args.head]:
                        print(row)
                if args.tail > 0:
                    print(f"\n--- tail {args.tail} ---")
                    for row in rows[-args.tail :]:
                        print(row)
                if not args.head and not args.tail and not args.export_csv:
                    print("\nTip: --all | --head 10 --tail 5 | --export-csv /tmp/demo.csv")


if __name__ == "__main__":
    main()
