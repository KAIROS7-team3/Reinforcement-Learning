#!/usr/bin/env python3
"""
책상 위 공구 6종만 있는 Replicator 데이터셋 생성.

두 가지 모드:
  simple (기본)  — 아이보리 테이블 + 공구 6개만 (로봇/공구함 없음)
  real-scene     — with_camera_backup.usd 실제 작업대, 공구함/로봇 숨김, 공구 6개 고정
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_SIMPLE = YOLO_DIR / "replicator_output" / "desk6_tools"
DEFAULT_OUTPUT_REALSCENE = YOLO_DIR / "replicator_output" / "desk6_tools_realscene"
DEFAULT_SCENE_USD = Path("/home/user/Desktop/with_camera_backup.usd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="책상 위 공구 6종만 있는 Replicator 합성 데이터 생성"
    )
    parser.add_argument(
        "--mode",
        choices=("simple", "real-scene"),
        default="simple",
        help="simple=아이보리 테이블 only, real-scene=실제 workbench USD",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resolution", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, help="출력 폴더 (미지정 시 mode별 기본값)")
    parser.add_argument("--scene-usd", type=Path, default=DEFAULT_SCENE_USD)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="랜덤화/캡처 없이 GUI에서 고정 씬만 확인",
    )
    parser.add_argument(
        "--tight-cluster-prob",
        type=float,
        default=0.0,
        help="real-scene: tight cluster 배치 확률 (기본 0=흩어진 배치)",
    )
    return parser.parse_args()


def run_simple(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or DEFAULT_OUTPUT_SIMPLE
    cmd = [
        sys.executable,
        str(YOLO_DIR / "generate_tool_replicator_dataset.py"),
        "--num-frames",
        str(args.num_frames),
        "--seed",
        str(args.seed),
        "--resolution",
        str(args.resolution),
        "--camera-model",
        "rsd455",
        "--camera-height",
        "0.95",
        "--output-dir",
        str(output_dir),
    ]
    if args.headless:
        cmd.append("--headless")
    print("▶ simple mode: 아이보리 테이블 + 공구 6개만")
    print("  " + " ".join(cmd))
    return subprocess.call(cmd)


def run_real_scene(args: argparse.Namespace) -> int:
    output_dir = args.output_dir or DEFAULT_OUTPUT_REALSCENE
    cmd = [
        sys.executable,
        str(YOLO_DIR / "generate_realscene_replicator_dataset.py"),
        "--num-frames",
        str(args.num_frames),
        "--seed",
        str(args.seed),
        "--resolution",
        str(args.resolution),
        "--scene-usd",
        str(args.scene_usd.resolve()),
        "--output-dir",
        str(output_dir),
        "--min-tools",
        "6",
        "--max-tools",
        "6",
        "--hide-robot",
        "--hide-toolbox",
        "--drawer-open-prob",
        "0.0",
        "--tight-cluster-prob",
        str(args.tight_cluster_prob),
    ]
    if args.headless:
        cmd.append("--headless")
    print("▶ real-scene mode: 실제 작업대 + 공구 6개 (로봇/공구함 숨김)")
    print("  " + " ".join(cmd))
    return subprocess.call(cmd)


def run_preview(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        str(YOLO_DIR / "preview_desk6_tools_scene.py"),
        "--scene-usd",
        str(args.scene_usd.resolve()),
    ]
    print("▶ preview: 작업대 + 공구 6개 고정 배치 (GUI)")
    print("  " + " ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    args = parse_args()
    if args.preview:
        raise SystemExit(run_preview(args))
    if args.mode == "simple":
        raise SystemExit(run_simple(args))
    raise SystemExit(run_real_scene(args))


if __name__ == "__main__":
    main()
