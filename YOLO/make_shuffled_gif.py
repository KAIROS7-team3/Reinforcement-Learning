#!/usr/bin/env python3
"""폴더 내 PNG를 랜덤 순서로 섞어 GIF 생성 (리사이즈·팔레트 압축 지원)."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

YOLO_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = YOLO_DIR.parent / "docs" / "assets" / "syn0_rgb_sample10"
DEFAULT_OUTPUT = DEFAULT_INPUT / "syn0_rgb_shuffled.gif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PNG 목록을 섞어 GIF 생성")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration", type=float, default=0.5, help="프레임 간격 (초)")
    parser.add_argument("--seed", type=int, default=None, help="재현용 시드 (미지정 시 매번 다름)")
    parser.add_argument("--loop", type=int, default=0, help="0=무한 반복")
    parser.add_argument(
        "--max-size",
        type=int,
        default=640,
        help="긴 변 최대 픽셀 (2048 원본은 640 권장, 0=리사이즈 안 함)",
    )
    parser.add_argument(
        "--colors",
        type=int,
        default=128,
        help="GIF 팔레트 색 수 (2~256, 낮을수록 용량↓)",
    )
    return parser.parse_args()


def resize_frame(image, max_size: int):
    from PIL import Image

    if max_size <= 0 or max(image.size) <= max_size:
        return image.convert("RGB")
    out = image.copy()
    out.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return out.convert("RGB")


def quantize_frames(frames, colors: int):
    from PIL import Image

    colors = max(2, min(256, colors))
    palette = frames[0].quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    return [frame.quantize(palette=palette) for frame in frames], palette


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output = args.output.resolve()

    if args.seed is not None:
        random.seed(args.seed)

    images = sorted(args.input_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG files in {args.input_dir}")

    order = images.copy()
    random.shuffle(order)

    from PIL import Image

    rgb_frames = [resize_frame(Image.open(path), args.max_size) for path in order]
    frames, _palette = quantize_frames(rgb_frames, args.colors)

    duration_ms = max(1, int(round(args.duration * 1000)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=args.loop,
        optimize=True,
    )

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"✓ GIF: {args.output} ({size_mb:.1f} MB)")
    print(f"  frames: {len(order)} @ {args.duration}s")
    if rgb_frames:
        print(f"  resolution: {rgb_frames[0].size[0]}x{rgb_frames[0].size[1]}, colors≤{args.colors}")
    print("  order:")
    for path in order:
        print(f"    {path.name}")


if __name__ == "__main__":
    main()
