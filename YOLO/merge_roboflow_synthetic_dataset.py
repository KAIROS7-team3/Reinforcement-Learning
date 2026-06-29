#!/usr/bin/env python3
"""
Roboflow top-view segmentation dataset과 Isaac synthetic Replicator dataset을
동일 class schema로 merge합니다.

Roboflow class (배포 기준):
  0 multi_tool
  1 ratchet_wrench
  2 screwdriver
  3 socket_19mm
  4 spanner_16mm
  5 utility_knife
"""

from __future__ import annotations

import argparse
import shutil
import struct
import zlib
from pathlib import Path

import yaml
from tqdm import tqdm

from convert_replicator_to_yolo_seg import (
    DEFAULT_CLASS_ORDER,
    ReplicatorToYOLOSegmentationConverter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOFLOW_DIR = Path(
    "/home/user/Downloads/home/iys/Final_project/datasets/tools/top_view_seg"
)
DEFAULT_SYNTHETIC_RAW_DIR = (
    PROJECT_ROOT / "YOLO" / "replicator_output" / "tool_table_raw_topview_rsd455"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "YOLO" / "yolo_seg_dataset_mixed"

ROBOFLOW_CLASS_NAMES = [
    "multi_tool",
    "ratchet_wrench",
    "screwdriver",
    "socket_19mm",
    "spanner_16mm",
    "utility_knife",
]

def verify_png(path: Path) -> bool:
    """PNG 헤더/IDAT CRC를 검사합니다. 손상 파일은 학습 중 libpng CRC error를 유발합니다."""
    try:
        data = path.read_bytes()
    except OSError:
        return False
    if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return False

    pos = 8
    idat = b""
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        ctype = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        if ctype == b"IEND":
            break
        if ctype == b"IDAT":
            idat += chunk
        pos += 12 + length

    try:
        zlib.decompress(idat)
    except zlib.error:
        return False
    return True


SYNTHETIC_TO_ROBOFLOW = {
    "allen_key_tool_assembly": "multi_tool",
    "husky_socket_wrench": "ratchet_wrench",
    "screw_driver": "screwdriver",
    "socket": "socket_19mm",
    "spanner_16mm": "spanner_16mm",
    "paper_cutter": "utility_knife",
}


class SyntheticToRoboflowConverter(ReplicatorToYOLOSegmentationConverter):
    """Synthetic Replicator 데이터를 Roboflow class ID로 변환합니다."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, class_order=DEFAULT_CLASS_ORDER, **kwargs)
        self.roboflow_class_to_id = {
            name: idx for idx, name in enumerate(ROBOFLOW_CLASS_NAMES)
        }

    def load_class_mappings(self) -> None:
        super().load_class_mappings()
        missing = set(SYNTHETIC_TO_ROBOFLOW) - set(self.class_to_yolo)
        if missing:
            raise ValueError(f"Synthetic class mapping 누락: {sorted(missing)}")

    def convert_frame(self, seg_path: Path) -> list[str]:
        from PIL import Image
        import numpy as np

        seg_img = np.array(Image.open(seg_path))
        if len(seg_img.shape) == 3 and seg_img.shape[2] == 3:
            rgba_img = np.zeros((seg_img.shape[0], seg_img.shape[1], 4), dtype=np.uint8)
            rgba_img[:, :, :3] = seg_img
            rgba_img[:, :, 3] = 255
            seg_img = rgba_img

        yolo_annotations: list[str] = []
        for rgba, class_name in self.rgba_to_class.items():
            roboflow_name = SYNTHETIC_TO_ROBOFLOW.get(class_name)
            if roboflow_name is None:
                continue
            yolo_class = self.roboflow_class_to_id[roboflow_name]

            color = np.array(rgba, dtype=seg_img.dtype)
            instance_mask = np.all(seg_img == color, axis=-1).astype(np.uint8)
            if instance_mask.sum() == 0:
                continue

            for polygon in self.extract_polygons_from_mask(instance_mask):
                coords = " ".join(f"{coord:.6f}" for coord in polygon)
                yolo_annotations.append(f"{yolo_class} {coords}")

        return yolo_annotations

    def create_yaml_config(self) -> None:
        config = {
            "path": str(self.output_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {idx: name for idx, name in enumerate(ROBOFLOW_CLASS_NAMES)},
            "nc": len(ROBOFLOW_CLASS_NAMES),
        }
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def setup_output_dirs(output_dir: Path) -> None:
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_roboflow_split(
    roboflow_dir: Path,
    output_dir: Path,
    roboflow_split: str,
    output_split: str,
) -> int:
    image_dir = roboflow_dir / roboflow_split / "images"
    label_dir = roboflow_dir / roboflow_split / "labels"
    if not image_dir.exists():
        raise FileNotFoundError(f"Roboflow split not found: {image_dir}")

    count = 0
    skipped = 0
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file():
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        if image_path.suffix.lower() == ".png" and not verify_png(image_path):
            print(f"  [skip] corrupted PNG: {image_path.name}")
            skipped += 1
            continue

        shutil.copy2(image_path, output_dir / "images" / output_split / image_path.name)
        shutil.copy2(label_path, output_dir / "labels" / output_split / label_path.name)
        count += 1
    if skipped:
        print(f"  Roboflow {roboflow_split}: skipped {skipped} corrupted image(s)")
    return count


def convert_synthetic_split(
    synthetic_raw_dir: Path,
    output_dir: Path,
    split: str,
    rgb_files: list[Path],
    seg_dir: Path,
    *,
    name_prefix: str = "syn_",
) -> tuple[int, int]:
    converter = SyntheticToRoboflowConverter(
        isaac_dir=synthetic_raw_dir,
        output_dir=output_dir,
    )
    converter.load_class_mappings()

    image_count = 0
    polygon_count = 0
    skipped = 0
    for rgb_path in tqdm(rgb_files, desc=f"synthetic->{split}"):
        if not verify_png(rgb_path):
            print(f"  [skip] corrupted synthetic PNG: {rgb_path.name}")
            skipped += 1
            continue

        frame_num = int(rgb_path.stem.split("_")[-1])
        seg_path = seg_dir / f"instance_segmentation_{frame_num:04d}.png"
        if not seg_path.exists():
            continue
        if not verify_png(seg_path):
            print(f"  [skip] corrupted segmentation PNG: {seg_path.name}")
            skipped += 1
            continue

        annotations = converter.convert_frame(seg_path)
        if not annotations:
            continue

        out_stem = f"{name_prefix}{rgb_path.stem}"
        shutil.copy2(rgb_path, output_dir / "images" / split / f"{out_stem}.png")
        with open(output_dir / "labels" / split / f"{out_stem}.txt", "w") as f:
            f.write("\n".join(annotations))

        image_count += 1
        polygon_count += len(annotations)

    if skipped:
        print(f"  synthetic->{split}: skipped {skipped} corrupted frame(s)")
    return image_count, polygon_count


def merge_synthetic_source(
    synthetic_raw_dir: Path,
    output_dir: Path,
    *,
    synthetic_val_ratio: float,
    max_synthetic_frames: int | None,
    name_prefix: str,
) -> dict[str, int]:
    converter = SyntheticToRoboflowConverter(
        isaac_dir=synthetic_raw_dir,
        output_dir=output_dir,
    )
    rgb_files, seg_dir, _ = converter.find_replicator_files()
    if max_synthetic_frames is not None:
        rgb_files = rgb_files[:max_synthetic_frames]

    num_val = int(len(rgb_files) * synthetic_val_ratio)
    synthetic_splits = {
        "train": rgb_files[num_val:],
        "val": rgb_files[:num_val],
    }

    stats = {"synthetic_train": 0, "synthetic_val": 0}
    for split, files in synthetic_splits.items():
        if not files:
            continue
        image_count, _ = convert_synthetic_split(
            synthetic_raw_dir,
            output_dir,
            split,
            files,
            seg_dir,
            name_prefix=name_prefix,
        )
        stats[f"synthetic_{split}"] = image_count
    return stats


def merge_datasets(
    roboflow_dir: Path,
    synthetic_raw_dirs: list[Path],
    output_dir: Path,
    synthetic_val_ratio: float = 0.1,
    max_synthetic_frames: int | None = None,
) -> dict[str, int]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    setup_output_dirs(output_dir)

    stats = {
        "roboflow_train": copy_roboflow_split(roboflow_dir, output_dir, "train", "train"),
        "roboflow_val": copy_roboflow_split(roboflow_dir, output_dir, "valid", "val"),
        "roboflow_test": copy_roboflow_split(roboflow_dir, output_dir, "test", "test"),
        "synthetic_train": 0,
        "synthetic_val": 0,
    }

    for source_idx, synthetic_raw_dir in enumerate(synthetic_raw_dirs):
        source_stats = merge_synthetic_source(
            synthetic_raw_dir,
            output_dir,
            synthetic_val_ratio=synthetic_val_ratio,
            max_synthetic_frames=max_synthetic_frames,
            name_prefix=f"syn{source_idx}_",
        )
        stats["synthetic_train"] += source_stats["synthetic_train"]
        stats["synthetic_val"] += source_stats["synthetic_val"]
        stats[f"synthetic_train_{source_idx}"] = source_stats["synthetic_train"]
        stats[f"synthetic_val_{source_idx}"] = source_stats["synthetic_val"]

    SyntheticToRoboflowConverter(
        isaac_dir=synthetic_raw_dirs[0],
        output_dir=output_dir,
    ).create_yaml_config()
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roboflow + Isaac synthetic YOLO segmentation dataset merge"
    )
    parser.add_argument("--roboflow-dir", type=Path, default=DEFAULT_ROBOFLOW_DIR)
    parser.add_argument(
        "--synthetic-raw-dir",
        type=Path,
        nargs="+",
        default=[DEFAULT_SYNTHETIC_RAW_DIR],
        help="Replicator raw 디렉토리 (여러 개 지정 가능)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--synthetic-val-ratio",
        type=float,
        default=0.1,
        help="Synthetic 데이터 중 val로 넣을 비율 (나머지는 train)",
    )
    parser.add_argument("--max-synthetic-frames", type=int, help="Synthetic 변환 프레임 상한")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.roboflow_dir = args.roboflow_dir.resolve()
    args.synthetic_raw_dir = [path.resolve() for path in args.synthetic_raw_dir]
    args.output_dir = args.output_dir.resolve()

    print("Roboflow dir:", args.roboflow_dir)
    print("Synthetic raw dirs:")
    for path in args.synthetic_raw_dir:
        print(f"  - {path}")
    print("Output dir:", args.output_dir)
    print("Class schema:", ROBOFLOW_CLASS_NAMES)
    print()

    stats = merge_datasets(
        roboflow_dir=args.roboflow_dir,
        synthetic_raw_dirs=args.synthetic_raw_dir,
        output_dir=args.output_dir,
        synthetic_val_ratio=args.synthetic_val_ratio,
        max_synthetic_frames=args.max_synthetic_frames,
    )

    print("\n[Merge 완료]")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"  dataset.yaml: {args.output_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
