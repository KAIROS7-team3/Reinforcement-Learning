#!/usr/bin/env python3
"""
Replicator flat output을 YOLO segmentation dataset 형식으로 변환합니다.

현재 생성 스크립트의 출력:
  rgb_0000.png
  instance_segmentation_0000.png
  instance_segmentation_semantics_mapping_0000.json
  ...
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ISAAC_DIR = PROJECT_ROOT / "YOLO" / "replicator_output" / "tool_table_raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "YOLO" / "yolo_seg_dataset"
DEFAULT_CLASS_ORDER = [
    "screw_driver",
    "paper_cutter",
    "husky_socket_wrench",
    "allen_key_tool_assembly",
    "spanner_16mm",
    "socket",
]


class ReplicatorToYOLOSegmentationConverter:
    """Isaac Sim instance segmentation 데이터를 YOLO segmentation 형식으로 변환합니다."""

    def __init__(
        self,
        isaac_dir: Path,
        output_dir: Path,
        class_order: list[str] | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.isaac_dir = isaac_dir
        self.output_dir = output_dir
        self.class_order = class_order or DEFAULT_CLASS_ORDER
        self.max_frames = max_frames
        self.rgba_to_class: dict[tuple[int, int, int, int], str] = {}
        self.class_names: dict[int, str] = {}
        self.class_to_yolo: dict[str, int] = {}

    def find_replicator_files(self) -> tuple[list[Path], Path, list[Path]]:
        """flat 구조와 강의 예제식 nested 구조를 모두 지원합니다."""
        nested_rgb_dir = self.isaac_dir / "rgb"
        nested_seg_dir = self.isaac_dir / "instance_segmentation"

        if nested_rgb_dir.exists() and nested_seg_dir.exists():
            rgb_files = sorted(nested_rgb_dir.glob("rgb_*.png"))
            mapping_files = sorted(nested_seg_dir.glob("*semantics_mapping*.json"))
            return self.limit_files(rgb_files), nested_seg_dir, mapping_files

        rgb_files = sorted(self.isaac_dir.glob("rgb_*.png"))
        mapping_files = sorted(self.isaac_dir.glob("instance_segmentation_semantics_mapping_*.json"))
        if rgb_files and mapping_files:
            return self.limit_files(rgb_files), self.isaac_dir, mapping_files

        raise FileNotFoundError(
            "Replicator 데이터를 찾을 수 없습니다. "
            "rgb_0000.png flat 구조 또는 rgb/ + instance_segmentation/ 구조를 확인하세요."
        )

    def limit_files(self, rgb_files: list[Path]) -> list[Path]:
        if self.max_frames is None:
            return rgb_files
        return rgb_files[: self.max_frames]

    def load_class_mappings(self) -> None:
        """semantics mapping 파일에서 RGBA 색상과 클래스 매핑을 로드합니다."""
        _, _, mapping_files = self.find_replicator_files()
        if not mapping_files:
            raise FileNotFoundError("Instance segmentation semantics mapping 파일을 찾을 수 없습니다.")

        class_names = set()
        for mapping_file in mapping_files:
            with open(mapping_file, "r") as f:
                mapping_data = json.load(f)

            for rgba_str, class_info in mapping_data.items():
                class_name = class_info["class"]
                if class_name.upper() in ["BACKGROUND", "UNLABELLED"]:
                    continue

                rgba = ast.literal_eval(rgba_str)
                self.rgba_to_class[rgba] = class_name
                class_names.add(class_name)

        ordered_names = [name for name in self.class_order if name in class_names]
        ordered_names.extend(sorted(class_names - set(ordered_names)))
        self.class_names = {yolo_id: name for yolo_id, name in enumerate(ordered_names)}
        self.class_to_yolo = {name: yolo_id for yolo_id, name in self.class_names.items()}

        print(f"✓ {len(self.class_names)}개 클래스 로드 완료")
        for yolo_id, name in self.class_names.items():
            print(f"  [{yolo_id}] {name}")

    def setup_directories(self) -> None:
        for split in ["train", "val", "test"]:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def extract_polygons_from_mask(self, instance_mask: np.ndarray) -> list[np.ndarray]:
        contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        height, width = instance_mask.shape

        for contour in contours:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            if len(simplified) < 3:
                continue

            polygon = simplified.reshape(-1, 2).astype(np.float32)
            polygon[:, 0] /= width
            polygon[:, 1] /= height
            polygons.append(polygon.flatten())

        return polygons

    def convert_frame(self, seg_path: Path) -> list[str]:
        seg_img = np.array(Image.open(seg_path))
        if len(seg_img.shape) == 3 and seg_img.shape[2] == 3:
            rgba_img = np.zeros((seg_img.shape[0], seg_img.shape[1], 4), dtype=np.uint8)
            rgba_img[:, :, :3] = seg_img
            rgba_img[:, :, 3] = 255
            seg_img = rgba_img

        yolo_annotations = []
        for rgba, class_name in self.rgba_to_class.items():
            yolo_class = self.class_to_yolo.get(class_name)
            if yolo_class is None:
                continue

            color = np.array(rgba, dtype=seg_img.dtype)
            instance_mask = np.all(seg_img == color, axis=-1).astype(np.uint8)
            if instance_mask.sum() == 0:
                continue

            for polygon in self.extract_polygons_from_mask(instance_mask):
                coords = " ".join(f"{coord:.6f}" for coord in polygon)
                yolo_annotations.append(f"{yolo_class} {coords}")

        return yolo_annotations

    def convert_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.2) -> dict[str, int]:
        print("\n[데이터 변환 시작]")
        self.load_class_mappings()
        self.setup_directories()

        rgb_files, seg_dir, _ = self.find_replicator_files()
        print(f"✓ {len(rgb_files)}개 이미지 발견")

        num_train = int(len(rgb_files) * train_ratio)
        num_val = int(len(rgb_files) * val_ratio)
        splits = {
            "train": rgb_files[:num_train],
            "val": rgb_files[num_train : num_train + num_val],
            "test": rgb_files[num_train + num_val :],
        }
        print(
            "✓ 데이터 분할: "
            f"Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}"
        )

        stats = {"train": 0, "val": 0, "test": 0}
        total_polygons = 0

        for split, files in splits.items():
            if not files:
                continue

            print(f"\n{split.upper()} 세트 처리 중...")
            for rgb_path in tqdm(files, desc=f"{split} 변환"):
                frame_num = int(rgb_path.stem.split("_")[-1])
                seg_path = seg_dir / f"instance_segmentation_{frame_num:04d}.png"
                if not seg_path.exists():
                    continue

                yolo_annotations = self.convert_frame(seg_path)
                if not yolo_annotations:
                    continue

                shutil.copy2(rgb_path, self.output_dir / "images" / split / rgb_path.name)
                label_path = self.output_dir / "labels" / split / f"{rgb_path.stem}.txt"
                with open(label_path, "w") as f:
                    f.write("\n".join(yolo_annotations))

                stats[split] += 1
                total_polygons += len(yolo_annotations)

        self.create_yaml_config()
        print("\n[변환 완료]")
        print(f"✓ 총 {sum(stats.values())}개 이미지 변환")
        print(f"✓ 총 {total_polygons}개 polygon 어노테이션")
        for split, count in stats.items():
            print(f"  - {split}: {count}개")

        return stats

    def create_yaml_config(self) -> None:
        config = {
            "path": str(self.output_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": self.class_names,
            "nc": len(self.class_names),
        }

        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"✓ 설정 파일 생성: {yaml_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replicator 데이터를 YOLO segmentation 형식으로 변환")
    parser.add_argument("--isaac-dir", type=Path, default=DEFAULT_ISAAC_DIR, help="Replicator raw 데이터 경로")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="YOLO 데이터셋 출력 경로")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="train split 비율")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="val split 비율")
    parser.add_argument("--max-frames", type=int, help="앞에서부터 일부 프레임만 변환")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converter = ReplicatorToYOLOSegmentationConverter(
        isaac_dir=args.isaac_dir,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
    )
    converter.convert_dataset(train_ratio=args.train_ratio, val_ratio=args.val_ratio)


if __name__ == "__main__":
    main()
