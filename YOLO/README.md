# YOLO Synthetic Tool Dataset

Isaac Sim Replicator로 책상 위 공구 합성 데이터를 생성하고, 이후 YOLO 학습 데이터로 변환하기 위한 작업 폴더입니다.

## 1단계: Raw Replicator 데이터 생성

Isaac Sim이 설치된 환경에서 실행합니다.

```bash
cd /home/user/Reinforcement-Learning
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
python YOLO/generate_tool_replicator_dataset.py --num-frames 10
```

GUI 없이 생성하려면:

```bash
python YOLO/generate_tool_replicator_dataset.py --headless --num-frames 100
```

기존 테스트 출력과 섞이지 않게 다시 생성하려면:

```bash
rm -rf YOLO/replicator_output/tool_table_raw
python YOLO/generate_tool_replicator_dataset.py --num-frames 10
```

기본 환경은 스크립트 안에서 생성하는 `0.50m x 0.75m` 아이보리 무광 판자 테이블입니다. 테이블은 `2 x 3 = 6`개 그리드로 나뉘며, 각 그리드는 `0.25m x 0.25m`입니다. 6개 공구는 그리드당 하나씩 배치됩니다. `assets/tools`의 원본 USDZ는 meter 단위로 변환되어 있고, world bounding box 중심이 root 원점에 오며 바닥면이 `z=0`에 오도록 정규화되어 있습니다. class semantic label도 각 USDZ 내부의 reference geometry prim에만 들어 있어 instance segmentation에서 공구 1개가 instance 1개로 처리됩니다. 이후 bbox 기준으로 테이블 상단에 붙고 X/Y 경계도 테이블 안으로 들어오도록 자동 보정됩니다. 배치 후 공구 bbox끼리 겹치면 최대 80회까지 다시 샘플링합니다.

프레임마다 공구 배치/회전, 조명 밝기/위치/각도, 카메라 높이/위치/기울기, RGB 카메라 노이즈가 랜덤화됩니다. 공구는 Z축 yaw 회전에 더해 낮은 확률로 180도 flip되어 뒷면도 일부 포함됩니다. 노이즈는 저장된 `rgb_*.png`에만 적용되고 segmentation/bbox annotation은 그대로 유지됩니다.

프레임 간 잔상을 줄이기 위해 캡처는 `rt_subframes=8`로 수행하며, 가능한 temporal/denoising/motion blur 계열 렌더 설정은 비활성화합니다.

더 높은 해상도로 생성하려면:

```bash
python YOLO/generate_tool_replicator_dataset.py --num-frames 100 --resolution 1024
```

실제 top-view D455f/RSD455처럼 넓은 시야각으로 만들고, 기존 raw 데이터를 보존하려면 새 출력 폴더를 지정합니다. 기본 카메라 높이는 실제 top-view에 맞춰 `0.95m`를 사용합니다. `rsd455` 프로파일은 Isaac Sim 5.1 RSD455 asset의 RGB 카메라(`Camera_OmniVision_OV9782_Color`) 속성을 사용합니다.

```bash
python YOLO/generate_tool_replicator_dataset.py \
  --num-frames 1000 \
  --resolution 2048 \
  --camera-model rsd455 \
  --camera-height 0.95 \
  --camera-height-jitter 0.05 \
  --output-dir YOLO/replicator_output/tool_table_raw_topview_rsd455
```

## 1b단계: Real-scene USD (`with_camera_backup.usd`) 기반 synthetic

실제 workbench와 비슷하게 구성한 USD를 그대로 로드해 Replicator 데이터를 생성합니다.

```bash
python YOLO/generate_realscene_replicator_dataset.py --num-frames 10
python YOLO/generate_realscene_replicator_dataset.py \
  --headless --num-frames 1000 --resolution 1024 \
  --min-tools 1 --max-tools 4 \
  --output-dir YOLO/replicator_output/realscene_topview_rsd455
```

- Scene: `/home/user/Desktop/with_camera_backup.usd` (카메라 시야가 올바른 backup)
- `with_camera.usda`는 카메라 xform이 잘못 저장되어 있으므로 사용하지 않음
- Camera: `/World/table/Realsense/RSD455/Camera_OmniVision_OV9782_Color`
- Tools: `/World/Tools/*` (프레임당 1~6개 랜덤 visible)
- Robot `/World/e0509`는 캡처 시 숨김 (기본)
- 서랍 열림/닫힘 랜덤 (`--drawer-open-prob`)
- staging bounds: `--staging-min-x` 등으로 조정

merge 시 `--synthetic-raw-dir`를 `realscene_topview_rsd455`로 지정하면 됩니다.

**주의 (segfault 방지):**
- GUI Isaac Sim이 켜져 있으면 종료 후 headless 실행 (GPU/Replicator graph 충돌)
- `with_camera_backup.usd`에 저장된 `/Render/.../Replicator` graph가 있으면 스크립트가 자동 제거
- 여전히 crash 나면 `--resolution 640`으로 먼저 테스트

`--output-dir`는 프로젝트 루트(`/home/user/Reinforcement-Learning`) 기준 상대 경로로 해석되며, 스크립트 내부에서 절대 경로로 변환해 저장합니다. Replicator에 상대 경로만 넘기면 `~/omni.replicator_out/` 아래에 저장될 수 있습니다.

기본 입력 공구 에셋:

```text
assets/tools/
├── Allen Key Tool Assembly.usdz
├── Husky Socket Wrench.usdz
├── Paper Cutter.usdz
├── Screw Driver.usdz
├── socket.usdz
└── Spanner 16mm.usdz
```

정규화 전 원본은 같은 폴더에 `*.orig.usdz` 백업으로 보관되어 있습니다.

기본 출력 위치:

```text
YOLO/replicator_output/tool_table_raw/
```

생성되는 주요 데이터:

```text
rgb_0000.png
instance_segmentation_0000.png
semantic_segmentation_0000.png
bounding_box_2d_tight_0000.npy
camera_params_0000.json
occlusion_0000.npy
...
```

## 클래스

```text
0: screw_driver
1: paper_cutter
2: husky_socket_wrench
3: allen_key_tool_assembly
4: spanner_16mm
5: socket
```

## 다음 단계

`YOLO/convert_replicator_to_yolo_seg.py`를 사용해서 `YOLO/replicator_output/tool_table_raw`를 YOLO segmentation dataset 구조로 변환합니다.

```bash
python YOLO/convert_replicator_to_yolo_seg.py \
  --isaac-dir /home/user/Reinforcement-Learning/YOLO/replicator_output/tool_table_raw \
  --output-dir /home/user/Reinforcement-Learning/YOLO/yolo_seg_dataset
```

변환 결과는 기본 `70/20/10` 비율로 나뉩니다.

```text
YOLO/yolo_seg_dataset/
├── dataset.yaml
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

## 3단계: Roboflow + Synthetic mixed dataset

Roboflow 실제 top-view 데이터와 synthetic Replicator 데이터를 같은 class schema로 merge합니다.

Roboflow class (배포 기준):

```text
0: multi_tool            <- allen_key_tool_assembly
1: ratchet_wrench        <- husky_socket_wrench
2: screwdriver           <- screw_driver
3: socket_19mm           <- socket
4: spanner_16mm
5: utility_knife          <- paper_cutter
```

merge:

```bash
python /home/user/Reinforcement-Learning/YOLO/merge_roboflow_synthetic_dataset.py \
  --roboflow-dir /home/user/Downloads/home/iys/Final_project/datasets/tools/top_view_seg \
  --synthetic-raw-dir /home/user/Reinforcement-Learning/YOLO/replicator_output/tool_table_raw_topview_rsd455 \
  --output-dir /home/user/Reinforcement-Learning/YOLO/yolo_seg_dataset_mixed
```

학습:

```bash
python /home/user/yolo_v11_lecture/lecture-yolo-segmentation/yolo_seg_training.py \
  --output-dir /home/user/Reinforcement-Learning/YOLO/yolo_seg_dataset_mixed \
  --train-only \
  --epochs 50 \
  --batch-size 8 \
  --imgsz 1024
```

기본 동작:

- Roboflow `train/valid/test`는 그대로 유지
- Synthetic 1000장은 기본 `90% train / 10% val`로 추가
- Roboflow `test`는 real-only 평가용으로 유지
- merge 시 PNG CRC 검사 → 손상 파일 자동 skip (`rgb_0794.png` 등)

학습 중 `libpng error: IDAT: CRC error`가 나면:

```bash
# 손상 PNG 검사 (mixed dataset)
python3 - <<'PY'
from pathlib import Path
from PIL import Image
root = Path("/home/user/Reinforcement-Learning/YOLO/yolo_seg_dataset_mixed/images")
for p in root.rglob("*"):
    if p.suffix.lower() != ".png":
        continue
    try:
        with Image.open(p) as im:
            im.verify(); im.load()
    except Exception as e:
        print("BAD", p, e)
PY

# YOLO label cache 삭제 후 재학습/재개
rm -f /home/user/Reinforcement-Learning/YOLO/yolo_seg_dataset_mixed/labels/*.cache
```

epoch 17에서 끊긴 경우 resume:

```bash
yolo segment train resume \
  model=/home/user/Reinforcement-Learning/runs/segment/runs/segment/yolo11_seg_isaac_20260618_205025/weights/last.pt
```
