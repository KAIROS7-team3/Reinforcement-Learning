# Synthetic Data Generation (SDG) — 공구 Top-View Segmentation

Isaac Sim + Omniverse Replicator로 YOLOv11 segmentation 학습용 합성 데이터를 생성한 파이프라인 정리.

---

## 1. 목적

| 문제 | SDG 해결 |
|---|---|
| Roboflow 실사 데이터 ~1,300장 — **희귀 배치**(밀집, 서랍 열림) 부족 | Domain randomization으로 edge case 대량 생성 |
| Segmentation GT는 **polygon/mask 수작업** — 비용·시간 큼 | Replicator가 캡처와 동시에 **pixel-perfect mask** 자동 생성 |
| Sim-to-real gap | 실제 작업대 USD + RSD455 카메라 구도 유지, real 데이터와 merge |

**한 줄 요약:** Omniverse Replicator로 **라벨링 노가다 없이** segmentation GT를 대량 확보하고, Roboflow 실사 데이터와 merge해 YOLO 성능을 올린다.

---

## 2. NVIDIA SDG 워크플로우에서의 위치

NVIDIA Software-in-the-Loop **Synthetic Data Generation** 4단계 중 우리가 사용한 범위:

| Step | NVIDIA | 우리 작업 | 사용 |
|:---:|---|---|:---:|
| 01 | **NuRec** — 실환경 Gaussian-splat 재구성 | — | ✗ |
| 02 | **Isaac Sim** — 씬 조립·로봇·센서·물리 | `with_camera_backup.usd` 로드, 카메라·공구·staging 설정 | ✓ |
| 03 | **Replicator** — variation 정의, 시뮬레이션, annotation 저장 | randomize → capture → BasicWriter | **✓ 핵심** |
| 04 | **Cosmos Transfer** — photoreal 후처리 | — | ✗ |

> 발표 포인트: **Step 02(씬) + Step 03(Replicator)**. Replicator는 렌더러가 아니라 **자동 라벨러 + 데이터 증강 엔진**.

참고: [Omniverse Replicator 문서](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html)

---

## 3. 전체 파이프라인

```
실제 작업대 USD (with_camera_backup.usd)
        │
        ▼  Isaac Sim — Step 02
   씬 로드 · 카메라 · 공구 semantic label · 로봇 숨김
        │
        ▼  Domain Randomization
   공구 배치(spread/tight) · 서랍 열림 · 조명 · RGB 노이즈
        │
        ▼  Replicator — Step 03
   BasicWriter → rgb_*.png + instance_segmentation_*.png + mapping JSON
        │
        ▼  merge_roboflow_synthetic_dataset.py
   Roboflow 실사 + synthetic → YOLO 포맷 (동일 6-class schema)
        │
        ▼  YOLOv11-seg 학습
   yolo11_seg_mixed_4k (Roboflow + synthetic 4,000장)
```

---

## 4. 씬 구성

### 4.1 Real-scene (메인 SDG)

| 항목 | 경로/값 |
|---|---|
| Scene USD | `/home/user/Desktop/with_camera_backup.usd` |
| Camera | `/World/table/Realsense/RSD455/Camera_OmniVision_OV9782_Color` |
| Tools | `/World/Tools/*` (6종) |
| Robot | `/World/e0509` — 캡처 시 숨김 |
| Toolbox | `/World/toolbox_with_handle` — 선택적 숨김 (`--hide-toolbox`) |
| Staging | 테이블 회색 영역 bbox 자동 계산 |

> **주의:** `with_camera.usd`는 카메라 xform 이슈 있음. Replicator/캡처용으로는 **`with_camera_backup.usd`** 사용.

### 4.2 공구 6종

| Prim 이름 | Synthetic label | Roboflow class |
|---|---|---|
| Allen_Key_Tool_Assembly | allen_key_tool_assembly | multi_tool |
| Husky_Socket_Wrench | husky_socket_wrench | ratchet_wrench |
| Screw_Driver | screw_driver | screwdriver |
| socket | socket | socket_19mm |
| Spanner_16mm | spanner_16mm | spanner_16mm |
| Paper_Cutter | paper_cutter | utility_knife |

### 4.3 Simple scene (발표/데모용)

아이보리 테이블 + 공구 6개만 — `generate_tool_replicator_dataset.py` / `generate_desk6_tools_dataset.py`

---

## 5. Domain Randomization

프레임마다 아래 항목을 랜덤화 (`generate_realscene_replicator_dataset.py`):

| 카테고리 | 내용 | 파라미터 예시 |
|---|---|---|
| **Content** | visible 공구 수, 배치 위치·yaw | `--min-tools 2 --max-tools 6` |
| **Content** | spread vs tight cluster | `--tight-cluster-prob 0.7` |
| **Content** | 서랍 열림/닫힘 | `--drawer-open-prob 0.35` |
| **Appearance** | dome light intensity/color | 런타임 random |
| **Appearance** | exposure, RGB noise/brightness | post-capture noise |

### Tight cluster 배치

- 첫 공구 랜덤 → 이후 공구를 기존 bbox 옆에 **3~12mm 간격**으로 붙임
- mesh bbox 기준 겹침 방지
- 실사에서 어려운 **공구 밀집/접촉** 패턴 의도적 생성

---

## 6. Replicator 출력 (프레임 1장당)

```
rgb_0000.png
instance_segmentation_0000.png
instance_segmentation_semantics_mapping_0000.json
bounding_box_2d_tight_0000.json
semantic_segmentation_0000.png
camera_params_0000.json
...
```

- **RGB** — 학습 이미지
- **instance_segmentation** — 픽셀별 instance ID (mask GT)
- **semantics_mapping** — instance ID ↔ class 이름
- 사람이 polygon을 그릴 필요 없음 → YOLO polygon은 `convert_replicator_to_yolo_seg.py` / merge 스크립트가 contour 추출

---

## 7. 생성 데이터 규모

| Run | 출력 폴더 | 장수 | 비고 |
|---|---|---:|---|
| 1차 | `YOLO/replicator_output/topview_realscene` | 1,000 | spread 위주 |
| 2차 | `YOLO/replicator_output/topview_realscene_3k_tight70` | 3,000 | tight 70% |
| **합계** | — | **4,000** | merge에 사용 |

### Merge 결과 (`yolo_seg_dataset_mixed_4k`)

| Split | 구성 | 대략 장수 |
|---|---|---:|
| train | Roboflow train + synthetic 90% | ~4,896 |
| val | Roboflow valid + synthetic 10% | ~685 |
| test | Roboflow test only (real) | 93 |

---

## 8. 스크립트 목록

| 스크립트 | 역할 |
|---|---|
| `YOLO/generate_realscene_replicator_dataset.py` | **메인 SDG** — real-scene Replicator 캡처 |
| `YOLO/generate_tool_replicator_dataset.py` | simple 아이보리 테이블 SDG |
| `YOLO/generate_desk6_tools_dataset.py` | 책상+공구6 전용 wrapper |
| `YOLO/merge_roboflow_synthetic_dataset.py` | Roboflow + synthetic merge (다중 raw dir 지원) |
| `YOLO/convert_replicator_to_yolo_seg.py` | Replicator raw → YOLO seg 변환 |
| `YOLO/preview_desk6_tools_scene.py` | 랜덤화 없이 GUI 프리뷰 |
| `YOLO/place_desk6_tools_in_usd.py` | USD에 공구 6개 고정 배치 후 저장 |
| `YOLO/split_yolo_results_plots.py` | `results.png` → 개별 그래프 분리 |

---

## 9. 실행 명령

### 9.1 환경

```bash
cd /home/user/Reinforcement-Learning
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
```

GUI Isaac Sim이 켜져 있으면 **종료 후** headless 실행 (GPU/Replicator graph 충돌 방지).

### 9.2 Synthetic 3,000장 생성 (tight 70%)

```bash
python YOLO/generate_realscene_replicator_dataset.py \
  --headless \
  --num-frames 3000 \
  --resolution 2048 \
  --scene-usd /home/user/Desktop/with_camera_backup.usd \
  --output-dir /home/user/Reinforcement-Learning/YOLO/replicator_output/topview_realscene_3k_tight70 \
  --tight-cluster-prob 0.7 \
  --min-tools 2 \
  --max-tools 6 \
  --seed 42
```

### 9.3 Roboflow + synthetic 1k + 3k merge

```bash
python /home/user/Reinforcement-Learning/YOLO/merge_roboflow_synthetic_dataset.py \
  --roboflow-dir /home/user/Downloads/home/iys/Final_project/datasets/tools/top_view_seg \
  --synthetic-raw-dir \
    /home/user/Reinforcement-Learning/YOLO/replicator_output/topview_realscene \
    /home/user/Reinforcement-Learning/YOLO/replicator_output/topview_realscene_3k_tight70 \
  --output-dir /home/user/Reinforcement-Learning/YOLO/yolo_seg_dataset_mixed_4k

rm -f /home/user/Reinforcement-Learning/YOLO/yolo_seg_dataset_mixed_4k/labels/*.cache
```

### 9.4 YOLO 학습 (4k mixed, fresh train 권장)

```bash
yolo segment train \
  model=yolo11n-seg.pt \
  data=/home/user/Reinforcement-Learning/YOLO/yolo_seg_dataset_mixed_4k/dataset.yaml \
  epochs=50 \
  batch=4 \
  imgsz=1024 \
  project=/home/user/Reinforcement-Learning/runs/segment \
  name=yolo11_seg_mixed_4k
```

### 9.5 Real-only test 평가

```bash
yolo segment val \
  model=/home/user/Reinforcement-Learning/runs/segment/yolo11_seg_mixed_4k/weights/best.pt \
  data=/home/user/Reinforcement-Learning/YOLO/yolo_seg_dataset_mixed_4k/dataset.yaml \
  split=test
```

### 9.6 씬 프리뷰 (랜덤화 없음)

```bash
python YOLO/preview_desk6_tools_scene.py \
  --scene-usd /home/user/Desktop/with_camera_backup.usd
```

---

## 10. 학습 결과 (참고)

Run: `runs/segment/yolo11_seg_mixed_4k` (50 epoch)

| Metric | Box | Mask |
|---|---:|---:|
| mAP50 | 0.988 | 0.987 |
| mAP50-95 | 0.928 | 0.768 |

> val 기준. **최종 성능은 real-only test split으로 확인**할 것.

개별 학습 곡선:

```bash
python YOLO/split_yolo_results_plots.py \
  /home/user/Reinforcement-Learning/runs/segment/yolo11_seg_mixed_4k
```

---

## 11. 발표용 핵심 메시지

### Before / After

| | 수작업 (Roboflow) | Replicator SDG |
|---|---|---|
| 라벨링 | polygon/mask **수작업** | 캡처 = **자동 GT** |
| 희귀 케이스 | 촬영+라벨링 비용 큼 | 파라미터만 바꿔 **수천 장** |
| 역할 | 실사 **appearance** | 합성 **content + 무료 라벨** |

### 30초 피치

> 공구 segmentation은 mask 라벨링이 병목이다. Omniverse Isaac Sim에 실제 작업대 USD를 올리고, Replicator로 domain-randomized 합성 데이터 4,000장을 생성했다. 3D 씬에 semantic label만 붙이면 instance mask가 자동 생성되므로, 공구 밀집·서랍 열림 같은 희귀 패턴을 **라벨링 비용 0**으로 대량 확보할 수 있다. 이를 Roboflow 실사 데이터와 merge해 YOLOv11 segmentation을 학습했다.

---

## 12. 주요 경로 요약

```text
/home/user/Desktop/with_camera_backup.usd          # SDG 씬
/home/user/Reinforcement-Learning/YOLO/            # 스크립트
/home/user/Reinforcement-Learning/YOLO/replicator_output/   # Replicator raw
/home/user/Reinforcement-Learning/YOLO/yolo_seg_dataset_mixed_4k/  # merge 결과
/home/user/Reinforcement-Learning/runs/segment/yolo11_seg_mixed_4k/  # 학습 run
```

---

## 13. 알려진 이슈

- GUI Isaac Sim과 headless SDG **동시 실행 금지**
- `with_camera.usd` 카메라 xform — `with_camera_backup.usd` 사용
- 서랍 joint path / unitsResolve scale — `drawer_joint` 오프셋 보정 로직 포함
- merge 시 손상 PNG CRC error → merge 스크립트가 자동 skip
- synthetic-only fine-tune은 real 성능 저하 가능 → **항상 merge 후 학습**

---

*Last updated: 2026-06-24*
