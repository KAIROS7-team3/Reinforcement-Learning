# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

공구 전달 로봇팔 RL 프로젝트 (Isaac Lab, Track B).  
Long-horizon task(공구 가져오기/반납하기)를 4개의 sub-task로 분해하여 각각 RL 정책을 학습한다.  
BT(Behavior Tree)가 팔 이동과 시퀀스 전환을 담당하고, RL 정책이 실제 조작을 담당한다.  
로봇: Doosan e0509 (6축). 카메라: RealSense D455f (top-view + wrist-view).

## Training Commands

```bash
# Stage 1: Teacher PPO 학습 (Task별 각각 실행, 4096 병렬 env, 카메라 OFF)
python scripts/train.py --task Isaac-OpenDrawer-v0
python scripts/train.py --task Isaac-FetchTool-v0
python scripts/train.py --task Isaac-ReturnTool-v0
python scripts/train.py --task Isaac-CloseDrawer-v0

# Stage 2a: Teacher 오프라인 데모 수집 (Student 개입 없음, β=1.0)
python scripts/collect_demos.py --task Isaac-FetchTool-v0 --teacher_ckpt <path>

# Stage 2b: Student BC 사전학습 (오프라인 버퍼 기반)
python scripts/pretrain_student.py --task Isaac-FetchTool-v0 --demo_buffer <path>

# Stage 2c: DAgger 온라인 학습 (β=0.9 → β=0.0)
python scripts/train_dagger.py --task Isaac-FetchTool-v0 --student_ckpt <path>

# 평가 및 시각 검증
python scripts/eval.py --task Isaac-FetchTool-v0 --ckpt <path>
python scripts/play.py --task Isaac-FetchTool-v0 --ckpt <path>
```

## Architecture

### Sub-task 정의

| Task | 이름 | 난이도 |
|---|---|---|
| Task 1 | OpenDrawer — 공구함 열기 | 🟢 쉬움 |
| Task 2 | FetchTool — 공구함 → Staging Area pick & place | 🔴 어려움 |
| Task 3 | ReturnTool — Staging Area → 공구함 pick & place | 🟡 보통 |
| Task 4 | CloseDrawer — 공구함 닫기 | 🟢 쉬움 |

Task 1↔4, Task 2↔3은 정책 공유 불가 (시작 상태 및 공간 특성이 다름).

### Teacher-Student 순차 Distillation

**Stage 1 (Teacher)**: GT state만으로 4096 병렬 env에서 PPO 학습. 성공률 ≥ 80% 달성 후 checkpoint 저장 → frozen.

**Stage 2 (Student)**: 카메라 이미지 입력. 3단계로 분리:
- **2a**: Teacher β=1.0으로 (cam_obs, teacher_action) 오프라인 버퍼 수집
- **2b**: 오프라인 BC 사전학습 (`MSE(student(cam_obs), teacher_action)`)
- **2c**: DAgger β-Scheduling (β=0.9 → 0.0, 30+ iterations)

> Teacher 성공률 80% 미만이면 Stage 2 진입 금지.

### Manager-Based Env 구조

`ManagerBasedRLEnvCfg`를 채택. `base_env_cfg.py` 하나로 공통 정의, Task/Teacher/Student별 필요한 부분만 오버라이드.

```
base_env_cfg.py (ManagerBasedRLEnvCfg)
  ├── SceneCfg        — 로봇, 공구, 서랍, Staging Area + FrameTransformerCfg (핸들 추적)
  ├── ActionsCfg      — delta_pos(3) / delta_quat(4) / gripper(1), DiffIK DLS
  ├── RewardsCfg      — RewTerm 객체
  ├── EventCfg        — 물리 DR (Teacher) / 비전 DR 추가 (Student)
  └── ObservationsCfg — Teacher: GT state / Student: Dual-Cam + Frame Stacking

teacher_env_cfg.py → ObservationsCfg: GT state
student_env_cfg.py → ObservationsCfg: Dual-Cam Frame Stacking + proprioception
                     EventCfg 확장: 비전 DR
```

### Observation Space

**Teacher (Stage 1)**:
```
robot_joint_angles(6), ee_pos(3), ee_quat(4), gripper_joint_pos(1)
rel_ee_object_distance(3), object_pos(3), object_quat(4)
target_pos(3), target_quat(4), target_tool_id(6)  ← Goal-Conditioning (필수)
robot_joint_velocities(6)  ← critic만 사용 (asymmetric actor-critic)
```

**Student (Stage 2)**:
```
top_cam   (H×W×(3×k)) → CNN Encoder → latent(128,)
wrist_cam (H×W×(3×k)) → CNN Encoder → latent(128,)  ← k=3~4 Frame Stacking
proprioception: joint_angles(6) + ee_pos/quat + gripper_joint_pos(1) + target_tool_id(6)
→ concat + MLP → action(8,)
```

### Action Space

```
delta_pos(3) + delta_quat(4) + gripper(1)
IK: DifferentialInverseKinematicsActionCfg, ik_method="dls", use_relative_mode=True
```

### Reward 설계

**Task 1, 4 (열기/닫기)** — Multi-stage:
- approach_ee_handle (w=2.0), align_ee_handle (w=0.5)
- approach_gripper_handle (w=5.0), grasp_handle (w=0.5)
- open_drawer_bonus (w=7.5), multi_stage_open_drawer (w=1.0, 1cm→20cm→30cm)
- action_rate_l2 (w=-0.01), joint_vel_l2 (w=-0.0001)

**Task 2, 3 (pick & place)**:
- object_goal_dist (w=-1.0), sdf_alignment (w=-1.0, IndustReal 방식)
- success_bonus (w=10.0), force_penalty (w=-0.01), action_rate_l2 (w=-0.01)

### Domain Randomization

**Teacher (물리 DR, reset마다)**: robot_init_joint_pos ±0.05 rad, object_pos ±5~15mm, object_quat ±5~20°, obs_object_pos_noise ±5mm Gaussian  
**Teacher (startup마다)**: drawer_friction ±20%, object_mass ±30%, object_surface_friction ±30%  
**Student (비전 DR, reset마다)**: lighting ×0.5~2.0, camera_extrinsic ±2mm/±0.5°, image_gaussian_noise σ=0.01~0.03, object_color ±15% HSV

## Task 학습 권장 순서

```
1단계 (병렬 시작): Task 1, 4 + Task 2 (가장 오래 걸리므로 최우선)
2단계:             Task 3
3단계:             Fetch(Task1→2→4) / Return(Task1→3→4) 전체 체이닝 테스트
```

## Project Structure

```
tool_transfer_bot/
├── scripts/          — 실행 엔트리포인트 (train / collect_demos / pretrain_student / train_dagger / eval / play)
├── source/tool_transfer_bot/
│   ├── assets/       — doosan_e0509.py, environments.py, usd/ (toolbox + 6종 공구)
│   ├── tasks/
│   │   ├── base_env_cfg.py          — 공통 베이스 (모든 Task 상속)
│   │   ├── mdp/                     — drawer_rewards.py, manipulation_rewards.py, terminations.py, observations.py
│   │   └── {open_drawer, fetch_tool, return_tool, close_drawer}/
│   │       ├── teacher_env_cfg.py
│   │       └── student_env_cfg.py
│   └── agents/
│       ├── ppo_cfg/                 — Task별 PPO 하이퍼파라미터
│       └── student_networks.py     — Frame Stacking CNN + MLP
├── data/             — demos/ + checkpoints/ (git 제외)
└── logs/             — TensorBoard + rsl_rl checkpoint (git 제외)
```

## Deployment

**1차 배포 (기본)**: Teacher policy를 그대로 실제에 배포. YOLO+D455f depth로 object_pos 추정, FK로 ee_pos, BT에서 target_tool_id 주입 → Teacher observation space와 동일한 형태로 구성.

**2차 배포 (여유 시)**: Stage 2 DAgger 파이프라인 완료 후 Student CNN end-to-end 배포.
