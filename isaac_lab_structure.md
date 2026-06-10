# Isaac Lab 프로젝트 폴더 구조 — tool_transfer_bot

## Isaac Lab Task Workflow 선택

Isaac Lab은 두 가지 환경 설계 방식을 제공한다.

| 방식 | 기반 클래스 | 특징 |
|---|---|---|
| **Manager-Based** | `ManagerBasedRLEnv` / `ManagerBasedRLEnvCfg` | Observation·Reward·DR을 별도 Manager로 분리. 컴포넌트 교체 용이 |
| **Direct** | `DirectRLEnv` / `DirectRLEnvCfg` | 단일 클래스. PyTorch JIT/Warp 최적화 가능 |

**이 프로젝트는 Manager-Based 방식을 채택한다.**

채택 이유:

**1. 4개 Task 공통 베이스 상속**
`base_env_cfg.py` 하나로 SceneCfg / ActionsCfg / RewardsCfg / EventCfg를 공통 정의하고,
각 Task는 필요한 부분만 오버라이드한다. 코드 중복 없이 4개 Task를 관리할 수 있다.

**2. Teacher/Student ObservationsCfg 교체**
Teacher/Student cfg 분리 구조가 ObservationManager 교체와 정확히 대응된다.
`teacher_env_cfg.py`와 `student_env_cfg.py`는 ObservationsCfg만 오버라이드하면 된다.

**3. RewTerm 단위 reward logging**
각 reward 항목이 독립적으로 logging되어 Isaac Lab이 자동으로 항목별 기여도를 출력한다.
reward 수렴 실패 시 어떤 term이 문제인지 즉시 파악 가능하다.

**4. EventCfg의 mode 지원 (startup / reset / interval)**
DR 항목별로 적용 시점을 명시적으로 구분할 수 있다.
Teacher DR과 Student DR을 EventCfg 상속으로 깔끔하게 분리한다.

**5. Isaac Lab 공식 예제 호환**
cabinet, lift, reach 등 Isaac Lab 공식 예제가 모두 Manager-Based다.
참고 코드가 풍부하고 커뮤니티 지원 및 버그 픽스가 빠르다.

```
base_env_cfg.py (ManagerBasedRLEnvCfg 상속)
  ├── SceneCfg        — 로봇, 공구, 서랍, Staging Area
  ├── ActionsCfg      — delta_pos / delta_quat / gripper
  ├── RewardsCfg      — RewTerm 객체로 보상 항목 정의
  ├── EventCfg        — Domain Randomization (물리 DR)
  └── ObservationsCfg — (Teacher/Student가 각각 오버라이드)

teacher_env_cfg.py
  └── ObservationsCfg → GT state (joint, ee_pos/quat, gripper_joint_pos, object_pos/quat, target_tool_id)

student_env_cfg.py
  └── ObservationsCfg → Dual-Cam Frame Stacking + proprioception + target_tool_id
                        proprioception = joint_angles(6) + ee_pos/quat + gripper_joint_pos(1) + target_tool_id(6)
                        + 비전 DR (EventCfg 확장)
```

---

## 전체 디렉토리

```
tool_transfer_bot/                                 # Git Repository 루트
├── .vscode/
│   └── settings.json                             # Isaac Sim / Lab PyTorch API 경로 설정
│
├── config/
│   └── extension.toml                            # Omniverse Extension 메타데이터 및 의존성
│
├── data/                                         # 학습 산출물 (git 제외)
│   ├── demos/                                    # Stage 2a: (cam_obs, teacher_action) 버퍼
│   │   └── <task>/<datetime>/buffer.hdf5
│   └── checkpoints/                              # Teacher / Student checkpoint
│       └── <task>/<datetime>/model_<step>.pt
│
├── docs/
│   └── README.md
│
├── scripts/                                      # 터미널 실행용 엔트리포인트
│   ├── train.py                                  # Stage 1: Teacher PPO 훈련 (rsl_rl)
│   ├── collect_demos.py                          # Stage 2a: Teacher β=1.0 오프라인 데이터 수집
│   ├── pretrain_student.py                       # Stage 2b: 오프라인 BC 사전학습
│   ├── train_dagger.py                           # Stage 2c: DAgger β-Scheduling 온라인 학습
│   ├── eval.py                                   # 성공률 정량 평가 (play.py와 분리)
│   └── play.py                                   # 훈련된 정책 + BT 시각 검증
│
├── logs/                                         # Isaac Lab 자동 생성 (git 제외)
│   └── <task>/<datetime>/                        # TensorBoard + rsl_rl checkpoint
│
├── setup.py                                      # pip install -e . 배포 설정
│
└── source/
    └── tool_transfer_bot/                        # 핵심 Python 패키지
        ├── __init__.py                           # 하위 Task 환경 등록(gym.register) 트리거
        │
        ├── assets/                               # USD 모델 및 기구학 파라미터
        │   ├── __init__.py
        │   ├── doosan_e0509.py                   # 관절 한계, Stiffness, Damping 설정
        │   ├── environments.py                   # 서랍장 + 6종 공구 + Staging Area USD 경로 설정
        │   └── usd/                              # 실제 USD 모델 파일
        │       ├── toolbox.usd                   # 서랍장 (Prismatic joint 포함)
        │       ├── staging_area.usd              # Staging Area
        │       └── tools/                        # 6종 공구 개별 USD
        │           ├── screwdriver.usd
        │           ├── utility_knife.usd
        │           ├── ratchet_wrench.usd
        │           ├── multi_tool.usd
        │           ├── spanner_16mm.usd
        │           └── socket_19mm.usd
        │
        ├── tasks/                                # 4개 Sub-task 환경 명세
        │   ├── __init__.py
        │   ├── base_env_cfg.py                   # ManagerBasedRLEnvCfg 상속
        │   │                                     #   SceneCfg (로봇 + 공구 + 서랍 + Staging
        │   │                                     #            + FrameTransformerCfg ← 핸들 추적)
        │   │                                     #   ActionsCfg / RewardsCfg / EventCfg
        │   │                                     #   TerminationsCfg ← time_out + task_success
        │   │                                     #   CurriculumCfg   ← DR 단계적 확대 (선택)
        │   │                                     #   target_tool_id (6,) One-hot
        │   ├── mdp/                              # 공통 MDP 함수
        │   │   ├── __init__.py
        │   │   ├── drawer_rewards.py             # approach_ee_handle, align_ee_handle,
        │   │   │                                 # open_drawer_bonus, multi_stage_open_drawer
        │   │   ├── manipulation_rewards.py       # object_goal_dist, sdf_alignment,
        │   │   │                                 # success_bonus, force_penalty
        │   │   ├── terminations.py               # task_success (Task별 성공 조건)
        │   │   │                                 # time_out
        │   │   └── observations.py               # rel_ee_object_distance, ee_pos, ee_quat 등
        │   │
        │   ├── open_drawer/                      # [Task 1] 공구함 열기
        │   │   ├── __init__.py                   # Isaac-OpenDrawer-Teacher-v0
        │   │   │                                 # Isaac-OpenDrawer-Student-v0 등록
        │   │   ├── teacher_env_cfg.py            # ObservationsCfg → GT state + 물리 DR
        │   │   └── student_env_cfg.py            # ObservationsCfg → Dual-Cam + Frame Stacking
        │   │                                     #   EventCfg 확장 → 비전 DR
        │   │
        │   ├── fetch_tool/                       # [Task 2] 공구함에서 공구 꺼내기 (🔴 최고 난이도)
        │   │   ├── __init__.py                   # Isaac-FetchTool-Teacher-v0
        │   │   │                                 # Isaac-FetchTool-Student-v0 등록
        │   │   ├── teacher_env_cfg.py            # ObservationsCfg → tool_id One-hot + GT state + 물리 DR
        │   │   └── student_env_cfg.py            # ObservationsCfg → Frame Stacking + Dual-Cam
        │   │                                     #   EventCfg 확장 → 비전 DR
        │   │
        │   ├── return_tool/                      # [Task 3] Staging → 공구함 위에 내려놓기
        │   │   ├── __init__.py                   # Isaac-ReturnTool-Teacher-v0
        │   │   │                                 # Isaac-ReturnTool-Student-v0 등록
        │   │   ├── teacher_env_cfg.py            # ObservationsCfg → Staging → INNER 위치 GT state + 물리 DR
        │   │   └── student_env_cfg.py            # ObservationsCfg → Dual-Cam + Frame Stacking
        │   │                                     #   EventCfg 확장 → 비전 DR
        │   │
        │   └── close_drawer/                     # [Task 4] 공구함 닫기
        │       ├── __init__.py                   # Isaac-CloseDrawer-Teacher-v0
        │       │                                 # Isaac-CloseDrawer-Student-v0 등록
        │       ├── teacher_env_cfg.py            # ObservationsCfg → GT state + 물리 DR
        │       └── student_env_cfg.py            # ObservationsCfg → Dual-Cam + Frame Stacking
        │                                         #   EventCfg 확장 → 비전 DR
        │
        └── agents/                               # RL / 모방학습 네트워크 아키텍처
            ├── __init__.py
            ├── ppo_cfg/                          # Task별 PPO 하이퍼파라미터 분리
            │   ├── open_drawer_ppo_cfg.py
            │   ├── fetch_tool_ppo_cfg.py         # Task 2: lr / entropy 별도 튜닝 필요
            │   ├── return_tool_ppo_cfg.py
            │   └── close_drawer_ppo_cfg.py
            └── student_networks.py               # Stage 2: Frame Stacking CNN 인코더 + MLP
```

---

## 파일별 역할 상세

### scripts/

| 파일 | Stage | 역할 |
|---|---|---|
| `train.py` | Stage 1 | Teacher PPO 훈련. 4096 병렬 env, 카메라 OFF. 성공률 ≥ 80% 시 checkpoint 저장 |
| `collect_demos.py` | Stage 2a | Teacher β=1.0으로 환경 완전 제어. Student 개입 없음. (cam_obs, teacher_action) 오프라인 버퍼 수집 |
| `pretrain_student.py` | Stage 2b | 오프라인 버퍼로 Student BC loss 수렴까지 사전학습. 랜덤 초기화 탈출 목적 |
| `train_dagger.py` | Stage 2c | β=0.9 시작 → β=0.0까지 점진적 감소. Student가 실제로 환경 제어하며 Teacher 레이블 누적 수집 |
| `eval.py` | 평가 | Task별 / 전체 시퀀스(Fetch/Return) 성공률 정량 측정. play.py와 분리 |
| `play.py` | 검증 | 훈련된 정책 시각화 + BT 연동 시각 검증 |

---

### tasks/base_env_cfg.py

`ManagerBasedRLEnvCfg`를 상속받는 공통 베이스 클래스. 모든 Task 환경이 이를 상속한다.

```python
@configclass
class BaseEnvCfg(ManagerBasedRLEnvCfg):
    scene:        SceneCfg        # 로봇(e0509), 공구, 서랍, Staging Area
                                  # + drawer_frame: FrameTransformerCfg (핸들 위치 추적)
    actions:      ActionsCfg      # delta_pos / delta_quat / gripper (DiffIK DLS)
    rewards:      RewardsCfg      # RewTerm 객체로 보상 항목 정의
    terminations: TerminationsCfg # time_out + task_success (Task별 성공 조건)
    events:       EventCfg        # 물리 DR (마찰력, 질량, 초기 위치 등)
    observations: ObservationsCfg # Teacher/Student가 각각 오버라이드
    curriculum:   CurriculumCfg   # DR 단계적 확대 (선택 — Task 2 수렴 실패 시 활성화)
    # Goal-Conditioning
    # target_tool_id (6,) One-hot → ObservationsCfg 내 공통 포함
```

**RewardsCfg 패턴 (RewTerm 사용):**
```python
@configclass
class RewardsCfg:
    # Task 2, 3 (pick & place)
    object_goal_dist = RewTerm(func=mdp.object_goal_distance, weight=-1.0)
    sdf_alignment    = RewTerm(func=mdp.sdf_query_distance,   weight=-1.0)
    success_bonus    = RewTerm(func=mdp.task_success,         weight=10.0)
    force_penalty    = RewTerm(func=mdp.contact_force,        weight=-0.01)
```

---

### tasks/ — Teacher / Student ObservationsCfg 오버라이드 구조

| Task | teacher_env_cfg | student_env_cfg |
|---|---|---|
| Task 1 (OpenDrawer) | ObservationsCfg → GT state | ObservationsCfg → Dual-Cam + Frame Stacking + 비전 DR |
| Task 2 (FetchTool) 🔴 | ObservationsCfg → GT state + tool_id | ObservationsCfg → Dual-Cam + Frame Stacking + 비전 DR |
| Task 3 (ReturnTool) | ObservationsCfg → GT state + tool_id | ObservationsCfg → Dual-Cam + Frame Stacking + 비전 DR |
| Task 4 (CloseDrawer) | ObservationsCfg → GT state | ObservationsCfg → Dual-Cam + Frame Stacking + 비전 DR |

**Teacher cfg 공통 물리 DR (EventCfg) — reset마다:**
```
robot_init_joint_pos    ± 0.05 rad

공구 초기 위치 (object_pos):
  Task 2 (FetchTool)    ± 15mm
  Task 3 (ReturnTool)   ± 10mm
  Task 1, 4             ± 5mm

공구 초기 자세 (object_quat):
  Task 2 (FetchTool)    ± 20°
  Task 3 (ReturnTool)   ± 15°
  Task 1, 4             ± 5°

drawer_init_joint_pos   ± 5~10mm

# 1차 배포 캘리브레이션 오차 모델링
obs_object_pos_noise    ± 5mm Gaussian
obs_target_pos_noise    ± 3mm Gaussian
```

**Teacher cfg 공통 물리 DR (EventCfg) — startup마다:**
```
drawer_friction         ± 20%
object_mass             ± 30%
object_surface_friction ± 30%
joint_torque_noise      ± 3% Gaussian
```

**Student cfg 공통 비전 DR (EventCfg 확장) — reset마다:**
```
lighting_intensity      0.5x ~ 2.0x
lighting_position       반경 1m 내 랜덤
camera_extrinsic_pos    ± 2mm
camera_extrinsic_rot    ± 0.5°
camera_fov              ± 2°
image_gaussian_noise    σ = 0.01~0.03     # D455f 특성 (Gaussian)
image_motion_blur       kernel 3~5        # 속도 비례
object_color            ± 15% HSV
object_reflectance      ± 30%
```

**Student cfg 공통 비전 DR (EventCfg 확장) — startup마다:**
```
workspace_texture       텍스처 풀 샘플링
```

---

### agents/

| 파일 | 역할 |
|---|---|
| `rsl_rl_ppo_cfg.py` | Stage 1 PPO 하이퍼파라미터 (learning rate, entropy coeff, clip range 등) |
| `student_networks.py` | Stage 2 Student 네트워크: Frame Stacking CNN 인코더 (top + wrist) + MLP |

**Student 네트워크 구조:**
```
top_cam   (H×W×(3×k)) → CNN Encoder → latent_top   (128,)
                                             ↓
                                        concat + MLP → action (8,)
                                             ↑
wrist_cam (H×W×(3×k)) → CNN Encoder → latent_wrist (128,)
                                             ↑
proprioception (joint_angles(6) + ee_pos/quat + gripper_joint_pos(1) + target_tool_id(6))

k = 3~4 (Frame Stacking)
```

---

## 실제 배포 전략

### 1차 (기본): Teacher + 캘리브레이션 GT 주입

```
시뮬 학습:  Teacher PPO (GT state, 4096 env)
실제 배포:  캘리브레이션 → GT 좌표 구성 → Teacher policy 직접 입력
```

| 관측값 | 실제 획득 방법 |
|---|---|
| robot_joint_angles (6,) | 조인트 엔코더 |
| ee_pos / ee_quat | FK |
| gripper_joint_pos (1,) | 그리퍼 엔코더 |
| object_pos (3,) | YOLO + D455f depth |
| object_quat (4,) | depth 포인트클라우드 or 탑뷰 회전각 |
| target_pos / target_quat | 사전 측정 고정값 |
| target_tool_id (6,) | BT에서 주입 |

Student 없이 Teacher를 그대로 실제에 배포한다. `scripts/play.py`가 이 경로를 담당한다.

### 2차 (여유 시): Student CNN end-to-end

Stage 2a → 2b → 2c DAgger 파이프라인. 1차 배포 안정 확인 후 시도한다.

---

## 학습 실행 순서

```bash
# Stage 1: Teacher 학습 (Task별로 각각 실행)
python scripts/train.py --task Isaac-OpenDrawer-v0
python scripts/train.py --task Isaac-FetchTool-v0
python scripts/train.py --task Isaac-ReturnTool-v0
python scripts/train.py --task Isaac-CloseDrawer-v0

# Stage 2a: 오프라인 데모 수집
python scripts/collect_demos.py --task Isaac-FetchTool-v0 --teacher_ckpt <path>

# Stage 2b: Student BC 사전학습
python scripts/pretrain_student.py --task Isaac-FetchTool-v0 --demo_buffer <path>

# Stage 2c: DAgger 온라인 학습
python scripts/train_dagger.py --task Isaac-FetchTool-v0 --student_ckpt <path>

# 평가
python scripts/eval.py --task Isaac-FetchTool-v0 --ckpt <path>

# 시각 검증
python scripts/play.py --task Isaac-FetchTool-v0 --ckpt <path>
```

---

## Task 학습 권장 순서

```
1단계 (병렬 시작): Task 1, 4 + Task 2 (가장 오래 걸리므로 최우선)
2단계:             Task 3
3단계:             Fetch (Task1→2→4) / Return (Task1→3→4) 전체 체이닝 테스트
```
