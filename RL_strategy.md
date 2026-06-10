# RL Strategy — 공구 전달 로봇팔 (Track B)

## 개요

Long-horizon task(공구 가져오기 / 반납하기)를 4개의 sub-task로 분해하여 각각 RL 정책을 학습한다.
BT(Behavior Tree)가 팔 이동과 시퀀스 전환을 담당하고, RL 정책이 실제 조작을 담당한다.

---

## Sub-task 정의

| Task | 이름 | 내용 |
|---|---|---|
| **Task 1** | OpenDrawer | 공구함 열기 |
| **Task 2** | FetchTool | 공구함에서 공구 집어 → Staging Area에 내려놓기 (pick & place) |
| **Task 3** | ReturnTool | Staging Area에서 공구 집어 → 공구함 위에 내려놓기 (pick & place) |
| **Task 4** | CloseDrawer | 공구함 닫기 |

---

## 실행 시퀀스

### 공구 가져오기 (Fetch)

```
BT: Task1 수행 위치로 이동
→ Task1 (공구함 열기)
→ BT: wrist cam이 공구 전부 보이는 위치로 이동
→ Task2 (공구함 → Staging Area pick & place)
→ BT: 공구함 닫을 수 있는 위치로 이동
→ Task4 (공구함 닫기)
→ BT: 홈 복귀
```

### 공구 반납하기 (Return)

```
BT: Task1 수행 위치로 이동
→ Task1 (공구함 열기)
→ BT: Task3 수행 위치로 이동
→ Task3 (Staging Area → 공구함 pick & place)
→ BT: 공구함 닫을 수 있는 위치로 이동
→ Task4 (공구함 닫기)
→ BT: 홈 복귀
```

---

## 역할 분담

| 역할 | 담당 |
|---|---|
| 각 task 수행 위치로 이동 | BT (고정 궤적) |
| Task 1, 2, 3, 4 실제 조작 | RL 정책 |
| 홈 복귀 | BT |

---

## 난이도 분류

| 난이도 | Task | 이유 |
|---|---|---|
| 🟢 쉬움 | Task 1, 4 | 고정 궤적에 가까운 단순 조작, 실물 검증 완료 |
| 🟡 보통 | Task 3 | Staging 파지 + 공구함 위 내려놓기, 비교적 관대한 공간 |
| 🔴 어려움 | Task 2 | 좁은 서랍 내부 파지 + wrist cam 의존, 공구별 형상 다양 |

---

## 학습 전략 — Teacher-Student (순차 Distillation)

### 배경 및 목적

Isaac Sim에서 카메라 2대를 observation으로 쓰면 병렬 env 연산량이 폭발적으로 증가한다.
이를 해결하기 위해 두 단계로 분리한다.

- **Stage 1 (Teacher)**: 카메라 없이 GT state만으로 4096 병렬 env에서 빠르게 정책 학습
- **Stage 2 (Student)**: Teacher가 완전히 수렴한 후, 소수 env에서 카메라 관측으로 Teacher 행동 재현

두 단계는 완전히 순차적이다. Stage 1이 수렴하지 않으면 Stage 2를 시작하지 않는다.

---

### Stage 1 — Teacher 학습

#### Observation Space

```
robot_joint_angles       (6,)   # 관절 각도 (rad) — e0509 6축
robot_joint_velocities   (6,)   # 관절 속도 (critic만 사용 — asymmetric actor-critic)
ee_pos                   (3,)   # end-effector 위치 (m)
ee_quat                  (4,)   # end-effector 자세 (quaternion)
gripper_joint_pos        (1,)   # 그리퍼 개폐 상태
rel_ee_object_distance   (3,)   # EE → 공구/핸들 상대 거리벡터 (FrameTransformer)
object_pos               (3,)   # 공구 위치 GT
object_quat              (4,)   # 공구 자세 GT
target_pos               (3,)   # 목표 위치 GT (staging / 공구함 INNER 위치)
target_quat              (4,)   # 목표 자세 GT
target_tool_id           (6,)   # ★ Goal-Conditioning: 목표 공구 One-hot (공구 6종)
```

> **[치명적 누락 보완] Goal-Conditioning**
> Task 2 / Task 3은 6종 공구 중 특정 공구를 타깃으로 해야 한다.
> `target_tool_id` 없이는 정책이 "지금 어떤 공구를 집어야 하는가"를 알 수 없어
> 여러 공구가 있을 때 랜덤하게 집거나 갈피를 못 잡는 문제가 발생한다.
> Stage 1 / Stage 2 Observation Space 모두에 반드시 포함해야 한다.

#### Action Space

```
delta_pos    (3,)   # end-effector 위치 증분 (m)
delta_quat   (4,)   # end-effector 자세 증분 (quaternion)
gripper      (1,)   # 그리퍼 명령 [0=open, 1=close]
```

> **IK 방식**: `DifferentialInverseKinematicsActionCfg` + `ik_method="dls"` (Damped Least Squares)
> 특이점(singularity) 회피를 위해 DLS를 사용한다. `use_relative_mode=True`로 delta 입력을 받는다.
> Joint space 대비 policy가 운동학을 직접 학습할 필요 없어 수렴 속도가 30~50% 빠르다.

#### Reward 설계

**Task 1, 4 (열기/닫기) — Multi-stage Reward (Isaac Lab cabinet 예제 기반)**

```
# 1단계: 핸들 접근 (inverse-square law)
approach_ee_handle        weight=2.0   # EE ↔ 핸들 거리 감소
align_ee_handle           weight=0.5   # EE 자세 ↔ 핸들 방향 정렬

# 2단계: 핸들 파지
approach_gripper_handle   weight=5.0   # 손가락 ↔ 핸들 거리
align_grasp_around_handle weight=0.125 # 손가락이 핸들을 감싸는 방향 확인
grasp_handle              weight=0.5   # 핸들 근처에서 그리퍼 닫기

# 3단계: 서랍 열기/닫기 (multi-stage threshold)
open_drawer_bonus         weight=7.5   # 서랍 joint pos 비례 보너스
multi_stage_open_drawer   weight=1.0   # 1cm→20cm→30cm 단계별 추가 보너스

# 패널티
action_rate_l2            weight=-0.01
joint_vel_l2              weight=-0.0001
```

> FrameTransformer로 핸들 위치를 실시간 추적한다.
> (`SceneCfg`에 `drawer_frame` 추가, `prim_path`를 공구함 서랍 핸들 USD prim으로 지정)

**Task 2, 3 (pick & place)**

```
object_goal_dist   weight=-1.0   # 공구-목표 거리 감소 (L2)
sdf_alignment      weight=-1.0   # SDF 정렬 보상 ← IndustReal 방식 참고
success_bonus      weight=10.0   # 성공 보너스
force_penalty      weight=-0.01  # 과도한 힘 패널티
action_rate_l2     weight=-0.01
```

#### 학습 루프

```
4096 병렬 env에서 rollout 수집 (카메라 OFF)
→ PPO로 policy / value network 업데이트
→ 성공률 모니터링
→ 성공률 ≥ 80% 안정 유지 시 Stage 1 종료
→ Teacher checkpoint 저장 (이후 frozen)
```

---

### Stage 2 — Student 학습

Stage 2는 랜덤 초기화 Student를 바로 환경에 투입하면 의미 없는 데이터만 쌓이므로
**3단계(2a → 2b → 2c)로 분리**하여 안전하게 수렴시킨다.

---

#### Stage 2a — Teacher 데모 수집 (Student 환경 개입 없음)

Student는 아예 건드리지 않고 **Teacher만 환경을 100% 제어**하면서 데이터를 모은다.

```
Teacher가 β=1.0으로 환경 완전 제어 (Student 개입 없음)
  → 각 timestep에서 (cam_obs, teacher_action) 쌍 수집
  → 오프라인 버퍼에 저장 (수만 ~ 수십만 쌍)
```

시뮬레이션 불안정 위험 없음. Student가 환경에 전혀 개입하지 않기 때문.

---

#### Stage 2b — 오프라인 BC 사전학습

수집한 Teacher 데모로 Student를 순수 지도학습으로 먼저 훈련한다.
이 단계가 끝나면 Student는 랜덤이 아닌 **Teacher를 어느 정도 모방하는 초기 정책**이 된다.

```
for epoch in range(N):
    batch = sample(offline_buffer)
    loss = MSE(student_policy(batch.cam_obs), batch.teacher_action)
    optimizer.step(loss)

→ BC loss 수렴까지 반복 후 Stage 2c 진입
```

#### Student 네트워크 구조

```
top_cam   (H×W×(3×k)) → CNN Encoder → latent_top   (128,)
                                             ↓
                                        concat + MLP → action (8,)  # delta_pos(3)+delta_quat(4)+gripper(1)
                                             ↑
wrist_cam (H×W×(3×k)) → CNN Encoder → latent_wrist (128,)
                                             ↑
proprioception (joint_angles(6) + ee_pos/quat + gripper_joint_pos(1) + target_tool_id(6))
```

> **[네트워크 개선] 시간적 맥락 — Frame Stacking**
> Single frame CNN만으로는 "로봇팔이 이동 중인지, 공구가 흔들리고 있는지" 등
> 동적 상태(velocity/dynamics)를 파악할 수 없다.
> Teacher는 GT state에서 관절 속도를 직접 관측하지만, Student는 이미지만 보므로
> 시간 정보를 별도로 주입해야 Teacher 행동을 완벽히 모방할 수 있다.
>
> **방법 A — Frame Stacking (권장, 구현 단순)**
> 최근 k=3~4 프레임을 채널 방향으로 합쳐 입력
> ```
> top_cam   입력: H × W × (3×k)   # 예: 84×84×9 (k=3)
> wrist_cam 입력: H × W × (3×k)
> ```
>
> **방법 B — CNN + GRU (표현력 높음, 구현 복잡)**
> ```
> CNN → feature (128,) → GRU (hidden 128,) → latent (128,)
> ```

---

#### Stage 2c — DAgger 온라인 학습 (β-Scheduling)

BC 사전학습된 Student로 DAgger를 시작한다. β를 높게 잡아 Teacher가 대부분 제어하다가
iteration이 쌓일수록 Student 비율을 높인다.

> **왜 DAgger인가 — Covariate Shift 문제**
> 순수 BC만 사용하면 Student가 실수로 Teacher가 경험하지 못한 상태에 도달했을 때
> 어떻게 행동할지 모르고 오류가 오류를 부르는 연쇄 실패가 발생한다.
> DAgger는 Student가 실제로 방문하는 상태에서도 Teacher 레이블을 수집하여 이를 해결한다.

```
β = Teacher 제어 비율  (0~1)
action_env = β * action_teacher + (1-β) * action_student

iteration  0:  β = 0.9   (Teacher 90%, Student 10%)
iteration  5:  β = 0.7
iteration 10:  β = 0.5
iteration 20:  β = 0.2
iteration 30+: β = 0.0   (Student 100% — 완전 자율)
```

```
iteration i:
  β 스케줄에 따라 혼합된 action으로 환경 rollout
  각 timestep에서 Teacher action 레이블 생성 (frozen)
  (cam_obs, teacher_action) 쌍을 누적 버퍼에 추가
  전체 버퍼로 BC loss 재학습
  β 감소 (다음 iteration 준비)
```

#### Loss 함수

```
loss = ||π_student(obs_cam) - π_teacher(obs_gt)||²   (MSE)
```

Student는 RL reward 없이 BC loss만으로 업데이트된다.

#### 환경 설정

```
병렬 env 수: 64~256개   (Teacher 4096개 대비 감소 → 렌더링 부담 감소)
카메라: top-view + wrist-view ON
Teacher: frozen (레이블 생성만, 학습 없음)
Student: BC loss로만 업데이트
```

---

#### Stage 2 전체 흐름 요약

```
[Stage 2a]  Teacher β=1.0 → 오프라인 데이터 수집 (Student 개입 없음)
               ↓
[Stage 2b]  오프라인 BC → Student 사전학습 (loss 수렴까지)
               ↓
[Stage 2c]  DAgger β=0.9 → 점진적으로 β 낮춤 → Student 완전 자율 (β=0.0)
```

---

### 도메인 무작위화 (Domain Randomization)

#### Stage 1 — 물리 무작위화

**[reset마다 적용]**

```
robot_init_joint_pos    ± 0.05 rad        # BT 위치 오차 모델링
drawer_init_joint_pos   ± 5~10mm          # 서랍이 완전히 닫히지 않은 상태 모델링

공구 초기 위치 (object_pos):
  Task 2 (FetchTool)    ± 15mm            # 서랍 내 공구가 굴러있을 수 있음
  Task 3 (ReturnTool)   ± 10mm            # Staging에 놓인 공구 변동
  Task 1, 4             ± 5mm             # 고정된 핸들, 변동 작음

공구 초기 자세 (object_quat):
  Task 2 (FetchTool)    ± 20°             # 서랍 안에서 기울어짐
  Task 3 (ReturnTool)   ± 15°             # Staging에서 기울어짐
  Task 1, 4             ± 5°              # 핸들 자세, 거의 고정

# 1차 배포(캘리브레이션 GT 주입) 오차 모델링
obs_object_pos_noise    ± 5mm Gaussian    # YOLO+depth 측정 오차
obs_target_pos_noise    ± 3mm Gaussian    # 사전 측정 좌표 오차
```

**[startup마다 적용]**

```
drawer_friction         ± 20%             # 서랍 마찰력 변동
object_mass             ± 30%             # 공구 종류별 질량 차이 큼
object_surface_friction ± 30%             # 금속 공구 표면 마찰 변동
joint_torque_noise      ± 3% Gaussian     # 모터 imperfection
```

#### Stage 2 — 비전 무작위화 (Sim-to-Real 핵심)

Student는 카메라 이미지를 입력으로 받으므로, 시뮬 이미지와 실제 RealSense D455f 이미지의
외형 차이(도메인 갭)를 줄이기 위해 적용한다.

**[reset마다 적용]**

```
lighting_intensity      0.5x ~ 2.0x       # 조명 밝기 변동
lighting_position       반경 1m 내 랜덤   # 그림자 패턴 변화
camera_extrinsic_pos    ± 2mm             # 카메라 마운트 위치 오차
camera_extrinsic_rot    ± 0.5°            # 카메라 마운트 각도 오차
camera_fov              ± 2°              # D455f FOV 미세 오차
image_gaussian_noise    σ = 0.01~0.03     # D455f 센서 노이즈 (Gaussian 특징적)
image_motion_blur       kernel 3~5        # 로봇 팔 이동 시 블러 (속도 비례)
object_color            ± 15% HSV         # 공구 색상 변동
object_reflectance      ± 30%             # 금속 공구 반사율 변동
```

**[startup마다 적용]**

```
workspace_texture       텍스처 풀 샘플링  # 작업대 표면 텍스처
```

> Stage 1 캘리브레이션 오차 노이즈(obs_object_pos_noise 등)는 1차 배포 robustness에 직결된다.
> Stage 2 비전 DR은 실제 D455f 환경을 최대한 커버하도록 설정한다.

---

### Task별 학습 순서 (권장)

```
1단계: Task 1, 4 (단순, 병렬 학습 가능)
2단계: Task 3
3단계: Task 2 (가장 어려움 — 충분한 학습 시간 확보를 위해 최우선 시작)
4단계: Fetch / Return 전체 체이닝 테스트
```

> Task 2는 가장 오래 걸리므로, 1단계와 병렬로 가장 먼저 시작하는 것을 권장한다.

---

## Isaac Lab 프로젝트 구조

### Isaac Lab Workflow 선택: Manager-Based

Isaac Lab은 Manager-Based / Direct 두 방식을 제공하며, 이 프로젝트는 **Manager-Based**(`ManagerBasedRLEnvCfg`)를 채택한다.

채택 이유:
- **4개 Task 공통 상속**: `base_env_cfg.py` 하나로 공통 정의, Task별 필요한 부분만 오버라이드
- **ObservationManager 교체**: Teacher/Student cfg 분리와 정확히 대응
- **RewTerm 단위 logging**: reward 항목별 기여도 자동 출력 → 수렴 실패 디버깅 용이
- **EventCfg mode 지원**: DR 항목별 startup / reset / interval 적용 시점 명시적 구분
- **공식 예제 호환**: Isaac Lab cabinet / lift 등 공식 예제가 모두 Manager-Based → 참고 코드 풍부

```
base_env_cfg.py  (ManagerBasedRLEnvCfg 상속)
  ├── SceneCfg        — 로봇, 공구, 서랍, Staging Area
  ├── ActionsCfg      — delta_pos / delta_quat / gripper
  ├── RewardsCfg      — RewTerm 객체로 보상 항목 정의
  ├── EventCfg        — 물리 DR (Domain Randomization)
  └── ObservationsCfg — Teacher / Student가 각각 오버라이드

teacher_env_cfg.py → ObservationsCfg: GT state
student_env_cfg.py → ObservationsCfg: Dual-Cam + Frame Stacking
                     EventCfg 확장:   비전 DR (Domain Randomization) 추가
```

---

### 디렉토리

```
tool_transfer_bot/
├── .vscode/
│   └── settings.json                             # Isaac Sim / Lab PyTorch API 경로 설정
├── config/
│   └── extension.toml                            # Omniverse Extension 메타데이터 및 의존성
├── docs/
│   └── README.md
├── scripts/                                      # 터미널 실행용 엔트리포인트
│   ├── train.py                                  # Stage 1: Teacher PPO 훈련 (rsl_rl)
│   ├── collect_demos.py                          # Stage 2a: Teacher β=1.0 오프라인 데이터 수집
│   ├── pretrain_student.py                       # Stage 2b: 오프라인 BC 사전학습
│   ├── train_dagger.py                           # Stage 2c: DAgger β-Scheduling 온라인 학습
│   ├── eval.py                                   # 성공률 정량 평가 (play.py와 분리)
│   └── play.py                                   # 훈련된 정책 + BT 시각 검증
├── setup.py
└── source/
    └── tool_transfer_bot/
        ├── __init__.py                           # 하위 Task 환경 등록(gym.register) 트리거
        ├── assets/
        │   ├── __init__.py
        │   ├── doosan_e0509.py                   # 관절 한계, Stiffness, Damping 설정
        │   ├── environments.py                   # 서랍장 + 6종 공구 + Staging Area USD 경로 설정
        │   └── usd/                              # 실제 USD 모델 파일
        │       ├── toolbox.usd                   # 서랍장 (Prismatic joint 포함)
        │       ├── staging_area.usd
        │       └── tools/                        # 6종 공구 개별 USD
        │           ├── screwdriver.usd
        │           ├── utility_knife.usd
        │           ├── ratchet_wrench.usd
        │           ├── multi_tool.usd
        │           ├── spanner_16mm.usd
        │           └── socket_19mm.usd
        ├── tasks/
        │   ├── __init__.py
        │   ├── base_env_cfg.py                   # ManagerBasedRLEnvCfg 상속
        │   │                                     #   SceneCfg (+ FrameTransformerCfg 핸들 추적)
        │   │                                     #   ActionsCfg / RewardsCfg / EventCfg
        │   │                                     #   TerminationsCfg ← time_out + task_success
        │   │                                     #   CurriculumCfg   ← DR 단계적 확대 (선택)
        │   ├── mdp/                              # 공통 MDP 함수
        │   │   ├── __init__.py
        │   │   ├── drawer_rewards.py             # approach_ee_handle, open_drawer_bonus (Task1,4)
        │   │   ├── manipulation_rewards.py       # object_goal_dist, sdf_alignment (Task2,3)
        │   │   ├── terminations.py               # task_success, time_out
        │   │   └── observations.py               # rel_ee_object_distance, ee_pos, ee_quat 등
        │   ├── open_drawer/
        │   │   ├── __init__.py                   # Isaac-OpenDrawer-Teacher-v0 / Student-v0 등록
        │   │   ├── teacher_env_cfg.py
        │   │   └── student_env_cfg.py
        │   ├── fetch_tool/                       # [Task 2] 🔴 최고 난이도
        │   │   ├── __init__.py                   # Isaac-FetchTool-Teacher-v0 / Student-v0 등록
        │   │   ├── teacher_env_cfg.py
        │   │   └── student_env_cfg.py
        │   ├── return_tool/
        │   │   ├── __init__.py                   # Isaac-ReturnTool-Teacher-v0 / Student-v0 등록
        │   │   ├── teacher_env_cfg.py
        │   │   └── student_env_cfg.py
        │   └── close_drawer/
        │       ├── __init__.py                   # Isaac-CloseDrawer-Teacher-v0 / Student-v0 등록
        │       ├── teacher_env_cfg.py
        │       └── student_env_cfg.py
        └── agents/
            ├── __init__.py
            ├── ppo_cfg/                          # Task별 PPO 하이퍼파라미터 분리
            │   ├── open_drawer_ppo_cfg.py
            │   ├── fetch_tool_ppo_cfg.py         # Task 2: lr / entropy 별도 튜닝
            │   ├── return_tool_ppo_cfg.py
            │   └── close_drawer_ppo_cfg.py
            └── student_networks.py               # Stage 2: Frame Stacking CNN 인코더 + MLP
```

---

### 학습 실행 순서

```bash
# Stage 1: Teacher 학습 (Task별 각각 실행)
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

## 실제 배포 전략

### 1차 (기본): Teacher + 캘리브레이션 GT 주입

Teacher policy를 그대로 실제에 배포하되, 카메라 캘리브레이션으로 GT 좌표를 구성하여 입력으로 제공한다.

```
실제 배포 관측값 구성:
  robot_joint_angles  ← 조인트 엔코더 직접
  robot_joint_vel     ← 조인트 엔코더 직접
  ee_pos / ee_quat    ← FK로 계산
  gripper_joint_pos   ← 그리퍼 엔코더 직접
  object_pos          ← YOLO + D455f depth → 3D 위치
  object_quat         ← depth 포인트클라우드 or 탑뷰 회전각 추정
  target_pos          ← 사전 측정된 고정 좌표 (Staging / INNER)
  target_quat         ← 사전 측정된 고정값
  target_tool_id      ← 상위 BT에서 주입
```

Teacher observation space와 동일한 형태의 숫자를 실제에서 구성하므로 **Student 없이 Teacher를 그대로 배포**할 수 있다.

### 2차 (여유 시): Student CNN end-to-end

Stage 2a → 2b → 2c DAgger 파이프라인을 통해 카메라 픽셀을 직접 입력받는 Student 정책을 학습한다. 1차 배포가 안정적으로 동작하는 것을 확인한 후 시도한다.

---

## 정책 공유 가능 여부

| 비교 | 공유 가능 여부 | 이유 |
|---|---|---|
| Task 1 ↔ Task 4 | **불가** | setup_j 관절 자세 다름, 경유 waypoint 다름 (silence / opendown), 시작 상태 가정 다름 |
| Task 2 ↔ Task 3 | **불가** | 시작/종료 위치 반대, 파지 대상 공간 특성 다름 |

---

## 주요 참고 사항

- Task 1, 4는 `toolbox_motion.py`에서 실물 4종 시퀀스 검증 완료 (`open_0`, `close_0`, `open_1`, `close_1`)
- Task 2에서 wrist cam observation은 공구함 내부 좁은 공간 파지에 필수
- Teacher 성공률 80% 미만 시 Student 학습 진행 금지
- 공구 6종 (screwdriver · utility_knife · ratchet_wrench · multi_tool · spanner_16mm · socket_19mm) 모두 커버해야 함
