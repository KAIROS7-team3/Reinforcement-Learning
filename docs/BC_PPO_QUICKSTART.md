# BC + PPO Quickstart (Joint-space, SO ARM) — **ReturnTool first**

ReturnTool(staging pick & place) 데모 → BC warm-start → PPO fine-tune.

## Demo mode: BC trajectory sketch (default)

데모 수집은 **PhysX grasp 없음**. kinematic joint follow + grasp assist(object weld).


|               | 데모 수집 (sketch)                                                   | PPO 학습 (Teacher)              |
| ------------- | ---------------------------------------------------------------- | ----------------------------- |
| 팔             | JSON → joint teleport                                            | PD + PhysX                    |
| 공구 pick       | grasp assist weld                                                | **마찰 grasp (PhysX)**          |
| place 목표      | 서랍 안 **대략** (마커2 가이드)                                            | 마커2 중심 **5 cm 이내**            |
| 성공 종료         | 수동 `demo_reset.flag` / `--auto_success` → **demo_place_success** | dist ≤ 5 cm → episode success |
| BC 가치         | approach / carry / place **궤적 스케치**                              | —                             |
| grasp/contact | PPO가 reward로 학습                                                  | ✅                             |


> Real grasp는 PPO 단계에서 배움. sketch 데모만으로 pick 성공 기대하지 말 것.

## Pipeline

```
SO ARM → leader_to_isaac → isaac/joint_command
                              ↓ T4 JSON bridge
    → collect_demos_teleop.py (sketch + grasp assist) → HDF5 → pretrain_bc.py → PPO
```

> `leader_to_isaac.md` = standalone Isaac Sim (3터미널).  
> Isaac Lab 데모 = **T4 JSON bridge 추가** (4터미널).

## 1. Collect demos (4 terminals)


| 터미널 | 역할                       | Python          |
| --- | ------------------------ | --------------- |
| T2  | SO 리더 USB                | 3.10            |
| T3  | leader_to_isaac          | 3.10            |
| T4  | ros_joint_command_bridge | 3.10            |
| T1  | collect_demos_teleop     | 3.11 (isaaclab) |


**기동 순서: T2 → T3 → T4 → T1**

```bash
# T2
export ROS_DOMAIN_ID=71
USB_PORT=/dev/ttyACM0 LEADER_DOF=7 \
  ~/doosan-lerobot-stack/so-doosan-teleoperation-ver3/run_leader_usb.sh

# T3
export ROS_DOMAIN_ID=71 && source /opt/ros/humble/setup.bash
python3 ~/doosan-lerobot-stack/leader_to_isaac.py

# T4
export ROS_DOMAIN_ID=71 && source /opt/ros/humble/setup.bash
cd /home/user/Reinforcement-Learning
python3 scripts/ros_joint_command_bridge.py

# T1 — sketch demo recorder (defaults: cube + grasp assist + 7D HDF5)
cd /home/user/Reinforcement-Learning
../IsaacLab/isaaclab.sh -p scripts/collect_demos_teleop.py \
  --task Isaac-ReturnTool-Teacher-Demo-v0 \
  --dataset ./data/demos/return_tool/dataset.hdf5 \
  --num_demos 5 \
  --auto_success \
  --debug_interval 30
```

T1 시작 로그 확인:

- `teleop_physics=sketch`
- `grasp assist ON`
- `sketch teleop: kinematic JSON follow + grasp assist; no PhysX grasp; records 7D actions`

**조작:** cube 위 접근 → 그리퍼 닫기 → `grasp assist: ENGAGED` → lift → 서랍 안에 **대략** 내려놓기 → gripper 열기 (`released in place`) → 만족하면 `touch /tmp/demo_reset.flag`

> **텔레옵에서는 마커2에 정확히 맞출 필요 없음.** BC는 approach / carry / place **궤적**만 필요.

**데모 종료 (기본):** `--auto_success` 꺼짐 → **자동 export 안 됨** → `touch /tmp/demo_reset.flag`

**자동 export (`--auto_success`):** `demo_place_success` — **서랍 바닥 안착 heuristic** (PPO 5 cm 구와 다름):


| 조건         | 기본값                                                |
| ---------- | -------------------------------------------------- |
| gripper 열림 | `rh_r1 ≤ 0.20` rad                                 |
| 서랍 XY      | drawer floor 전체 (link frame AABB, 벽 margin 2.5 cm) |
| 바닥 Z       | `place_target` z ± `0.04` m                        |
| 속도         | `                                                  |
| 유지         | `--num_success_steps 10` 프레임 연속                    |


```bash
# 자동 export 예시
../IsaacLab/isaaclab.sh -p scripts/collect_demos_teleop.py \
  --task Isaac-ReturnTool-Teacher-Demo-v0 \
  --dataset ./data/demos/return_tool/dataset.hdf5 \
  --num_demos 5 --auto_success --debug_interval 30
```

**튜닝:**

- cube가 손가락 옆에서 따라오면: engage 거리 `--grasp_dist_m 0.08`, `--grasp_snap_inward_m 0.03` (기본)
- 서랍에 놓을 때 마커로 snap (비추천, 기본 OFF): `--place_snap`
- screwdriver: `--tool_asset screwdriver`
- legacy PhysX kinematic: `--teleop_physics legacy` (비추천)

**실패 / 수동 reset:** `touch /tmp/demo_reset.flag`

### 서랍 마커 2개 (`target_frame` 시각화)

`Isaac-ReturnTool-Teacher-`* 에서 서랍 안 RGB 축 + 노란 선 = **FrameTransformer 1개**가 그리는 기준점·목표점.


| 마커              | 이름                   | 의미                                                                  |
| --------------- | -------------------- | ------------------------------------------------------------------- |
| **마커1** (뒤쪽, 안) | `drawer` link **원점** | offset 계산 **기준** (PPO goal 아님)                                      |
| **마커2** (앞쪽)    | `place_target`       | `RETURN_TOOL_PLACE_OFFSET = (0, 0.06, 0.04)` m — **PPO place goal** |
| 노란 선            | —                    | 마커1 → 마커2 offset 벡터                                                 |


- **BC 데모:** 마커2는 **가이드라인** (대략 서랍 앞쪽 바닥).
- **PPO:** 마커2 중심 **5 cm 구 안**이 성공 (아래 §PPO 성공 기준).

`drawer_frame`(OpenDrawer knob)은 ReturnTool에서 `debug_vis=False` → **안 보임**.

### grasp assist (sketch 기본)


| 단계      | DEBUG `grasp=`    | 동작                                          |
| ------- | ----------------- | ------------------------------------------- |
| staging | `LOCK`            | table-lock (cube 고정)                        |
| 닫기      | `GRASP`           | finger midpoint weld (집어 올림)                |
| 열기      | `FREE` / `PLACED` | 현재 위치 release (`--place_snap` 시에만 마커2 snap) |


PhysX grasp 아님 — PPO Teacher에서 마찰 grasp 학습.

## 2. BC pretrain

**Isaac Sim 머신에서 단독 실행** (텔레옵 4터미널 불필요). §1 HDF5만 있으면 됨.

### 사전 확인

```bash
# 데모 개수·차원 (obs 19D, action 7D 기대)
python3 -c "
import h5py
p='./data/demos/return_tool/dataset.hdf5'
with h5py.File(p) as f:
    demos=sorted(k for k in f['data'] if k.startswith('demo_'))
    n=sum(f['data'][d]['actions'].shape[0] for d in demos)
    s0=f['data'][demos[0]]
    print(len(demos), 'demos,', n, 'transitions, obs', s0['obs'].shape, 'act', s0['actions'].shape)
"
```


| 항목         | ReturnTool Teacher                                                      |
| ---------- | ----------------------------------------------------------------------- |
| `--task`   | `Isaac-ReturnTool-Teacher-v0` (PPO와 **동일** actor 구조)                    |
| policy obs | 19D: `joint_pos(7) + object_pos(3) + target_pos(3) + target_tool_id(6)` |
| action     | 7D 절대 joint target (rad)                                                |
| loss       | MSE(`actor(obs)`, demo action), **actor만** 학습 (critic은 랜덤 유지)           |
| env reward | **사용 안 함** — BC는 오프라인 imitation; reward는 §3 PPO에서만 적용                   |


### 실행

```bash
cd /home/user/Reinforcement-Learning
mkdir -p data/checkpoints/return_tool

../IsaacLab/isaaclab.sh -p scripts/pretrain_bc.py \
  --task Isaac-ReturnTool-Teacher-v0 \
  --dataset ./data/demos/return_tool/dataset.hdf5 \
  --output ./data/checkpoints/return_tool/bc_warmstart.pt \
  --epochs 1000 --headless
```

**튜닝 (선택):** `--epochs 300`, `--lr 5e-4`, `--batch_size 128`, `--val_ratio 0.1`

### 기대 로그

```
[INFO] transitions=2180 obs_dim=19 action_dim=7 (Teacher policy=19, action=7)
[INFO] train=1962 val=218 epochs=200
[INFO] epoch 1/200 train_mse=... val_mse=...
...
[INFO] Saved BC checkpoint → .../bc_warmstart.pt (best val_mse=...)
[INFO] PPO warm-start:
  mkdir -p logs/rsl_rl/return_tool_teacher/bc_warmstart
  cp .../bc_warmstart.pt logs/rsl_rl/return_tool_teacher/bc_warmstart/model_0.pt
  ...
```

- `val_mse`가 epoch마다 내려가면 정상. best checkpoint가 자동 저장됨.
- obs/action 차원 불일치 시 스크립트가 즉시 에러 (데모 env ≠ Teacher env).

### 산출물

`data/checkpoints/return_tool/bc_warmstart.pt` — rsl_rl `model_0.pt` 형식 (`model_state_dict`, `iter=0`).  
→ §3 PPO `--resume` 에 복사해 사용.

## 3. PPO fine-tune

```bash
mkdir -p logs/rsl_rl/return_tool_teacher/bc_warmstart
cp data/checkpoints/return_tool/bc_warmstart.pt \
   logs/rsl_rl/return_tool_teacher/bc_warmstart/model_0.pt

../IsaacLab/isaaclab.sh -p scripts/train.py \
  --task Isaac-ReturnTool-Teacher-v0 \
  --num_envs 2048 --headless \
  --max_iterations 2000 \
  --resume --load_run bc_warmstart --checkpoint model_0.pt \
  --run_name return_tool_bc_ppo_v1
```

> VRAM 여유 있으면 `--num_envs 4096` (sample throughput ↑). 2048도 학습 가능 — iteration 수만 조금 더 필요할 수 있음.  
> iteration 수: `--max_iterations N` (또는 별칭 `--iterations N`). 미지정 시 task PPO cfg 기본값 (`return_tool`: 2000).

BC warm-start 효과 없으면 scratch PPO와 A/B 비교.

### PPO 성공 기준 (ReturnTool Teacher) — **바닥 안착 아님**

현재 `return_tool_success` / `success_bonus` 는 **물리 안착 검사 없음**:


| 검사함                                          | 검사 안 함                        |
| -------------------------------------------- | ----------------------------- |
| 큐브 중심 ↔ `place_target`(마커2) **3D 거리 ≤ 5 cm** | 서랍 바닥 **접촉**                  |
|                                              | gripper **열림**                |
|                                              | 속도 ≈ 0 (안착)                   |
|                                              | 공중에 들고 있어도 5 cm 안이면 **성공 가능** |


- Reward: `object_goal_dist` (거리↓, w=-1) + `success_bonus` (5 cm 이내, w=10).
- Observation: GT `target_pos` = 마커2 (goal-conditioning).
- 마커2 offset `(0, 0.06, 0.04)` 는 **“바닥에 놓인 pose” proxy** — sim에서 튜닝 필요.

**구현:** `source/tool_transfer_bot/tasks/mdp/terminations.py` → `demo_place_success`

**추후 PPO 강화 (미구현):** dist + `|vel| < ε` + gripper open + contact.

**데모 vs PPO place 정밀도**


|       | BC 텔레옵                             | PPO Teacher                         |
| ----- | ---------------------------------- | ----------------------------------- |
| 목표    | 서랍 안 대략 place                      | 마커2 **5 cm 이내**                     |
| grasp | grasp assist                       | PhysX 마찰                            |
| 성공 판정 | 수동 flag / `**demo_place_success`** | `return_tool_success` (dist ≤ 5 cm) |


## 4. Play

```bash
../IsaacLab/isaaclab.sh -p scripts/play_env.py \
  --task Isaac-ReturnTool-Teacher-Play-v0 --num_envs 1
```

## Home pose (ReturnTool)


| joint   | deg |
| ------- | --- |
| joint_1 | 0   |
| joint_2 | 0   |
| joint_3 | 90  |
| joint_4 | 0   |
| joint_5 | 90  |
| joint_6 | 0   |
| rh_r1   | 0   |


## Tunables

`source/tool_transfer_bot/assets/environments.py`:

- `RETURN_TOOL_STAGING_POS` — 공구 초기 위치
- `RETURN_TOOL_PLACE_OFFSET` — **마커2 / PPO `place_target`** (drawer link frame)
- `RETURN_TOOL_DRAWER_OPEN_JOINT` — `-0.2` m (fully open)

`scripts/collect_demos_teleop.py` (데모 전용):


| 플래그                             | 기본                 | 설명                                         |
| ------------------------------- | ------------------ | ------------------------------------------ |
| `--auto_success`                | OFF                | ON → `demo_place_success` N프레임 유지 시 export |
| `--demo_success_gripper_rad`    | 0.20               | gripper 열림 (rh_r1 ≤)                       |
| `--demo_success_z_band`         | 0.04               | drawer frame Z band (바닥 proxy)             |
| `--demo_success_tool_xy_margin` | cube half (2.5 cm) | drawer 벽 inset                             |
| `--demo_success_max_vel`        | 0.10               | 최대 선속도 (m/s)                               |
| `--place_snap`                  | OFF                | ON 시 gripper release 때 마커2로 teleport       |
| `--place_radius_m`              | 0.12               | `--place_snap` XY 허용 반경                    |
| `--grasp_dist_m`                | 0.10               | engage: finger midpoint–cube 거리            |
| `--grasp_snap_inward_m`         | 0.03               | engage 시 cube를 손가락 쪽으로 nudge               |


`source/tool_transfer_bot/assets/doosan_e0509.py`:

- `GRIPPER_STIFFNESS_RL` / `GRIPPER_DAMPING_RL` — **0.0** (PPO·teleop gripper PD off)

