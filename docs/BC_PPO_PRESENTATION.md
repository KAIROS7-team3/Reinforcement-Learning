# Pick & Place 학습: BC + PPO

**Task:** staging area 큐브 **pick → carry → place** (Doosan e0509, Isaac Lab)

---

## 1. 왜 단일 PPO만으로는 어려운가

Pick & place는 **한 에피소드 안에서** approach → grasp → lift → carry → place **여러 단계를 연속으로** 수행해야 하는 **multi-phase manipulation**이다. (에피소드 ~12 s — 전체 공정 BT를 말하는 “long-horizon”과는 다름)

| 어려움 | 설명 |
|--------|------|
| **Sparse reward** | 성공(물체 ↔ place goal 5 cm 이내) 전까지 terminal bonus가 거의 없음 |
| **Multi-phase** | pick 실패 시 place까지 reward 경로가 끊김 — 단계가 **순서대로** 열려야 함 |
| **PhysX grasp** | 마찰·접촉·그리퍼 타이밍 — 랜덤 정책이 우연히 집기 어려움 |
| **탐색 비용** | 2048 병렬 env로 돌려도, home에서 무작위로 grasp까지 가는 샘플이 극히 적음 |

### 실험 결과 — PPO scratch (BC 없이)

**설정:** `Isaac-ReturnTool-Teacher-v0`, random init (`--resume` 없음), 2048 env, episode 12 s (≈ **720 step**), Lift-style reward 7항목.

| 지표 (TensorBoard) | iter 0 | iter 200+ | 해석 |
|--------------------|--------|-----------|------|
| `Episode_Termination/task_success` | 0% | **0%** | place goal 5 cm 이내 **한 번도** 없음 |
| `Episode_Termination/time_out` | ~100% | **100%** | 매 에피소드 **끝까지** 버팀 — 조기 성공·실패 없음 |
| `Train/mean_episode_length` | ~720 | ~720 | 처음부터 max step — 짧게 끝나는 “의미 있는 시도” 없음 |

**보상 항목별 (에피소드 평균):**

| 단계 | TB 태그 | iter 0 → 200+ | 왜 이렇게 나오나 |
|------|---------|---------------|------------------|
| **Approach** | `reaching_object` | ~0.0002 → **~0.003** (소폭↑) | EE–큐브 거리 tanh — **항상 켜진 dense reward**. 팔이 큐브 쪽으로 조금 움직이면 오름 |
| **Grasp/Lift** | `lifting_object` | **0 → 0** | 큐브 Z > 4 cm 조건 — **PhysX grasp 없으면 영원히 0** |
| **Carry/Place** | `object_goal_tracking` (+ fine) | **0 → 0** | lift gate 통과 후에만 활성 — pick 실패 시 **신호 자체가 없음** |
| **Success** | `success_bonus` | **0 → 0** | goal 5 cm 이내 binary — 도달 불가 |

**행동 패턴 (정책이 실제로 하는 일):**

1. **Home 근처에서 팔만 살짝 흔듦** — `reaching_object`만 미세 상승, 큐브까지 도달·그리퍼 닫기는 학습 안 됨.
2. **Grasp 단계 미진입** — 마찰·접촉·핀치 타이밍은 random exploration으로 우연히 찾기 어려움 → lift/place 보상 경로 **완전 차단**.
3. **Timeout local optimum** — 실패해도 에피소드가 끊기지 않고 12 s 풀가동 → PPO 입장에선 “버티기”가 안전한 전략. `action_rate_l2` 페널티만 약간 맞으며 **의미 없는 동작을 유지**.

**한 줄 요약:** PPO는 **approach shaping만** 따라가는 **얕은 local optimum**에 갇히고, multi-phase 중 **grasp 이후 단계는 reward가 0**이라 gradient가 거의 없음 → 200 iter 이상 돌려도 `task_success` 0%.

> 참고: BC warm-start 후에도 PPO가 몇 iter 지나면 동일 패턴(timeout 100%, lift/place 0)으로 **수렴** — scratch의 실패 양상과 같다는 뜻에서 BC가 “시작점”만 줄 뿐 근본 해결은 아님 (§3.1).

단순 **reach-only**나 **lift-only**(Isaac Lab Lift)는 단계가 짧고 lift gate로 보상을 쪼갤 수 있다. **pick + carry + place**는 grasp 물리가 필수라, approach만 오르는 scratch PPO로는 **다음 단계로 넘어갈 탐색 신호가 없다**.

---

## 2. BC + PPO — 무엇이고, 왜 쓰는가

### 구조

```
사람 텔레옵 50 demos (HDF5)
        ↓  BC (offline, MSE)
   actor: "데모처럼 팔 움직이기" 학습
        ↓  warm-start (model_0.pt)
   PPO (online, PhysX + reward)
        ↓
   grasp · lift · place를 물리 시뮬에서 보완
```

### 역할 분담

| | BC | PPO |
|--|-----|-----|
| **입력** | 데모 (obs, action) 쌍 | PhysX rollout + reward |
| **배우는 것** | approach / carry / place **궤적 스케치** | **진짜 grasp**, lift, 정밀 place |
| **Reward** | 사용 안 함 | 사용 (Lift 스타일 7항목) |

### 데모는 “스케치”다 (중요)

데모 수집 시 **PhysX grasp 없음** — kinematic joint + **grasp assist**(큐브 weld).  
BC는 “영상 보고 핸들 외운 것”, PPO는 “실제 도로에서 운전 + 점수표”에 가깝다.

### 도입 이점

| 이점 | 내용 |
|------|------|
| **탐색 축소** | PPO가 home부터 random walk 안 해도 됨 — BC가 큐브 쪽 궤적 제공 |
| **사람 지식 활용** | SO ARM 텔레옵 ~50회로 cheap expert trajectory |
| **인프라 재사용** | Isaac Lab `rsl_rl` 그대로 — BC ckpt를 `model_0.pt`로 `--resume` |
| **BC 품질 검증됨** | open-loop val_mse **0.0013**, grasp 구간 action 오차 mean **0.15 rad** — **접근 궤적은 학습됨** |

---

## 3. 그런데도 어려웠다 — 원인과 문제

### 3.1 BC는 됐는데 PPO가 망가뜨림

| iter | 현상 |
|------|------|
| **0** (BC weight) | ep_len ~**79**, 큐브 쪽으로 가는 흔적 |
| **5+** | ep_len **720**, timeout 100%, BC 궤적 **붕괴** |

**원인: covariate shift** — BC는 데모 obs(offline)만 봤고, PPO는 PhysX closed-loop에서 policy가 바꾼 obs를 본다. PPO gradient 몇 번이면 데모와 다른 상태 분포로 drift.

### 3.2 Sim gap — 데모 환경 ≠ PPO 환경

| | 데모 / BC | PPO |
|--|-----------|-----|
| 팔 | kinematic teleport | **PhysX PD** |
| Pick | grasp assist weld | **마찰 grasp** (BC·데모에 없음) |
| Place 기준 | place goal **대략** | goal 중심 **5 cm 구** (정밀) |

BC가 배운 “접근”은 **kinematic 기준**이지, PPO PhysX에서 같은 action을 넣어도 EE 궤적이 달라진다.

### 3.3 보상 설계 시행착오

| 시도 | 문제 |
|------|------|
| 초기 4항목 (approach + dist + success) | **grasp 보상 없음** → 접근만 하고 timeout |
| multi-stage 14항목 (grasp 세분화) | 항목 과다, PPO가 BC faster 파괴 |
| Lift식 7항목 (현재) | `lifting_object` **0** — lift gate(Z>4cm) 통과 못 함 |

**병목:** grasp → lift → place **체인 전체**가 안 열림. `reaching_object` ≈ 0.001처럼 작아 보여도, 에피소드 평균이라 BC 접근 실패를 뜻하지는 않음.

### 3.4 정리 — 뭐가 문제였나

1. **단일 PPO:** sparse reward + multi-phase + PhysX grasp → 탐색 비효율  
2. **BC+PPO 도입:** 접근 궤적 bootstrap ✅ — 그러나 **warm-start가 PPO에 못 버팀**  
3. **근본 gap:** 데모는 sketch(kinematic + assist), PPO는 real physics — **BC만으로 grasp를 채울 수 없고**, PPO가 BC를 지키면서 grasp를 배우는 메커니즘이 아직 없음  

### 3.5 다음에 시도할 것 (한 줄)

- PPO 초기 **BC 고정 / KL penalty** (BC 파괴 방지)  
- PPO env **grasp assist fade** (데모→PhysX bridge)  
- grasp 전용 shaping 보상 추가 (Lift만으로는 부족)

---

*실행 커맨드·상세 설정: [`BC_PPO_QUICKSTART.md`](./BC_PPO_QUICKSTART.md)*
