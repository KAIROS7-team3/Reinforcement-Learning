#!/usr/bin/env python3
"""Teleop + per-term reward logging inside the ManagerBased RL env.

Uses ``Isaac-OpenDrawer-Teacher-Teleop-v0`` (joint-position actions, same scene/rewards
as training). External teleop via ``joint_teleop_gui.py`` + ROS JSON bridge.

Three-terminal workflow
-----------------------
T1 — RL env + reward log (Isaac Lab conda / isaaclab.sh):

    cd /home/user/Reinforcement-Learning
    ../IsaacLab/isaaclab.sh -p scripts/teleop_reward_monitor.py \\
        --task Isaac-OpenDrawer-Teacher-Teleop-v0 --num_envs 1

T2 — ROS bridge (system Python 3.10 + ROS, NOT env_isaaclab):

    export ROS_DOMAIN_ID=71
    source /opt/ros/humble/setup.bash
    python3 scripts/ros_joint_command_bridge.py

T3 — Joint teleop GUI (same ROS_DOMAIN_ID):

    export ROS_DOMAIN_ID=71
    source /opt/ros/humble/setup.bash
    python3 ~/Desktop/joint_teleop_gui.py

Pass criteria (reward design sanity check)
------------------------------------------
While opening the drawer with teleop, expect roughly this order:

1. ``approach_ee_handle``, ``align_ee_handle`` increase near the handle
2. ``align_grasp_around_handle``, ``approach_gripper_handle``, ``grasp_handle`` rise at pinch
3. ``open_drawer_bonus``, ``multi_stage_open_drawer`` rise as |drawer_joint| grows
4. ``task_success`` when drawer_joint ≤ -0.15 m

If the drawer opens but stage-2/3 terms stay ~0, reward shaping or knob frame is wrong.

USD prim note: reward uses ``drawer_frame`` (knob center). Prim ``handle`` is the toolbox
top carry handle — not the drawer-front pull (``drawer/drawer`` mesh texture).

Physics modes (``--physics-mode``)
----------------------------------
``step`` (default, v18): always ``env.step()`` — PhysX ON every frame. Slower arm teleop but
no freeze teleport through the knob; stable grasp + drawer pull E2E.

``hybrid`` (v17 legacy): freeze when far / idle; ``env.step`` near handle — fast arm teleop but
gripper jitter at contact. Use only for quick arm positioning checks.

The main loop is rate-limited to ``env.step_dt`` (120 Hz teleop env) so sim time stays real-time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RL env teleop with reward-term logging.")
parser.add_argument("--task", type=str, default="Isaac-OpenDrawer-Teacher-Teleop-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--joint_file",
    type=str,
    default="/tmp/isaac_teleop_joints.json",
    help="JSON from ros_joint_command_bridge.py (joint targets in rad).",
)
parser.add_argument(
    "--joint_stale_sec",
    type=float,
    default=0.5,
    help="If JSON is older than this, hold current sim joint pose.",
)
parser.add_argument(
    "--feedback_file",
    type=str,
    default="/tmp/isaac_sim_joint_states.json",
    help="Sim joint positions (rad) for GUI sync via ros_joint_command_bridge.",
)
parser.add_argument("--log_interval", type=int, default=15, help="Print every N env steps.")
parser.add_argument(
    "--csv",
    type=str,
    default="/tmp/teleop_reward_log.csv",
    help="CSV log path.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O."
)
parser.add_argument(
    "--home_settle_steps",
    type=int,
    default=60,
    help="Physics steps to hold home pose after reset before teleop.",
)
parser.add_argument(
    "--home_warmup_steps",
    type=int,
    default=0,
    help="Post-freeze env.step count (0 recommended — warm-up drifts home/gripper).",
)
parser.add_argument(
    "--physics-mode",
    type=str,
    choices=("step", "hybrid"),
    default="step",
    help="step=env.step when teleop moves (E2E grasp/pull); hybrid=v17 legacy.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import tool_transfer_bot  # noqa: F401, E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from tool_transfer_bot.tasks import mdp  # noqa: E402
from tool_transfer_bot.assets.doosan_e0509 import RL_HOME_JOINT_DEG  # noqa: E402

SCRIPT_VERSION = "teleop_reward_monitor 2026-06-09 physics-v18.3"
ARM_ACTION_DIM = 6
GRIPPER_ACTION_INDEX = 6  # rh_r1 in expanded action vector
# hybrid mode only: EE TCP distance to knob center (drawer_frame) for contact-zone stepping.
NEAR_HANDLE_M = 0.05
HOME_CHECK_FILE = "/tmp/teleop_home_check.txt"

# JSON bridge / GUI publish arm + rh_r1 only; we expand to full action below.
JSON_JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)] + ["rh_r1"]
MIMIC_GEARING = 1.0  # Isaac joint space: rh_l1/rh_l2/rh_r2 same sign as rh_r1 (see leader_to_isaac.py)
FEEDBACK_JOINTS = JSON_JOINT_NAMES + ["rh_l2"]
REWARD_TERMS = [
    "approach_ee_handle",
    "align_ee_handle",
    "approach_gripper_handle",
    "align_grasp_around_handle",
    "grasp_handle",
    "open_drawer_bonus",
    "multi_stage_open_drawer",
    "action_rate_l2",
    "joint_vel_l2",
]
RAW_REWARD_FUNCS = {
    "approach_ee_handle": lambda env: mdp.approach_ee_handle(env),
    "align_ee_handle": lambda env: mdp.align_ee_handle(env),
    "approach_gripper_handle": lambda env: mdp.approach_gripper_handle(
        env, z_tol=0.02, y_tol=0.03, x_tol=0.025
    ),
    "align_grasp_around_handle": lambda env: mdp.align_grasp_around_handle(
        env, z_sigma=0.015, y_min_sep=0.008
    ),
    "grasp_handle": lambda env: mdp.grasp_handle(
        env,
        threshold=0.06,
        grasp_align_threshold=0.3,
        open_joint_pos=0.0,
        close_joint_pos=math.radians(60.0),
        asset_cfg=SceneEntityCfg("robot", joint_names=["rh_r1"]),
    ),
    "open_drawer_bonus": lambda env: mdp.open_drawer_bonus(
        env,
        asset_cfg=SceneEntityCfg("toolbox", joint_names=["drawer_joint"]),
        gripper_asset_cfg=SceneEntityCfg("robot", joint_names=["rh_r1"]),
        close_threshold=math.radians(30.0),
    ),
    "multi_stage_open_drawer": lambda env: mdp.multi_stage_open_drawer(
        env,
        asset_cfg=SceneEntityCfg("toolbox", joint_names=["drawer_joint"]),
        gripper_asset_cfg=SceneEntityCfg("robot", joint_names=["rh_r1"]),
        close_threshold=math.radians(30.0),
    ),
}


class JointFileTeleop:
    """Read latest joint targets written by ros_joint_command_bridge (arm + rh_r1)."""

    def __init__(self, path: str, stale_sec: float, device: torch.device, json_joint_names: list[str]):
        self.path = path
        self.stale_sec = stale_sec
        self.device = device
        self.json_joint_names = json_joint_names
        self._last: torch.Tensor | None = None
        self._last_mtime = 0.0

    def reset_from_robot(self, robot, json_joint_names: list[str] | None = None) -> None:
        names = json_joint_names or self.json_joint_names
        pos = robot.data.joint_pos[0].clone()
        targets = [float(pos[robot.joint_names.index(name)]) for name in names]
        self._last = torch.tensor(targets, device=self.device, dtype=torch.float32)

    def read_action(
        self,
        num_envs: int,
        robot=None,
        hold_current: bool = False,
        fallback: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if hold_current and robot is not None:
            self.reset_from_robot(robot)
            return self._last.unsqueeze(0).repeat(num_envs, 1)

        if not os.path.isfile(self.path):
            if fallback is not None:
                return fallback.repeat(num_envs, 1)
            if self._last is None:
                return None
            return self._last.unsqueeze(0).repeat(num_envs, 1)

        mtime = os.path.getmtime(self.path)
        age = time.time() - mtime
        if age > self.stale_sec:
            if fallback is not None:
                return fallback.repeat(num_envs, 1)
            if self._last is not None:
                return self._last.unsqueeze(0).repeat(num_envs, 1)

        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            if fallback is not None:
                return fallback.repeat(num_envs, 1)
            if self._last is None:
                return None
            return self._last.unsqueeze(0).repeat(num_envs, 1)

        targets = []
        for name in self.json_joint_names:
            if name not in data:
                if fallback is not None:
                    return fallback.repeat(num_envs, 1)
                if self._last is None:
                    return None
                return self._last.unsqueeze(0).repeat(num_envs, 1)
            targets.append(float(data[name]))

        self._last = torch.tensor(targets, device=self.device, dtype=torch.float32)
        self._last_mtime = mtime
        return self._last.unsqueeze(0).repeat(num_envs, 1)


def _weighted_reward_terms(env) -> dict[str, float]:
    rm = env.unwrapped.reward_manager
    return {name: vals[0] for name, vals in rm.get_active_iterable_terms(0)}


def _raw_reward_terms(env) -> dict[str, float]:
    u = env.unwrapped
    out = {}
    for name, fn in RAW_REWARD_FUNCS.items():
        out[f"raw_{name}"] = float(fn(u)[0].item())
    return out


def _scene_metrics(env) -> dict[str, float]:
    u = env.unwrapped
    toolbox = u.scene["toolbox"]
    drawer = float(toolbox.data.joint_pos[0, toolbox.joint_names.index("drawer_joint")].item())
    ee = u.scene["ee_frame"].data.target_pos_w[0, 0]
    handle = u.scene["drawer_frame"].data.target_pos_w[0, 0]
    dist = float(torch.linalg.norm(ee - handle).item())
    return {"drawer_joint": drawer, "ee_handle_dist_m": dist}


def _ee_handle_distance_m(raw) -> float:
    ee = raw.scene["ee_frame"].data.target_pos_w[0, 0]
    handle = raw.scene["drawer_frame"].data.target_pos_w[0, 0]
    return float(torch.linalg.norm(ee - handle).item())


def _targets_to_action(joint_names: list[str], targets_rad: dict[str, float], device: torch.device) -> torch.Tensor:
    vals = [float(targets_rad[name]) for name in joint_names]
    return torch.tensor(vals, device=device, dtype=torch.float32).unsqueeze(0)


def _home_targets_rad() -> dict[str, float]:
    return {name: math.radians(RL_HOME_JOINT_DEG[name]) for name in RL_HOME_JOINT_DEG if name.startswith("joint_") or name == "rh_r1"}


def _write_joint_feedback(robot, path: str) -> None:
    """Write current sim joint positions for GUI sync (via ROS bridge)."""
    data: dict[str, float] = {"stamp_sec": time.time()}
    for name in FEEDBACK_JOINTS:
        if name in robot.joint_names:
            idx = robot.joint_names.index(name)
            data[name] = float(robot.data.joint_pos[0, idx].item())
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _expand_gripper_mimic(core: torch.Tensor) -> torch.Tensor:
    """Expand (N, 7) arm+rh_r1 → (N, 10) with rh_r2/rh_l1/rh_l2 = rh_r1."""
    r1 = core[:, -1:]
    mimic = MIMIC_GEARING * r1
    return torch.cat([core, mimic, mimic, mimic], dim=-1)


def _home_core(device: torch.device) -> torch.Tensor:
    return _targets_to_action(JSON_JOINT_NAMES, _home_targets_rad(), device)


def _home_action(device: torch.device) -> torch.Tensor:
    return _expand_gripper_mimic(_home_core(device))


def _log_joint_pose(robot, label: str, lines_out: list[str] | None = None) -> None:
    block = [f"[INFO] {label}"]
    block.append("[INFO] default_joint_pos (deg) from cfg:")
    for name in [f"joint_{i}" for i in range(1, 7)]:
        idx = robot.joint_names.index(name)
        default_deg = math.degrees(float(robot.data.default_joint_pos[0, idx].item()))
        block.append(f"  {name}: default={default_deg:6.1f}°")
    for name in [f"joint_{i}" for i in range(1, 7)]:
        idx = robot.joint_names.index(name)
        actual = math.degrees(float(robot.data.joint_pos[0, idx].item()))
        expected = RL_HOME_JOINT_DEG[name]
        delta = actual - expected
        block.append(f"  {name}: expected={expected:6.1f}°  actual={actual:6.1f}°  Δ={delta:+5.1f}°")
    for name in ("rh_r1", "rh_l1", "rh_l2", "rh_r2"):
        if name in robot.joint_names:
            idx = robot.joint_names.index(name)
            actual = math.degrees(float(robot.data.joint_pos[0, idx].item()))
            if name == "rh_r1":
                block.append(f"  {name}: actual={actual:6.1f}°")
            elif name == "rh_l1":
                r1_idx = robot.joint_names.index("rh_r1")
                r1 = math.degrees(float(robot.data.joint_pos[0, r1_idx].item()))
                block.append(
                    f"  {name}: actual={actual:6.1f}°  expected={r1:6.1f}°  "
                    f"Δ={actual - r1:+5.1f}°"
                )
            else:
                block.append(f"  {name}: actual={actual:6.1f}°")
    text = "\n".join(block)
    print(text, flush=True)
    if lines_out is not None:
        lines_out.append(text)
    else:
        with open(HOME_CHECK_FILE, "w", encoding="utf-8") as f:
            f.write(text + "\n")


def _arm_action_changed(actions: torch.Tensor, prev: torch.Tensor | None, tol: float = 1e-4) -> bool:
    """True only when joint_1..joint_6 targets changed (not gripper)."""
    if prev is None:
        return True
    delta = torch.max(torch.abs(actions[:, :ARM_ACTION_DIM] - prev[:, :ARM_ACTION_DIM]))
    return float(delta.item()) > tol


def _gripper_action_changed(actions: torch.Tensor, prev: torch.Tensor | None, tol: float = 1e-4) -> bool:
    if prev is None:
        return True
    delta = torch.abs(actions[:, GRIPPER_ACTION_INDEX] - prev[:, GRIPPER_ACTION_INDEX])
    return float(delta.item()) > tol


def _physics_action_hold_arm(robot, teleop_action: torch.Tensor, arm_joint_ids: list[int]) -> torch.Tensor:
    """Keep arm at current sim pose while applying teleop gripper targets (contact-safe step)."""
    physics_action = teleop_action.clone()
    physics_action[:, :ARM_ACTION_DIM] = robot.data.joint_pos[:, arm_joint_ids]
    return physics_action


def _freeze_pose(
    robot,
    raw,
    target: torch.Tensor,
    action_joint_ids: list[int],
) -> None:
    """Teleport all action joints (incl. mimic fingers) — no env.step / reward physics."""
    if target.dim() == 1:
        target = target.unsqueeze(0)
    zero_vel = torch.zeros_like(target)
    robot.write_joint_state_to_sim(target, zero_vel, joint_ids=action_joint_ids)
    robot.set_joint_position_target(target, joint_ids=action_joint_ids)
    raw.scene.write_data_to_sim()
    _refresh_visual(raw)


def _refresh_visual(raw) -> None:
    # Fabric OFF: viewport reads USD/PhysX directly — need sim.step to show joint teleports.
    if not raw.sim.cfg.use_fabric:
        raw.sim.step(render=False)
    if raw.sim.has_gui() or raw.sim.has_rtx_sensors():
        raw.sim.render()
    raw.scene.update(dt=raw.physics_dt)


def _compute_idle_rewards(raw) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reward/termination without env.step (robot frozen at home)."""
    raw.episode_length_buf += 1
    raw.common_step_counter += 1
    raw.reset_buf = raw.termination_manager.compute()
    raw.reset_terminated = raw.termination_manager.terminated
    raw.reset_time_outs = raw.termination_manager.time_outs
    raw.reward_buf = raw.reward_manager.compute(dt=raw.step_dt)
    return raw.reward_buf, raw.reset_terminated, raw.reset_time_outs


def _settle_at_home_step(env, home: torch.Tensor, num_steps: int) -> None:
    """Short PhysX warm-up from an already-frozen home pose."""
    for _ in range(max(0, num_steps)):
        env.step(home)


def _settle_at_home(
    env,
    robot,
    raw,
    action_joint_names: list[str],
    num_steps: int,
    physics_mode: str,
    warmup_steps: int = 5,
) -> torch.Tensor:
    """Snap to exact home via freeze, then optional PhysX warm-up (step mode)."""
    home = _home_action(robot.device)
    action_joint_ids = [robot.joint_names.index(n) for n in action_joint_names]

    report: list[str] = [f"[INFO] {SCRIPT_VERSION}  physics_mode={physics_mode}"]
    _freeze_pose(robot, raw, home, action_joint_ids)
    _log_joint_pose(robot, "joint pose after freeze snap to home", report)

    if physics_mode == "step" and warmup_steps > 0:
        print(
            f"[INFO] PhysX warm-up: {warmup_steps} env.step(home) from frozen pose ...",
            flush=True,
        )
        _settle_at_home_step(env, home, warmup_steps)
        _log_joint_pose(robot, "joint pose after step warm-up", report)
    elif physics_mode != "step":
        for _ in range(min(num_steps, 5)):
            _refresh_visual(raw)
        _log_joint_pose(robot, "joint pose after hybrid settle renders", report)

    # Warn if arm drifted from RL home after settle.
    for name in [f"joint_{i}" for i in range(1, 7)]:
        idx = robot.joint_names.index(name)
        actual = math.degrees(float(robot.data.joint_pos[0, idx].item()))
        expected = RL_HOME_JOINT_DEG[name]
        if abs(actual - expected) > 5.0:
            msg = (
                f"[WARN] {name} drifted after home settle: "
                f"expected={expected:.1f}° actual={actual:.1f}° "
                f"(try --home_warmup_steps 0 or restart T1)"
            )
            print(msg, flush=True)
            report.append(msg)
            break

    with open(HOME_CHECK_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n".join(report) + "\n")
    return home


def _apply_teleop_physics(
    env,
    raw,
    robot,
    full: torch.Tensor,
    last_applied_full: torch.Tensor | None,
    json_fresh: bool,
    physics_mode: str,
    action_joint_ids: list[int],
    arm_sim_joint_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run one teleop frame: step (default) or hybrid freeze/step."""
    if physics_mode == "step":
        if not json_fresh:
            hold = last_applied_full if last_applied_full is not None else full
            _freeze_pose(robot, raw, hold, action_joint_ids)
            reward, terminated, truncated = _compute_idle_rewards(raw)
            return reward, terminated, truncated, last_applied_full

        # Always PhysX while teleop is live — freeze-on-idle made fingers pass through knob.
        _, reward, terminated, truncated, _ = env.step(full)
        last_applied_full = full.clone()
        return reward, terminated, truncated, last_applied_full

    # hybrid (v17 legacy): fast freeze far; step near handle for contact
    if json_fresh:
        arm_changed = _arm_action_changed(full, last_applied_full)
        gripper_changed = _gripper_action_changed(full, last_applied_full)
        near_handle = _ee_handle_distance_m(raw) < NEAR_HANDLE_M

        if near_handle:
            if gripper_changed:
                step_action = _physics_action_hold_arm(robot, full, arm_sim_joint_ids)
                _, reward, terminated, truncated, _ = env.step(step_action)
            elif arm_changed:
                _freeze_pose(robot, raw, full, action_joint_ids)
                reward, terminated, truncated = _compute_idle_rewards(raw)
            else:
                hold = last_applied_full if last_applied_full is not None else full
                _, reward, terminated, truncated, _ = env.step(hold)
        elif arm_changed or gripper_changed:
            step_action = (
                full if arm_changed else _physics_action_hold_arm(robot, full, arm_sim_joint_ids)
            )
            _, reward, terminated, truncated, _ = env.step(step_action)
        else:
            _freeze_pose(robot, raw, full, action_joint_ids)
            reward, terminated, truncated = _compute_idle_rewards(raw)
        last_applied_full = full.clone()
    else:
        hold = last_applied_full if last_applied_full is not None else full
        _freeze_pose(robot, raw, hold, action_joint_ids)
        _refresh_visual(raw)
        reward, terminated, truncated = _compute_idle_rewards(raw)

    return reward, terminated, truncated, last_applied_full


def main() -> None:
    print(f"[INFO] {SCRIPT_VERSION}", flush=True)
    print(f"[INFO] home check file → {HOME_CHECK_FILE}", flush=True)

    task_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=False if args_cli.disable_fabric else None,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw = env.unwrapped
    use_fabric = raw.sim.cfg.use_fabric
    print(f"[INFO] sim.use_fabric={use_fabric}", flush=True)
    if not use_fabric:
        print(
            "[INFO] Fabric OFF: freeze frames call sim.step(render=False) so the viewport updates.",
            flush=True,
        )
    robot = raw.scene["robot"]
    active = raw.action_manager.active_terms
    if "joint_action" not in active:
        raise RuntimeError(
            f"Expected action term 'joint_action' for teleop. active_terms={active}. "
            "Reinstall package: pip install -e .  and use Isaac-OpenDrawer-Teacher-Teleop-v0"
        )
    action_term = raw.action_manager.get_term("joint_action")
    action_joint_names = list(action_term._joint_names)
    teleop = JointFileTeleop(args_cli.joint_file, args_cli.joint_stale_sec, raw.device, JSON_JOINT_NAMES)

    print(f"[INFO] task={args_cli.task}", flush=True)
    print(f"[INFO] robot joints={robot.joint_names}", flush=True)
    print(f"[INFO] json joints={JSON_JOINT_NAMES}  action joints={action_joint_names}", flush=True)
    print("[INFO] gripper mimic: rh_r2/rh_l1/rh_l2 = rh_r1 (same joint-space angle)", flush=True)
    print(f"[INFO] joint_file={args_cli.joint_file}", flush=True)
    print(f"[INFO] feedback_file={args_cli.feedback_file}  (→ isaac/joint_states via T2)", flush=True)
    print(f"[INFO] physics_mode={args_cli.physics_mode}", flush=True)
    if args_cli.physics_mode == "step":
        print(
            "[INFO] Physics (step): freeze snap at home; live teleop → env.step every frame "
            "(PhysX/collision ON). Aim at drawer_frame marker (FurnitureKnob_01 / "
            "drawer_handle_top). Not prim ``handle`` (toolbox top carry handle) and not "
            "the drawer-front pull texture on drawer/drawer mesh.",
            flush=True,
        )
    else:
        print(
            "[INFO] Physics (hybrid): far → freeze; far arm move → env.step; "
            f"within {NEAR_HANDLE_M:.2f} m → step for gripper/hold, arm tweak → freeze.",
            flush=True,
        )
    print(
        "[INFO] Knob pinch collision: assets/toolbox_rl_flat.usda "
        "(FurnitureKnob_01/Mesh + invisible grasp_collision/bar @ knob height). "
        "Re-run scripts/fix_toolbox_handle_grasp.py if missing.",
        flush=True,
    )
    print(f"[INFO] CSV → {args_cli.csv}", flush=True)

    csv_fields = ["step", "total_reward", "drawer_joint", "ee_handle_dist_m"]
    csv_fields += REWARD_TERMS
    csv_fields += [f"raw_{k}" for k in RAW_REWARD_FUNCS]

    if not os.path.isfile(args_cli.joint_file):
        print(
            f"[WARN] joint file not found: {args_cli.joint_file}\n"
            "       Teleop needs 3 terminals:\n"
            "       T1: isaaclab.sh -p scripts/teleop_reward_monitor.py (this)\n"
            "       T2: python3 scripts/ros_joint_command_bridge.py\n"
            "       T3: python3 ~/Desktop/joint_teleop_gui.py\n"
            "       (same ROS_DOMAIN_ID on T2/T3)",
            flush=True,
        )
    else:
        print(f"[INFO] joint json present: {args_cli.joint_file}", flush=True)

    step = 0
    teleop_warned = False

    with torch.inference_mode():
        print("[INFO] env.reset() ...", flush=True)
        env.reset()
        print(
            f"[INFO] settling at home ({args_cli.home_settle_steps} steps) ...",
            flush=True,
        )
        home = _settle_at_home(
            env,
            robot,
            raw,
            action_joint_names,
            args_cli.home_settle_steps,
            args_cli.physics_mode,
            args_cli.home_warmup_steps,
        )
        home_core = _home_core(raw.device)
        teleop._last = home_core[0].clone()
        action_joint_ids = [robot.joint_names.index(n) for n in action_joint_names]
        arm_sim_joint_ids = [robot.joint_names.index(f"joint_{i}") for i in range(1, 7)]
        last_applied_full: torch.Tensor | None = home.clone()
        _write_joint_feedback(robot, args_cli.feedback_file)

        with open(args_cli.csv, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            writer.writeheader()

            next_tick = time.monotonic()
            while simulation_app.is_running():
                json_fresh = (
                    os.path.isfile(args_cli.joint_file)
                    and time.time() - os.path.getmtime(args_cli.joint_file) <= args_cli.joint_stale_sec
                )
                core = teleop.read_action(
                    args_cli.num_envs,
                    robot=robot,
                    hold_current=False,
                    fallback=home_core,
                )
                if core is None:
                    core = home_core
                full = _expand_gripper_mimic(core)

                if (
                    not teleop_warned
                    and step > 0
                    and step % args_cli.log_interval == 0
                    and not json_fresh
                ):
                    print(
                        "[WARN] No fresh joint commands — holding home pose. "
                        "Run ros_joint_command_bridge.py + joint_teleop_gui.py."
                    )
                    teleop_warned = True

                reward, terminated, truncated, last_applied_full = _apply_teleop_physics(
                    env,
                    raw,
                    robot,
                    full,
                    last_applied_full,
                    json_fresh,
                    args_cli.physics_mode,
                    action_joint_ids,
                    arm_sim_joint_ids,
                )

                _write_joint_feedback(robot, args_cli.feedback_file)

                total_r = float(reward[0].item())
                weighted = _weighted_reward_terms(env)
                raw_terms = _raw_reward_terms(env)
                metrics = _scene_metrics(env)

                row = {
                    "step": step,
                    "total_reward": total_r,
                    **metrics,
                    **weighted,
                    **raw_terms,
                }
                writer.writerow(row)

                if step % args_cli.log_interval == 0:
                    print(
                        f"\n--- step {step} | total_r={total_r:+.4f} | "
                        f"drawer={metrics['drawer_joint']:+.4f} m | "
                        f"ee-handle={metrics['ee_handle_dist_m']:.3f} m ---"
                    )
                    for name in [
                        "approach_ee_handle",
                        "align_grasp_around_handle",
                        "grasp_handle",
                        "open_drawer_bonus",
                        "multi_stage_open_drawer",
                    ]:
                        w = weighted.get(name, 0.0)
                        r = raw_terms.get(f"raw_{name}", 0.0)
                        print(f"  {name:28s}  weighted={w:+.5f}  raw={r:+.5f}")

                step += 1
                if terminated.any() or truncated.any():
                    print(f"[INFO] Episode ended at step {step} (success or timeout). Resetting sim.")
                    env.reset()
                    home = _settle_at_home(
                        env,
                        robot,
                        raw,
                        action_joint_names,
                        args_cli.home_settle_steps,
                        args_cli.physics_mode,
                        args_cli.home_warmup_steps,
                    )
                    home_core = _home_core(raw.device)
                    teleop._last = home_core[0].clone()
                    last_applied_full = home.clone()
                    _write_joint_feedback(robot, args_cli.feedback_file)
                    teleop_warned = False
                    next_tick = time.monotonic()

                next_tick += raw.step_dt
                wait_s = next_tick - time.monotonic()
                if wait_s > 0:
                    time.sleep(wait_s)
                elif wait_s < -raw.step_dt:
                    next_tick = time.monotonic()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
