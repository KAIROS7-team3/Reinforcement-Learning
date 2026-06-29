#!/usr/bin/env python3
"""Record SO ARM teleop demos to HDF5 for BC warm-start (trajectory sketch mode).

Default ``--teleop_physics sketch``: kinematic joint teleport + grasp-assist object
follow — **no PhysX grasp**. Records 7D actions (joint_1..6 + rh_r1), same as PPO env.
Real contact/grasp is learned later via PPO fine-tune, not from these demos.

**Isaac Lab ≠ leader_to_isaac.md standalone Isaac Sim**

``leader_to_isaac.md`` T1 is Isaac Sim GUI + manual Action Graph on ``/World/e0509``.
This script runs **Isaac Lab** — Action Graph ArticulationController does not work here
(cuda tensor → numpy error). Use the **4-terminal** workflow below instead.

Four-terminal workflow (Isaac Lab demo collection)
--------------------------------------------------
T1 — Demo recorder (Isaac Lab, Python 3.11):

    cd /home/user/Reinforcement-Learning
    ../IsaacLab/isaaclab.sh -p scripts/collect_demos_teleop.py \\
        --task Isaac-ReturnTool-Teacher-Demo-v0 \\
        --dataset ./data/demos/return_tool/dataset.hdf5 \\
        --num_demos 5

T2 — SO ARM USB leader (system Python 3.10 + ROS):

    export ROS_DOMAIN_ID=71
    USB_PORT=/dev/ttyACM0 LEADER_DOF=7 \\
        ~/doosan-lerobot-stack/so-doosan-teleoperation-ver3/run_leader_usb.sh

T3 — Leader → isaac/joint_command (same as leader_to_isaac.md):

    export ROS_DOMAIN_ID=71
    source /opt/ros/humble/setup.bash
    python3 ~/doosan-lerobot-stack/leader_to_isaac.py

T4 — ROS → JSON bridge (required for Isaac Lab; system Python 3.10):

    export ROS_DOMAIN_ID=71
    source /opt/ros/humble/setup.bash
    python3 scripts/ros_joint_command_bridge.py

Start order: T2 → T3 → T4 → T1.

Touch ``/tmp/demo_reset.flag`` to reset the episode.
Successful demos (tool within 5 cm of place target) are exported automatically.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO ARM teleop demo collector → HDF5.")
parser.add_argument("--task", type=str, default="Isaac-ReturnTool-Teacher-Demo-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--dataset", type=str, default="./data/demos/return_tool/dataset.hdf5")
parser.add_argument("--num_demos", type=int, default=5)
parser.add_argument("--num_success_steps", type=int, default=10)
parser.add_argument(
    "--post_success_delay_sec",
    type=float,
    default=3.0,
    help="After auto_success export: hide cube, wait this long (move leader to home), "
    "then respawn cube at staging. 0 = immediate full reset (legacy).",
)
parser.add_argument(
    "--auto_success",
    action="store_true",
    default=False,
    help="Auto-export HDF5 when demo_place_success holds for --num_success_steps "
    "(gripper open + tool in drawer XY/Z band + low velocity). "
    "Default off — use /tmp/demo_reset.flag manually.",
)
parser.add_argument(
    "--demo_success_gripper_rad",
    type=float,
    default=0.35,
    help="demo_place_success: rh_r1 <= this (rad) counts as gripper open (~20° default).",
)
parser.add_argument(
    "--demo_success_z_band",
    type=float,
    default=0.04,
    help="demo_place_success: |tool_root_z - floor_z| in drawer frame (m).",
)
parser.add_argument(
    "--demo_success_tool_xy_margin",
    type=float,
    default=None,
    help="Inset from drawer floor walls for tool root XY (m). Default: cube half-edge.",
)
parser.add_argument(
    "--demo_success_max_vel",
    type=float,
    default=0.10,
    help="demo_place_success: max tool linear speed (m/s).",
)
parser.add_argument(
    "--teleop_source",
    type=str,
    default="json",
    choices=("sim", "json", "ros"),
    help="json = T4 JSON file (default for Isaac Lab); sim/ros = legacy Action Graph paths.",
)
parser.add_argument(
    "--joint_file",
    type=str,
    default="/tmp/isaac_teleop_joints.json",
    help="JSON joint targets (--teleop_source json only).",
)
parser.add_argument(
    "--feedback_file",
    type=str,
    default="/tmp/isaac_sim_joint_states.json",
    help="Sim joint feedback JSON (--teleop_source json only).",
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--joint_stale_sec", type=float, default=2.0)
parser.add_argument(
    "--home_settle_steps",
    type=int,
    default=0,
    help="Legacy PhysX warm-up steps after home freeze (0 recommended — use freeze teleop).",
)
parser.add_argument(
    "--debug_interval",
    type=int,
    default=120,
    help="Print teleop vs sim joint_1 every N steps (0=off). Default 120 ≈ 2 s at 60 Hz.",
)
parser.add_argument(
    "--joint_deadband_rad",
    type=float,
    default=0.002,
    help="Ignore arm joint teleop changes smaller than this (rad) — reduces idle jitter.",
)
parser.add_argument(
    "--gripper_deadband_rad",
    type=float,
    default=0.008,
    help="Ignore rh_r1 teleop changes smaller than this (rad) — reduces finger wobble.",
)
parser.add_argument(
    "--gripper_ramp_rad",
    type=float,
    default=0.0,
    help="Max rh_r1 change per control step (rad). 0 = instant close/open (sketch default).",
)
parser.add_argument(
    "--tool_asset",
    type=str,
    default="cube",
    choices=("screwdriver", "cube"),
    help="Staging object: cube (default, sketch demos) or screwdriver.",
)
parser.add_argument(
    "--tool_cube_size",
    type=float,
    default=0.05,
    help="Cube edge length (m) when --tool_asset cube.",
)
parser.add_argument(
    "--teleop_mode",
    type=str,
    default="json",
    choices=("json", "kinematic", "action_graph"),
    help="json = T4 JSON teleop (default); kinematic = legacy freeze; action_graph = deprecated.",
)
parser.add_argument(
    "--teleop_physics",
    type=str,
    default="sketch",
    choices=("sketch", "legacy"),
    help="sketch = kinematic JSON follow + grasp assist, no PhysX (BC default). "
    "legacy = old sim.step kinematic path.",
)
parser.add_argument("--ros_domain_id", type=int, default=71, help="ROS domain (T4 bridge / legacy action_graph).")
parser.add_argument(
    "--no_grasp_assist",
    action="store_true",
    help="Disable demo grasp weld (tool follows EE when gripper closed near object).",
)
parser.add_argument(
    "--grasp_dist_m",
    type=float,
    default=0.10,
    help="Max finger-midpoint–tool distance (m) to engage grasp assist.",
)
parser.add_argument(
    "--grasp_close_rad",
    type=float,
    default=0.35,
    help="Min rh_r1 (rad) to engage grasp assist (~20 deg default).",
)
parser.add_argument(
    "--grasp_release_rad",
    type=float,
    default=0.20,
    help="Max rh_r1 (rad) to release grasp assist (~11 deg default).",
)
parser.add_argument(
    "--grasp_hold_frames",
    type=int,
    default=1,
    help="Consecutive near+closed frames before weld engages.",
)
parser.add_argument(
    "--no_table_lock",
    action="store_true",
    help="Disable staging table-lock (cube can slide while closing gripper).",
)
parser.add_argument(
    "--grasp_snap_inward_m",
    type=float,
    default=0.03,
    help="On engage, nudge cube toward finger midpoint (m) — tighter pinch feel.",
)
parser.add_argument(
    "--place_snap",
    action="store_true",
    default=False,
    help="On gripper release near drawer, snap cube XYZ to place_target (marker 2). "
    "Default off — XY stays where you release; Z still settles to drawer floor.",
)
parser.add_argument(
    "--place_radius_m",
    type=float,
    default=0.12,
    help="XY radius (m) around place_target for --place_snap (ignored otherwise).",
)
parser.add_argument(
    "--place_z_band_m",
    type=float,
    default=0.10,
    help="Max |z - place marker| (m) to snap cube on drawer release.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Backward-compatible aliases.
if args_cli.teleop_physics in ("step", "kinematic"):
    args_cli.teleop_physics = "sketch" if args_cli.teleop_physics == "step" else "legacy"

if args_cli.teleop_mode == "action_graph":
    print(
        "[WARN] --teleop_mode action_graph is broken in Isaac Lab "
        "(ArticulationController cuda tensor error).\n"
        "  Falling back to json PD teleop. Run T4: scripts/ros_joint_command_bridge.py",
        flush=True,
    )
    args_cli.teleop_mode = "json"
    if args_cli.teleop_source == "sim":
        args_cli.teleop_source = "json"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import tool_transfer_bot  # noqa: F401, E402
from tool_transfer_bot.tasks.mdp.grasp_assist import GraspAssist, write_tool_root as _write_tool_root  # noqa: E402

from tool_transfer_bot.assets.environments import (  # noqa: E402
    RETURN_TOOL_DRAWER_FLOOR_X_MAX,
    RETURN_TOOL_DRAWER_FLOOR_X_MIN,
    RETURN_TOOL_DRAWER_FLOOR_Y_MAX,
    RETURN_TOOL_DRAWER_FLOOR_Y_MIN,
    RETURN_TOOL_DRAWER_TOOL_XY_MARGIN,
    RETURN_TOOL_STAGING_POS,
    RETURN_TOOL_STAGING_ROT,
    return_tool_home_joint_pos_rad,
)
from tool_transfer_bot.tasks.base_env_cfg import JointActionsCfg  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

JSON_JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)] + ["rh_r1"]
FEEDBACK_JOINTS = JSON_JOINT_NAMES + ["rh_l1"]
GRIPPER_ACTION_INDEX = 6
ARM_ACTION_DIM = 6
# rh_r1 USD limit 0..63 deg
GRIPPER_MAX_RAD = math.radians(63.0)


class JointFileTeleop:
    def __init__(self, path: str, stale_sec: float, device: torch.device):
        self.path = path
        self.stale_sec = stale_sec
        self.device = device
        self._last: torch.Tensor | None = None
        self._last_applied: torch.Tensor | None = None
        self._last_warn = 0.0

    def _json_age_sec(self) -> float | None:
        if not os.path.isfile(self.path):
            return None
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if "stamp_sec" in data:
                return time.time() - float(data["stamp_sec"])
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
        return time.time() - os.path.getmtime(self.path)

    def is_fresh(self) -> bool:
        """True when T4 is actively writing the JSON file (mtime-based, same as teleop_reward_monitor)."""
        return _json_file_live(self.path, self.stale_sec)

    def warn_if_stale(self) -> None:
        if self.is_fresh():
            return
        now = time.time()
        if now - self._last_warn < 3.0:
            return
        self._last_warn = now
        if not os.path.isfile(self.path):
            print(
                f"[WARN] No teleop JSON at {self.path}. "
                "Run T2 leader USB + T3 leader_to_isaac + T4 ros_joint_command_bridge.",
                flush=True,
            )
        else:
            age = self._json_age_sec()
            age_s = f"{age:.1f}" if age is not None else "?"
            print(
                f"[WARN] Teleop JSON stale ({age_s}s). Check T2–T4 and ROS_DOMAIN_ID=71.",
                flush=True,
            )

    def read(self, fallback: torch.Tensor) -> torch.Tensor:
        """Always load latest JSON — stale only affects is_fresh() warnings."""
        if not os.path.isfile(self.path):
            return fallback
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return fallback if self._last is None else self._last
        vals = []
        for name in JSON_JOINT_NAMES:
            if name not in data:
                return fallback if self._last is None else self._last
            vals.append(float(data[name]))
        self._last = torch.tensor(vals, device=self.device, dtype=torch.float32)
        return self._last

    def apply_stabilized(
        self,
        action: torch.Tensor,
        arm_deadband: float,
        gripper_deadband: float,
    ) -> torch.Tensor:
        if self._last_applied is None:
            return action
        out = action.clone()
        if arm_deadband > 0.0:
            arm_delta = torch.max(torch.abs(out[:ARM_ACTION_DIM] - self._last_applied[:ARM_ACTION_DIM])).item()
            if arm_delta <= arm_deadband:
                out[:ARM_ACTION_DIM] = self._last_applied[:ARM_ACTION_DIM]
        if gripper_deadband > 0.0:
            grip_delta = abs(float(out[GRIPPER_ACTION_INDEX] - self._last_applied[GRIPPER_ACTION_INDEX]))
            if grip_delta <= gripper_deadband:
                out[GRIPPER_ACTION_INDEX] = self._last_applied[GRIPPER_ACTION_INDEX]
        return out

    def apply_deadband(self, action: torch.Tensor, deadband: float) -> torch.Tensor:
        return self.apply_stabilized(action, deadband, deadband)


def _teleop_changed(
    prev: torch.Tensor | None,
    cur: torch.Tensor,
    arm_tol: float,
    grip_tol: float,
) -> bool:
    if prev is None:
        return True
    arm_delta = torch.max(torch.abs(cur[:ARM_ACTION_DIM] - prev[:ARM_ACTION_DIM])).item()
    grip_delta = abs(float(cur[GRIPPER_ACTION_INDEX] - prev[GRIPPER_ACTION_INDEX]))
    arm_changed = arm_tol <= 0.0 or arm_delta > arm_tol
    grip_changed = grip_tol <= 0.0 or grip_delta > grip_tol
    return arm_changed or grip_changed


def _idle_record_step(raw, action: torch.Tensor) -> None:
    """Advance recorder/terminations without PhysX (avoids PD/contact fight while holding pose)."""
    raw.action_manager.process_action(action.to(raw.device))
    raw.recorder_manager.record_pre_step()
    raw.episode_length_buf += 1
    raw.common_step_counter += 1
    raw.reset_buf = raw.termination_manager.compute()
    raw.reset_terminated = raw.termination_manager.terminated
    raw.reset_time_outs = raw.termination_manager.time_outs
    raw.reward_buf = raw.reward_manager.compute(dt=raw.step_dt)
    if len(raw.recorder_manager.active_terms) > 0:
        raw.obs_buf = raw.observation_manager.compute()
        raw.recorder_manager.record_post_step()
    reset_env_ids = raw.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if len(reset_env_ids) > 0:
        raw.recorder_manager.record_pre_reset(reset_env_ids)
        raw._reset_idx(reset_env_ids)
        raw.recorder_manager.record_post_reset(reset_env_ids)


def _disable_embedded_action_graph() -> None:
    """Deactivate legacy ActionGraph embedded under robot USD (conflicts with Isaac Lab)."""
    try:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if prim.GetName() == "ActionGraph" and "/real_to_sim/" in path:
                prim.SetActive(False)
                print(f"[INFO] Deactivated legacy ActionGraph: {path}", flush=True)
    except Exception as exc:
        print(f"[WARN] ActionGraph disable skipped: {exc}", flush=True)


def _ensure_timeline_playing(raw) -> None:
    """GUI mode: env.step blocks until timeline Play — auto-start if paused/stopped."""
    if not raw.sim.has_gui():
        return
    import omni.timeline

    tl = omni.timeline.get_timeline_interface()
    if not tl.is_playing():
        print("[INFO] Timeline not playing — clicking Play for teleop.", flush=True)
        tl.play()


def _home_action_tensor(robot) -> torch.Tensor:
    home = return_tool_home_joint_pos_rad()
    vals = [float(home[name]) for name in JSON_JOINT_NAMES]
    return torch.tensor(vals, device=robot.device, dtype=torch.float32)


def _arm_joint_ids(robot) -> list[int]:
    return [robot.joint_names.index(f"joint_{i}") for i in range(1, 7)]


def _expand_gripper_mimic(core: torch.Tensor) -> torch.Tensor:
    """7D (arm + rh_r1) → 10D — rh_r2/rh_l1/rh_l2 = rh_r1 (teleop_reward_monitor / OpenDrawer)."""
    if core.dim() == 1:
        r1 = core[-1:]
        return torch.cat([core, r1, r1, r1], dim=0)
    r1 = core[..., -1:]
    return torch.cat([core, r1, r1, r1], dim=-1)


def _demo_write_joint_ids(robot) -> list[int]:
    """All joints written during sketch teleop (arm + 4 gripper DOFs)."""
    names = [f"joint_{i}" for i in range(1, 7)] + ["rh_r1", "rh_r2", "rh_l1", "rh_l2"]
    return [robot.joint_names.index(n) for n in names if n in robot.joint_names]


def _sketch_viewport_step(raw) -> None:
    """sim.step for viewport (use_fabric=False); skipped when fabric is on."""
    if not raw.sim.cfg.use_fabric:
        raw.sim.step(render=raw.sim.has_gui() or raw.sim.has_rtx_sensors())


def _sketch_sync_joints(
    robot,
    raw,
    target: torch.Tensor,
    write_joint_ids: list[int],
    grasp_assist: GraspAssist | None = None,
    gripper_rad: float | None = None,
) -> None:
    """Kinematic joint teleport + optional tool assist; sim.step then re-lock (viewport + anti-drift)."""
    if target.dim() == 1:
        target = target.unsqueeze(0)
    zero_vel = torch.zeros_like(target)
    robot.write_joint_state_to_sim(target, zero_vel, joint_ids=write_joint_ids)
    robot.set_joint_position_target(target, joint_ids=write_joint_ids)
    raw.scene.write_data_to_sim()
    if grasp_assist is not None and gripper_rad is not None:
        grasp_assist.update(robot, raw, gripper_rad)
    _sketch_viewport_step(raw)
    robot.write_joint_state_to_sim(target, zero_vel, joint_ids=write_joint_ids)
    robot.set_joint_position_target(target, joint_ids=write_joint_ids)
    if grasp_assist is not None:
        grasp_assist.reassert_tool(robot, raw)
    raw.scene.write_data_to_sim()
    if raw.sim.has_gui() or raw.sim.has_rtx_sensors():
        raw.sim.render()
    raw.scene.update(dt=raw.physics_dt)


def _freeze_pose(
    robot,
    raw,
    target: torch.Tensor,
    write_joint_ids: list[int],
    *,
    sketch: bool = False,
) -> None:
    """Teleport joints. sketch: sim.step for viewport then re-lock (no env.step PD)."""
    if sketch:
        _sketch_sync_joints(robot, raw, target, write_joint_ids)
        return
    if target.dim() == 1:
        target = target.unsqueeze(0)
    zero_vel = torch.zeros_like(target)
    robot.write_joint_state_to_sim(target, zero_vel, joint_ids=write_joint_ids)
    robot.set_joint_position_target(target, joint_ids=write_joint_ids)
    raw.scene.write_data_to_sim()
    _refresh_visual(raw)


def _apply_sketch_teleop(
    robot,
    raw,
    core7: torch.Tensor,
    write_joint_ids: list[int],
    grasp_assist: GraspAssist | None = None,
) -> torch.Tensor:
    """Snap arm + gripper from JSON; returns 7D action for HDF5 (matches PPO)."""
    full = _expand_gripper_mimic(core7)
    grip = float(core7[GRIPPER_ACTION_INDEX].item())
    _sketch_sync_joints(robot, raw, full, write_joint_ids, grasp_assist, grip)
    return core7.unsqueeze(0)


def _action_joint_ids(robot, raw) -> list[int]:
    """Action manager joint indices (7D legacy path)."""
    action_term = raw.action_manager.get_term("joint_action")
    return [robot.joint_names.index(n) for n in action_term._joint_names]


def _apply_kinematic_arm_only(robot, raw, core7: torch.Tensor) -> None:
    """Track J1–J6 from teleop; gripper left to PhysX for grasp contact."""
    arm_ids = _arm_joint_ids(robot)
    arm_pos = core7[:ARM_ACTION_DIM].unsqueeze(0)
    arm_vel = torch.zeros_like(arm_pos)
    robot.write_joint_state_to_sim(arm_pos, arm_vel, joint_ids=arm_ids)
    robot.set_joint_position_target(arm_pos, joint_ids=arm_ids)
    raw.scene.write_data_to_sim()


def _clamp_gripper_rad(rad: float) -> float:
    return max(0.0, min(rad, GRIPPER_MAX_RAD))


def _rate_limit_gripper(prev: float | None, target: float, max_step: float) -> float:
    """Gradual gripper close/open — ported from doosan_e0509_RL gui_with_gripper (0.05 rad/tick)."""
    target = _clamp_gripper_rad(target)
    if prev is None or max_step <= 0.0:
        return target
    delta = target - prev
    if abs(delta) <= max_step:
        return target
    return _clamp_gripper_rad(prev + math.copysign(max_step, delta))


def _write_rh_r1_kinematic(robot, raw, gripper_rad: float) -> None:
    """Kinematic rh_r1 only (doosan_e0509_RL: scalar → single master DOF; mimics via PhysX)."""
    if "rh_r1" not in robot.joint_names:
        return
    gripper_rad = _clamp_gripper_rad(gripper_rad)
    r1_id = robot.joint_names.index("rh_r1")
    pos = torch.tensor([[gripper_rad]], device=robot.device, dtype=torch.float32)
    vel = torch.zeros_like(pos)
    robot.write_joint_state_to_sim(pos, vel, joint_ids=[r1_id])
    robot.set_joint_position_target(pos, joint_ids=[r1_id])
    raw.scene.write_data_to_sim()


def _refresh_visual(raw, *, physics: bool = True) -> None:
    """Push joint state to viewport. Skip physics step if env.step already ran this frame."""
    if physics and not raw.sim.cfg.use_fabric:
        raw.sim.step(render=raw.sim.has_gui() or raw.sim.has_rtx_sensors())
    elif raw.sim.has_gui() or raw.sim.has_rtx_sensors():
        raw.sim.render()
    raw.scene.update(dt=raw.physics_dt)


def _apply_kinematic_teleop(robot, raw, core7: torch.Tensor) -> None:
    """Teleop: arm + rh_r1 kinematic; mimics follow PhysX mimic (never write rh_l1/rh_l2/rh_r2)."""
    gripper_rad = _clamp_gripper_rad(float(core7[GRIPPER_ACTION_INDEX].item()))
    _apply_kinematic_arm_only(robot, raw, core7)
    _write_rh_r1_kinematic(robot, raw, gripper_rad)
    # use_fabric=False: viewport needs sim.step to show kinematic joint teleports.
    _refresh_visual(raw, physics=True)
    # PhysX step can drift kinematic joints — re-lock arm + rh_r1 only (never mimic fingers).
    _apply_kinematic_arm_only(robot, raw, core7)
    _write_rh_r1_kinematic(robot, raw, gripper_rad)
    if raw.sim.has_gui() or raw.sim.has_rtx_sensors():
        raw.sim.render()
    raw.scene.update(dt=raw.physics_dt)


def _json_file_live(path: str, stale_sec: float) -> bool:
    """True when joint JSON exists and was written recently (T4 bridge active)."""
    if not os.path.isfile(path):
        return False
    try:
        return (time.time() - os.path.getmtime(path)) <= stale_sec
    except OSError:
        return False


def _apply_sketch_teleop_frame(
    robot,
    raw,
    raw_action: torch.Tensor,
    teleop: JointFileTeleop,
    write_joint_ids: list[int],
    grasp_assist: GraspAssist | None = None,
    *,
    record: bool = True,
) -> torch.Tensor:
    """Sketch BC demo frame: kinematic JSON follow + optional grasp assist; record 7D."""
    if _json_file_live(teleop.path, teleop.stale_sec):
        core7 = raw_action
        teleop._last_applied = raw_action.clone()
    else:
        core7 = teleop._last_applied if teleop._last_applied is not None else raw_action
    action = _apply_sketch_teleop(robot, raw, core7, write_joint_ids, grasp_assist)
    if record:
        _idle_record_step(raw, action)
    return action


def _ee_body_name(robot) -> str:
    if "rh_p12_rn_base" in robot.body_names:
        return "rh_p12_rn_base"
    return "link_6"


def _ee_body_index(robot) -> int:
    return robot.body_names.index(_ee_body_name(robot))


def _ee_tool_distance_m(raw) -> float:
    robot = raw.scene["robot"]
    tool = raw.scene["tool"]
    ee_idx = _ee_body_index(robot)
    ee = robot.data.body_pos_w[0, ee_idx]
    return float(torch.norm(ee - tool.data.root_pos_w[0]).item())


def _tool_staging_pose_w() -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    if args_cli.tool_asset == "cube":
        edge = args_cli.tool_cube_size
        pos = (RETURN_TOOL_STAGING_POS[0], RETURN_TOOL_STAGING_POS[1], edge * 0.5)
    else:
        pos = RETURN_TOOL_STAGING_POS
    return pos, RETURN_TOOL_STAGING_ROT


_TOOL_HIDE_POS_W = (-100.0, -100.0, -10.0)


def _tool_visual_prims(tool):
    cache_attr = "_teleop_vis_prims"
    if not hasattr(tool, cache_attr):
        import isaaclab.sim as sim_utils

        setattr(tool, cache_attr, sim_utils.find_matching_prims(tool.cfg.prim_path))
    return getattr(tool, cache_attr)


def _set_tool_visible(raw, visible: bool) -> None:
    import isaaclab.sim as sim_utils

    for prim in _tool_visual_prims(raw.scene["tool"]):
        sim_utils.set_prim_visibility(prim, visible)


def _hide_tool(raw) -> None:
    tool = raw.scene["tool"]
    _set_tool_visible(raw, False)
    hide_pos = torch.tensor([list(_TOOL_HIDE_POS_W)], device=raw.device, dtype=torch.float32)
    _write_tool_root(raw, tool, hide_pos, tool.data.root_quat_w.clone())


def _show_tool(raw) -> None:
    _set_tool_visible(raw, True)


def _respawn_tool_at_staging(raw) -> None:
    tool = raw.scene["tool"]
    pos, rot = _tool_staging_pose_w()
    pos_w = torch.tensor([pos], device=raw.device, dtype=torch.float32)
    quat_w = torch.tensor([rot], device=raw.device, dtype=torch.float32)
    _write_tool_root(raw, tool, pos_w, quat_w)


def _begin_success_intermission(raw, grasp_assist: GraspAssist | None) -> None:
    env_recorder = raw.recorder_manager
    env_recorder.reset()
    if grasp_assist is not None:
        grasp_assist.reset()
    _hide_tool(raw)
    delay = args_cli.post_success_delay_sec
    print(
        f"[INFO] Cube hidden — move leader to home (~{delay:.0f}s until respawn at staging).",
        flush=True,
    )


def _finish_success_intermission(raw, robot, grasp_assist: GraspAssist | None) -> None:
    _respawn_tool_at_staging(raw)
    _show_tool(raw)
    if grasp_assist is not None:
        grasp_assist.reset(robot, raw)
    print("[INFO] Cube respawned at staging — ready for next demo.", flush=True)


def _setup_teleop_backend(robot, raw, mode: str, ros_domain_id: int) -> None:
    _disable_embedded_action_graph()
    if mode == "json":
        _ensure_timeline_playing(raw)
        return
    if mode == "action_graph":
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from isaaclab_ros_action_graph import (  # noqa: WPS433
            resolve_robot_controller_prim,
            setup_ros2_joint_teleop_graph,
        )

        target = resolve_robot_controller_prim(robot)
        try:
            # Never use ROS2PublishJointState in Isaac Lab — PhysX device -1 errors.
            # Subscribe + ArticulationController only. Domain ID is set on ROS2Context.
            setup_ros2_joint_teleop_graph(
                target,
                ros_domain_id=ros_domain_id,
                publish_joint_states=False,
                force=True,
            )
            print(f"[INFO] Action Graph ROS domain_id={ros_domain_id}", flush=True)
        except Exception as exc:
            print(
                f"[ERROR] Action Graph teleop setup failed: {exc}\n"
                "  Retry with legacy mode: --teleop_mode kinematic",
                flush=True,
            )
            raise
    _ensure_timeline_playing(raw)


def _teardown_teleop_backend(raw) -> None:
    """Stop playback before sim teardown — avoids Action Graph PhysX errors on exit."""
    try:
        import omni.timeline

        tl = omni.timeline.get_timeline_interface()
        if tl.is_playing():
            tl.stop()
    except Exception:
        pass
    try:
        import omni.graph.core as og
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is not None and stage.GetPrimAtPath("/ActionGraph").IsValid():
            og.Controller.destroy_graph("/ActionGraph")
    except Exception:
        pass


def _teleop_gripper_actuator():
    from isaaclab.actuators import ImplicitActuatorCfg
    from tool_transfer_bot.assets.doosan_e0509 import (
        GRIPPER_DAMPING_RL,
        GRIPPER_EFFORT_LIMIT_SIM,
        GRIPPER_STIFFNESS_RL,
    )

    return ImplicitActuatorCfg(
        joint_names_expr=["rh_r1"],
        effort_limit_sim=GRIPPER_EFFORT_LIMIT_SIM,
        stiffness=GRIPPER_STIFFNESS_RL,
        damping=GRIPPER_DAMPING_RL,
    )


def _settle_at_home(
    env,
    robot,
    raw,
    home: torch.Tensor,
    steps: int,
    *,
    use_sketch_teleop: bool,
    direct_teleop: bool,
    write_joint_ids: list[int] | None = None,
) -> None:
    if steps <= 0:
        return
    with torch.inference_mode():
        if use_sketch_teleop and write_joint_ids is not None:
            home10 = _expand_gripper_mimic(home)
            for _ in range(steps):
                _freeze_pose(robot, raw, home10, write_joint_ids, sketch=True)
        elif direct_teleop:
            for _ in range(steps):
                _apply_kinematic_teleop(robot, raw, home)
        else:
            action = home.unsqueeze(0)
            for _ in range(steps):
                env.step(action)


def _build_env_cfg():
    task = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = task

    if hasattr(env_cfg.terminations, "task_success"):
        env_cfg.terminations.task_success = None

    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = True

    out_dir = os.path.dirname(os.path.abspath(args_cli.dataset))
    out_name = os.path.basename(args_cli.dataset)
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = out_dir
    env_cfg.recorders.dataset_filename = out_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False
    elif args_cli.teleop_mode == "action_graph":
        env_cfg.sim.use_fabric = False

    if args_cli.teleop_mode in ("kinematic", "json"):
        import isaaclab.sim as sim_utils
        from tool_transfer_bot.assets.doosan_e0509 import (
            DOOSAN_E0509_TELEOP_ACTUATORS,
            MAX_DEPENETRATION_VELOCITY,
        )

        teleop_actuators = dict(DOOSAN_E0509_TELEOP_ACTUATORS)
        env_cfg.sim.use_fabric = False
        env_cfg.events.sync_gripper_mimic = None

        if args_cli.teleop_physics == "sketch":
            env_cfg.actions = JointActionsCfg()
        else:
            teleop_actuators["gripper"] = _teleop_gripper_actuator()
            env_cfg.actions = JointActionsCfg()

        env_cfg.scene.robot = env_cfg.scene.robot.replace(
            actuators=teleop_actuators,
            spawn=env_cfg.scene.robot.spawn.replace(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
                ),
            ),
        )

    if args_cli.tool_asset == "cube":
        from tool_transfer_bot.assets.environments import return_tool_staging_cube_cfg

        env_cfg.scene.tool = return_tool_staging_cube_cfg(
            prim_path="{ENV_REGEX_NS}/tool",
            edge=args_cli.tool_cube_size,
        )
    elif args_cli.tool_asset == "screwdriver":
        import isaaclab.sim as sim_utils
        from tool_transfer_bot.assets import TOOL_CFGS
        from tool_transfer_bot.assets.doosan_e0509 import MAX_DEPENETRATION_VELOCITY

        base_spawn = TOOL_CFGS["Screw_Driver"].spawn
        env_cfg.scene.tool = TOOL_CFGS["Screw_Driver"].replace(
            prim_path="{ENV_REGEX_NS}/tool",
            spawn=base_spawn.replace(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=2.5,
                    dynamic_friction=2.0,
                    restitution=0.0,
                ),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=min(0.05, MAX_DEPENETRATION_VELOCITY),
                    linear_damping=0.55,
                    angular_damping=0.75,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.004,
                    rest_offset=0.001,
                ),
            ),
        )
        env_cfg.scene.tool.init_state.pos = RETURN_TOOL_STAGING_POS
        env_cfg.scene.tool.init_state.rot = RETURN_TOOL_STAGING_ROT

    return env_cfg


def _demo_success_tool_xy_margin() -> float:
    if args_cli.demo_success_tool_xy_margin is not None:
        return float(args_cli.demo_success_tool_xy_margin)
    if args_cli.tool_asset == "cube":
        return args_cli.tool_cube_size * 0.5
    return RETURN_TOOL_DRAWER_TOOL_XY_MARGIN


def _make_demo_success_term():
    from types import SimpleNamespace

    from tool_transfer_bot.tasks.mdp.terminations import demo_place_success

    z_center_local = None
    if args_cli.tool_asset == "cube":
        z_center_local = args_cli.tool_cube_size * 0.5

    return SimpleNamespace(
        func=demo_place_success,
        params={
            "gripper_open_rad": args_cli.demo_success_gripper_rad,
            "max_linear_vel": args_cli.demo_success_max_vel,
            "tool_xy_margin": _demo_success_tool_xy_margin(),
            "z_band": args_cli.demo_success_z_band,
            "z_center_local": z_center_local,
        },
    )


def _format_demo_success_debug(env, success_term) -> str:
    from tool_transfer_bot.tasks.mdp.terminations import demo_place_success_parts

    parts = demo_place_success_parts(env.unwrapped, **success_term.params)
    ok = lambda name: "OK" if bool(parts[name][0].item()) else "FAIL"
    return (
        f"success grip={ok('gripper_open')}({float(parts['grip_r1'][0]):.3f}<="
        f"{success_term.params['gripper_open_rad']:.3f}) "
        f"xy={ok('in_drawer_xy')}"
        f"(root x={float(parts['tool_local_x'][0]):.3f} y={float(parts['tool_local_y'][0]):.3f} "
        f"in [{parts['drawer_x_min']:.2f},{parts['drawer_x_max']:.2f}]"
        f"x[{parts['drawer_y_min']:.2f},{parts['drawer_y_max']:.2f}]) "
        f"markerΔ=({float(parts['dx_marker'][0]):.3f},{float(parts['dy_marker'][0]):.3f}) "
        f"z={ok('on_floor_z')}(dz={float(parts['dz'][0]):.3f}<="
        f"{success_term.params['z_band']:.3f}) "
        f"vel={ok('slow')}({float(parts['speed'][0]):.3f}m/s)"
    )


def _write_joint_feedback(robot, path: str) -> None:
    data: dict[str, float] = {"stamp_sec": time.time()}
    for name in FEEDBACK_JOINTS:
        if name in robot.joint_names:
            idx = robot.joint_names.index(name)
            data[name] = float(robot.data.joint_pos[0, idx].item())
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _log_actuator_stiffness(robot) -> None:
    try:
        for name, actuator in robot.actuators.items():
            cfg = actuator.cfg
            print(
                f"[INFO] actuator {name}: stiffness={cfg.stiffness} damping={cfg.damping}",
                flush=True,
            )
    except Exception as exc:
        print(f"[WARN] actuator log skipped: {exc}", flush=True)


def _current_joint_action(robot) -> torch.Tensor:
    vals = []
    for name in JSON_JOINT_NAMES:
        idx = robot.joint_names.index(name)
        vals.append(float(robot.data.joint_pos[0, idx].item()))
    return torch.tensor(vals, device=robot.device, dtype=torch.float32)


def _process_success(env, success_term, success_count: int) -> tuple[int, bool]:
    if success_term is None:
        return success_count, False
    if bool(success_term.func(env.unwrapped, **success_term.params)[0]):
        success_count += 1
        if success_count >= args_cli.num_success_steps:
            env.unwrapped.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.unwrapped.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.unwrapped.device)
            )
            env.unwrapped.recorder_manager.export_episodes([0])
            print("[INFO] Success — demo exported (demo_place_success).", flush=True)
            return 0, True
    else:
        success_count = 0
    return success_count, False


def _reset_episode_immediate(
    env,
    robot,
    raw,
    home: torch.Tensor,
    grasp_assist: GraspAssist | None,
    *,
    use_sketch_teleop: bool,
    direct_teleop: bool,
    write_joint_ids: list[int],
) -> tuple[torch.Tensor, object]:
    """Full sim reset (manual flag or legacy instant success reset)."""
    raw.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    _setup_teleop_backend(robot, raw, args_cli.teleop_mode, args_cli.ros_domain_id)
    teleop = _make_teleop(robot, home, raw.device)
    hold = home.clone()
    if grasp_assist is not None:
        grasp_assist.reset(robot, raw)
    teleop._last = hold.clone()
    teleop._last_applied = hold.clone()
    teleop._applied_gripper_rad = float(home[GRIPPER_ACTION_INDEX].item())
    _respawn_tool_at_staging(raw)
    _show_tool(raw)
    if use_sketch_teleop and write_joint_ids:
        home10 = _expand_gripper_mimic(home)
        _sketch_sync_joints(robot, raw, home10, write_joint_ids, grasp_assist)
    _settle_at_home(
        env, robot, raw, home, args_cli.home_settle_steps,
        use_sketch_teleop=use_sketch_teleop,
        direct_teleop=direct_teleop and not use_sketch_teleop,
        write_joint_ids=write_joint_ids,
    )
    print(
        "[INFO] Scene reset: cube at staging. "
        "If the arm did not move, place the leader near home — sketch teleop follows the leader.",
        flush=True,
    )
    return hold, teleop


def _make_teleop(robot, home: torch.Tensor, device: torch.device):
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    if args_cli.teleop_source == "sim":
        from isaaclab_ros_teleop_io import ActionGraphSimTeleop  # noqa: WPS433

        return ActionGraphSimTeleop(robot, home, args_cli.joint_stale_sec)
    if args_cli.teleop_source == "ros":
        from isaaclab_ros_teleop_io import RosJointCommandTeleop  # noqa: WPS433

        return RosJointCommandTeleop(
            device,
            args_cli.joint_stale_sec,
            domain_id=args_cli.ros_domain_id,
        )
    return JointFileTeleop(args_cli.joint_file, args_cli.joint_stale_sec, device)


def main() -> None:
    env_cfg = _build_env_cfg()
    success_term = _make_demo_success_term() if args_cli.auto_success else None
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    raw = env
    robot = raw.scene["robot"]

    print(f"[INFO] Recording to {args_cli.dataset}", flush=True)
    print(f"[INFO] Target demos: {args_cli.num_demos}", flush=True)
    print(f"[INFO] teleop source: {args_cli.teleop_source}", flush=True)
    print("[INFO] Touch /tmp/demo_reset.flag to reset episode (or pass --auto_success).", flush=True)
    if args_cli.auto_success:
        print(
            "[INFO] auto_success ON: demo_place_success "
            f"(gripper<={math.degrees(args_cli.demo_success_gripper_rad):.0f}°, "
            f"drawer floor XY "
            f"x=[{RETURN_TOOL_DRAWER_FLOOR_X_MIN:.2f},{RETURN_TOOL_DRAWER_FLOOR_X_MAX:.2f}] "
            f"y=[{RETURN_TOOL_DRAWER_FLOOR_Y_MIN:.2f},{RETURN_TOOL_DRAWER_FLOOR_Y_MAX:.2f}] "
            f"margin={_demo_success_tool_xy_margin():.3f}m, "
            f"z_band={args_cli.demo_success_z_band:.2f}m, "
            f"vel<={args_cli.demo_success_max_vel:.2f}m/s) × "
            f"{args_cli.num_success_steps} frames; "
            f"post-success delay={args_cli.post_success_delay_sec:.1f}s",
            flush=True,
        )

    env.reset()
    use_direct_teleop = args_cli.teleop_mode in ("kinematic", "json")
    use_sketch_teleop = use_direct_teleop and args_cli.teleop_physics == "sketch"
    write_joint_ids = _demo_write_joint_ids(robot) if use_sketch_teleop else []
    _obs0 = raw.observation_manager.compute()["policy"]
    print(f"[INFO] policy obs_dim={_obs0.shape[-1]}", flush=True)
    _setup_teleop_backend(robot, raw, args_cli.teleop_mode, args_cli.ros_domain_id)
    home = _home_action_tensor(robot)
    teleop = _make_teleop(robot, home, raw.device)
    grasp_assist: GraspAssist | None = None
    if not args_cli.no_grasp_assist:
        grasp_assist = GraspAssist(
            dist_m=args_cli.grasp_dist_m,
            close_rad=args_cli.grasp_close_rad,
            release_rad=args_cli.grasp_release_rad,
            hold_frames=args_cli.grasp_hold_frames,
            snap_inward_m=args_cli.grasp_snap_inward_m,
            table_lock=not args_cli.no_table_lock,
            place_radius_m=args_cli.place_radius_m,
            place_z_band_m=args_cli.place_z_band_m,
            place_snap=args_cli.place_snap,
        )
        lock_note = "table-lock ON" if not args_cli.no_table_lock else "table-lock OFF"
        print(
            f"[INFO] grasp assist ON ({lock_note}): dist<{args_cli.grasp_dist_m}m, "
            f"close>{math.degrees(args_cli.grasp_close_rad):.0f}°, "
            f"release<{math.degrees(args_cli.grasp_release_rad):.0f}°",
            flush=True,
        )
    else:
        print("[INFO] grasp assist OFF (--no_grasp_assist)", flush=True)
        if args_cli.teleop_physics == "sketch":
            print(
                "[WARN] sketch mode without grasp assist — pick & place demos will not lift the tool.",
                flush=True,
            )
    hold = home.clone()
    teleop._last = hold.clone()
    teleop._last_applied = hold.clone()
    teleop._applied_gripper_rad = float(home[GRIPPER_ACTION_INDEX].item())
    print(f"[INFO] sim.use_fabric={raw.sim.cfg.use_fabric}", flush=True)
    print(f"[INFO] action_dim={raw.action_manager.total_action_dim}", flush=True)
    print(f"[INFO] teleop_physics={args_cli.teleop_physics}", flush=True)
    if use_sketch_teleop:
        print(
            "[INFO] sketch teleop: kinematic JSON + grasp assist; "
            "sim.step viewport + post-step re-lock; records 7D actions",
            flush=True,
        )
    elif use_direct_teleop:
        print(
            "[INFO] gripper: rh_r1 only + PhysX mimic (experimental kinematic path)",
            flush=True,
        )
    print(f"[INFO] teleop mode: {args_cli.teleop_mode}", flush=True)
    if args_cli.tool_asset == "cube":
        print(
            f"[INFO] tool_asset=cube edge={args_cli.tool_cube_size}m "
            f"(PhysX box collision; staging z={args_cli.tool_cube_size * 0.5:.3f})",
            flush=True,
        )
    else:
        print(
            f"[INFO] tool_asset=screwdriver (Screw_Driver @ staging {RETURN_TOOL_STAGING_POS})",
            flush=True,
        )
    _log_actuator_stiffness(robot)
    if use_direct_teleop:
        if use_sketch_teleop:
            print(
                f"[INFO] Teleop: 7D sketch from JSON; file={args_cli.joint_file}",
                flush=True,
            )
        else:
            print(
                f"[INFO] Teleop: legacy kinematic; json={args_cli.joint_file}",
                flush=True,
            )
        print(
            f"[INFO] deadband arm={args_cli.joint_deadband_rad} "
            f"gripper={args_cli.gripper_deadband_rad} ramp={args_cli.gripper_ramp_rad}",
            flush=True,
        )
        print("[INFO] HDF5 records 7D actions (arm + rh_r1) for BC/PPO.", flush=True)
    else:
        print(
            f"[INFO] gripper_deadband_rad={args_cli.gripper_deadband_rad} "
            "(legacy Action Graph path)",
            flush=True,
        )
    print(f"[INFO] debug_interval={args_cli.debug_interval} steps (~{args_cli.debug_interval * raw.step_dt:.1f}s)", flush=True)
    print(f"[INFO] settling home ({args_cli.home_settle_steps} steps)...", flush=True)
    _settle_at_home(
        env, robot, raw, home, args_cli.home_settle_steps,
        use_sketch_teleop=use_sketch_teleop,
        direct_teleop=use_direct_teleop and not use_sketch_teleop,
        write_joint_ids=write_joint_ids,
    )
    if grasp_assist is not None:
        grasp_assist.capture_staging_pose(robot, raw)
    if use_sketch_teleop:
        print(
            f"[INFO] ee_tool at home={_ee_tool_distance_m(raw):.3f}m — "
            "sketch JSON teleop (move leader to verify T4 mtime updates)",
            flush=True,
        )
    else:
        print(
            f"[INFO] ee_tool at home={_ee_tool_distance_m(raw):.3f}m — kinematic teleop",
            flush=True,
        )
    if args_cli.teleop_source == "json":
        jpath = args_cli.joint_file
        if os.path.isfile(jpath):
            age = time.time() - os.path.getmtime(jpath)
            try:
                with open(jpath, encoding="utf-8") as f:
                    jdata = json.load(f)
                j1 = jdata.get("joint_1", float("nan"))
                print(
                    f"[INFO] teleop JSON ok: {jpath} age={age:.2f}s joint_1={j1:+.3f} rad",
                    flush=True,
                )
            except (json.JSONDecodeError, OSError) as exc:
                print(f"[WARN] teleop JSON unreadable: {jpath} ({exc})", flush=True)
        else:
            print(
                f"[WARN] teleop JSON missing: {jpath} — start T2 leader, T3 leader_to_isaac, T4 bridge",
                flush=True,
            )
        j_probe_stamp_a = None
        try:
            with open(jpath, encoding="utf-8") as f:
                j_probe_stamp_a = json.load(f).get("stamp_sec")
        except (json.JSONDecodeError, OSError):
            pass
        time.sleep(0.35)
        j_probe_stamp_b = None
        try:
            with open(jpath, encoding="utf-8") as f:
                j_probe_stamp_b = json.load(f).get("stamp_sec")
        except (json.JSONDecodeError, OSError):
            pass
        if j_probe_stamp_a is None or j_probe_stamp_b is None or j_probe_stamp_a == j_probe_stamp_b:
            print(
                "[WARN] JSON stamp not advancing — check T3 leader_to_isaac publishes "
                "isaac/joint_command when SO ARM moves.",
                flush=True,
            )
        else:
            print("[INFO] JSON teleop stream OK (stamp_sec advancing).", flush=True)

    step = 0
    success_count = 0
    reset_flag = "/tmp/demo_reset.flag"
    next_tick = time.monotonic()
    intermission_until: float | None = None

    with torch.inference_mode():
        while simulation_app.is_running():
            if env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                print(f"[INFO] Collected {args_cli.num_demos} demos. Done.", flush=True)
                break

            if intermission_until is not None and time.monotonic() >= intermission_until:
                intermission_until = None
                _finish_success_intermission(raw, robot, grasp_assist)
                success_count = 0
                next_tick = time.monotonic()

            if os.path.isfile(reset_flag):
                os.remove(reset_flag)
                print("[INFO] Reset episode (flag).", flush=True)
                intermission_until = None
                hold, teleop = _reset_episode_immediate(
                    env, robot, raw, home, grasp_assist,
                    use_sketch_teleop=use_sketch_teleop,
                    direct_teleop=use_direct_teleop and not use_sketch_teleop,
                    write_joint_ids=write_joint_ids,
                )
                success_count = 0
                next_tick = time.monotonic()

            in_intermission = intermission_until is not None

            raw_action = teleop.read(hold)
            raw_action = raw_action.clone()
            raw_action[GRIPPER_ACTION_INDEX] = _clamp_gripper_rad(
                float(raw_action[GRIPPER_ACTION_INDEX].item())
            )
            if not use_sketch_teleop:
                raw_action = teleop.apply_stabilized(
                    raw_action,
                    args_cli.joint_deadband_rad,
                    args_cli.gripper_deadband_rad,
                )
            if use_sketch_teleop and args_cli.gripper_ramp_rad > 0.0:
                prev_g = getattr(teleop, "_applied_gripper_rad", None)
                limited = _rate_limit_gripper(
                    prev_g,
                    float(raw_action[GRIPPER_ACTION_INDEX].item()),
                    args_cli.gripper_ramp_rad,
                )
                raw_action = raw_action.clone()
                raw_action[GRIPPER_ACTION_INDEX] = limited
                teleop._applied_gripper_rad = limited
            elif use_direct_teleop and not use_sketch_teleop and args_cli.gripper_ramp_rad > 0.0:
                prev_g = getattr(teleop, "_applied_gripper_rad", None)
                limited = _rate_limit_gripper(
                    prev_g,
                    float(raw_action[GRIPPER_ACTION_INDEX].item()),
                    args_cli.gripper_ramp_rad,
                )
                raw_action = raw_action.clone()
                raw_action[GRIPPER_ACTION_INDEX] = limited
                teleop._applied_gripper_rad = limited

            active_grasp = None if in_intermission else grasp_assist

            if use_sketch_teleop:
                action = _apply_sketch_teleop_frame(
                    robot, raw, raw_action, teleop, write_joint_ids, active_grasp,
                    record=not in_intermission,
                )
            elif use_direct_teleop:
                action = raw_action.unsqueeze(0)
                changed = _teleop_changed(
                    teleop._last_applied,
                    raw_action,
                    args_cli.joint_deadband_rad,
                    args_cli.gripper_deadband_rad,
                )
                _apply_kinematic_teleop(robot, raw, raw_action)
                if active_grasp is not None:
                    raw.scene.update(dt=raw.physics_dt)
                    active_grasp.update(robot, raw, float(raw_action[GRIPPER_ACTION_INDEX].item()))
                if not in_intermission:
                    _idle_record_step(raw, action)
                if changed:
                    teleop._last_applied = raw_action.clone()
            else:
                action = raw_action.unsqueeze(0)
                _ensure_timeline_playing(raw)
                env.step(action)
                teleop._last_applied = raw_action.clone()
            if args_cli.teleop_source == "json":
                _write_joint_feedback(robot, args_cli.feedback_file)
            teleop.warn_if_stale()

            if args_cli.debug_interval > 0 and step % args_cli.debug_interval == 0:
                j1_i = robot.joint_names.index("joint_1")
                j1_sim = float(robot.data.joint_pos[0, j1_i].item())
                j1_json = float(raw_action[0].item()) if teleop.is_fresh() else float("nan")
                json_live = _json_file_live(args_cli.joint_file, args_cli.joint_stale_sec)
                mode = ("sketch" if json_live else "hold") if use_sketch_teleop else (
                    "kinematic" if use_direct_teleop else "pd"
                )
                ee_dist = _ee_tool_distance_m(raw)
                track_err = abs(j1_json - j1_sim) if teleop.is_fresh() else float("nan")
                track_ok = teleop.is_fresh() and track_err < 0.05
                r1_i = robot.joint_names.index("rh_r1")
                r1_sim = float(robot.data.joint_pos[0, r1_i].item())
                r1_cmd = float(raw_action[GRIPPER_ACTION_INDEX].item())
                l1_sim = float(robot.data.joint_pos[0, robot.joint_names.index("rh_l1")].item())
                r2_sim = float(robot.data.joint_pos[0, robot.joint_names.index("rh_r2")].item())
                ee_body = _ee_body_name(robot)
                ee_idx = robot.body_names.index(ee_body)
                ee_z = float(robot.data.body_pos_w[0, ee_idx, 2].item())
                grasp_tag = (
                    "GRASP" if (grasp_assist is not None and grasp_assist.active)
                    else "PLACED" if (grasp_assist is not None and grasp_assist._placed)
                    else "FREE" if (grasp_assist is not None and grasp_assist._released)
                    else "LOCK" if (grasp_assist is not None and grasp_assist.table_lock)
                    else "free"
                )
                dbg = (
                    f"[DEBUG] step={step} mode={mode} grasp={grasp_tag} ee_tool={ee_dist:.3f}m ee_z={ee_z:.3f} "
                    f"j1 json={j1_json:+.3f} sim={j1_sim:+.3f} err={track_err:.3f} "
                    f"grip cmd={r1_cmd:+.3f} r1={r1_sim:+.3f} l1={l1_sim:+.3f} r2={r2_sim:+.3f} "
                    f"{'TRACKING' if track_ok else 'stale/mismatch'}"
                )
                if success_term is not None and grasp_tag == "PLACED":
                    dbg += f" | {_format_demo_success_debug(env, success_term)}"
                elif success_term is None and grasp_tag == "PLACED":
                    dbg += " | auto_success OFF (add --auto_success)"
                print(dbg, flush=True)

            if not in_intermission:
                success_count, need_intermission = _process_success(env, success_term, success_count)
                if need_intermission:
                    if args_cli.post_success_delay_sec > 0:
                        _begin_success_intermission(raw, grasp_assist)
                        intermission_until = time.monotonic() + args_cli.post_success_delay_sec
                    else:
                        hold, teleop = _reset_episode_immediate(
                            env, robot, raw, home, grasp_assist,
                            use_sketch_teleop=use_sketch_teleop,
                            direct_teleop=use_direct_teleop and not use_sketch_teleop,
                            write_joint_ids=write_joint_ids,
                        )
                        success_count = 0
                        next_tick = time.monotonic()

            if raw.sim.has_gui():
                raw.sim.render()

            next_tick += raw.step_dt
            wait_s = next_tick - time.monotonic()
            if wait_s > 0:
                time.sleep(wait_s)
            elif wait_s < -raw.step_dt:
                next_tick = time.monotonic()

            step += 1

    exported = env.recorder_manager.exported_successful_episode_count
    print(f"[INFO] Exported {exported} successful demo(s) → {args_cli.dataset}", flush=True)
    if hasattr(teleop, "shutdown"):
        teleop.shutdown()
    _teardown_teleop_backend(raw)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
