"""Kinematic 7D joint commands — same execution path as sketch teleop demo collection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

ARM_ACTION_DIM = 6
GRIPPER_ACTION_INDEX = 6
POLICY_JOINT_NAMES = tuple(f"joint_{i}" for i in range(1, 7)) + ("rh_r1",)


def policy_joint_ids(robot: Articulation) -> list[int]:
    return [robot.joint_names.index(n) for n in POLICY_JOINT_NAMES]


def clip_policy_joint_action(robot: Articulation, core7: torch.Tensor) -> torch.Tensor:
    """Clamp 7D joint targets to soft limits (prevents kinematic closed-loop blow-up)."""
    if core7.dim() == 2:
        core7 = core7[0]
    ids = policy_joint_ids(robot)
    lows = robot.data.soft_joint_pos_limits[0, ids, 0]
    highs = robot.data.soft_joint_pos_limits[0, ids, 1]
    return core7[:7].clamp(lows, highs)


def _arm_joint_ids(robot: Articulation) -> list[int]:
    return [robot.joint_names.index(f"joint_{i}") for i in range(1, 7)]


def _expand_gripper_mimic(core: torch.Tensor) -> torch.Tensor:
    """7D (arm + rh_r1) → 10D for full gripper mimic write."""
    if core.dim() == 1:
        r1 = core[-1:]
        return torch.cat([core, r1, r1, r1], dim=0)
    r1 = core[..., -1:]
    return torch.cat([core, r1, r1, r1], dim=-1)


def _write_joint_ids(robot: Articulation) -> list[int]:
    names = [f"joint_{i}" for i in range(1, 7)] + ["rh_r1", "rh_r2", "rh_l1", "rh_l2"]
    return [robot.joint_names.index(n) for n in names if n in robot.joint_names]


def apply_kinematic_joint_action(
    env: ManagerBasedEnv,
    core7: torch.Tensor,
    *,
    viewport_step: bool = True,
) -> None:
    """Teleport arm + gripper to absolute joint targets (BC sketch replay).

    Matches ``collect_demos_teleop.py`` sketch path: write state, sim step, re-lock.
    """
    raw = env
    robot = raw.scene["robot"]
    if core7.dim() == 2:
        core7 = core7[0]
    full = _expand_gripper_mimic(core7)
    joint_ids = _write_joint_ids(robot)
    target = full.unsqueeze(0)[:, : len(joint_ids)]
    zero_vel = torch.zeros_like(target)

    robot.write_joint_state_to_sim(target, zero_vel, joint_ids=joint_ids)
    robot.set_joint_position_target(target, joint_ids=joint_ids)
    raw.scene.write_data_to_sim()

    if viewport_step and not raw.sim.cfg.use_fabric:
        raw.sim.step(render=raw.sim.has_gui() or raw.sim.has_rtx_sensors())

    robot.write_joint_state_to_sim(target, zero_vel, joint_ids=joint_ids)
    robot.set_joint_position_target(target, joint_ids=joint_ids)
    raw.scene.write_data_to_sim()

    if viewport_step and (raw.sim.has_gui() or raw.sim.has_rtx_sensors()):
        raw.sim.render()
    if viewport_step:
        raw.scene.update(dt=raw.physics_dt)


def apply_kinematic_joint_action_interp(
    env: ManagerBasedEnv,
    core7: torch.Tensor,
    substeps: int = 1,
) -> None:
    """Linearly interpolate from current 7D joints to target (smoother viewport motion)."""
    if substeps <= 1:
        apply_kinematic_joint_action(env, core7, viewport_step=True)
        return
    raw = env
    robot = raw.scene["robot"]
    ids = policy_joint_ids(robot)
    if core7.dim() == 2:
        core7 = core7[0]
    q0 = robot.data.joint_pos[0, ids].clone()
    target = core7[:7].to(q0.device)
    for k in range(1, substeps + 1):
        alpha = float(k) / float(substeps)
        q = q0 * (1.0 - alpha) + target * alpha
        apply_kinematic_joint_action(env, q, viewport_step=(k == substeps))


def advance_bc_sketch_timestep(
    env: ManagerBasedEnv,
    core7: torch.Tensor,
    *,
    interp_substeps: int = 1,
) -> None:
    """One BC sketch tick — kinematic pose only (no ``env.step`` PhysX PD).

    Demo collection uses sketch teleport + ``_idle_record_step``; calling ``env.step``
    first runs PD and corrupts the pose before kinematic lock.
    """
    raw = env
    robot = raw.scene["robot"]
    clipped = clip_policy_joint_action(robot, core7)
    actions = clipped.unsqueeze(0)
    raw.action_manager.process_action(actions.to(raw.device))
    apply_kinematic_joint_action_interp(raw, actions[0], substeps=interp_substeps)
    raw.episode_length_buf += 1
    raw.common_step_counter += 1

