"""Export open-drawer pre-grasp joint waypoint (Method A).

Pipeline:
  1. Isaac Sim — probe knob / base_link / home EE frames
  2. Build pre-grasp TCP target (~8 cm in front of knob, aligned with handle)
  3. DSR ``motion/ikin`` (ROS2) — try sol_space 0..7 (+ optional fkin calibration)
  4. Isaac Sim — validate candidates, pick lowest EE–knob distance
  5. Write ``source/tool_transfer_bot/data/open_drawer_pregrasp.yaml``

Requires DSR virtual controller for step 3:
  ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py mode:=virtual model:=e0509 host:=127.0.0.1

Usage:
  cd /home/user/Reinforcement-Learning
  /home/user/IsaacLab/isaaclab.sh -p scripts/export_open_drawer_pregrasp.py --headless
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any

import numpy as np
import yaml

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export open-drawer pre-grasp IK waypoint.")
parser.add_argument("--task", type=str, default="Isaac-OpenDrawer-Teacher-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--approach_offset_m", type=float, default=0.08, help="Stand-off distance in front of knob.")
parser.add_argument("--dsr_namespace", type=str, default="dsr01")
parser.add_argument("--skip_ros", action="store_true", help="Skip DSR ikin; use Isaac-only joint search.")
parser.add_argument("--output", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import tool_transfer_bot  # noqa: F401, E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from tool_transfer_bot.assets.doosan_e0509 import RL_HOME_JOINT_DEG  # noqa: E402
from tool_transfer_bot.paths import PROJECT_ROOT  # noqa: E402

OUTPUT_YAML = args_cli.output or os.path.join(
    PROJECT_ROOT, "source", "tool_transfer_bot", "data", "open_drawer_pregrasp.yaml"
)

ARM_JOINTS = [f"joint_{i}" for i in range(1, 7)]
GRIPPER_JOINTS = ["rh_r1", "rh_r2", "rh_l1", "rh_l2"]
DSR_BASE_Z_OFFSET_M = 0.45


def _wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rot_to_euler_xyz_deg(r: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        rx = math.degrees(math.atan2(r[2, 1], r[2, 2]))
        ry = math.degrees(math.atan2(-r[2, 0], sy))
        rz = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    else:
        rx = math.degrees(math.atan2(-r[1, 2], r[1, 1]))
        ry = math.degrees(math.atan2(-r[2, 0], sy))
        rz = 0.0
    return rx, ry, rz


def _set_robot_joints(env, joint_deg: dict[str, float]) -> None:
    robot = env.unwrapped.scene["robot"]
    raw = env.unwrapped
    pos = robot.data.default_joint_pos.clone()
    for name, deg in joint_deg.items():
        if name not in robot.joint_names:
            continue
        idx = robot.joint_names.index(name)
        pos[:, idx] = math.radians(deg) if name.startswith("joint_") else deg
    zero = torch.zeros_like(pos)
    robot.write_joint_state_to_sim(pos, zero)
    robot.set_joint_position_target(pos)
    raw.scene.write_data_to_sim()
    for _ in range(4):
        raw.sim.step(render=False)
        raw.scene.update(dt=raw.physics_dt)


def _ee_knob_dist_m(env) -> float:
    raw = env.unwrapped
    ee = raw.scene["ee_frame"].data.target_pos_w[0, 0]
    knob = raw.scene["drawer_frame"].data.target_pos_w[0, 0]
    return torch.norm(knob - ee).item()


def _probe_scene(env) -> dict[str, Any]:
    raw = env.unwrapped
    robot = raw.scene["robot"]
    base_idx = list(robot.body_names).index("base_link")

    base_pos = robot.data.body_pos_w[0, base_idx].cpu().numpy()
    base_quat = robot.data.body_quat_w[0, base_idx].cpu().numpy()

    knob_pos = raw.scene["drawer_frame"].data.target_pos_w[0, 0].cpu().numpy()
    knob_quat = raw.scene["drawer_frame"].data.target_quat_w[0, 0].cpu().numpy()

    ee_pos = raw.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()
    ee_quat = raw.scene["ee_frame"].data.target_quat_w[0, 0].cpu().numpy()

    r_base = _wxyz_to_rot(base_quat)
    r_base_t = r_base.T
    knob_in_base = r_base_t @ (knob_pos - base_pos)
    ee_in_base = r_base_t @ (ee_pos - base_pos)

    handle_x = _wxyz_to_rot(knob_quat)[:, 0]
    pregrasp_pos_w = knob_pos - args_cli.approach_offset_m * handle_x
    pregrasp_pos_base = r_base_t @ (pregrasp_pos_w - base_pos)

    return {
        "base_pos_w": base_pos.tolist(),
        "base_quat_w": base_quat.tolist(),
        "knob_pos_w": knob_pos.tolist(),
        "knob_quat_w": knob_quat.tolist(),
        "ee_pos_w": ee_pos.tolist(),
        "ee_quat_w": ee_quat.tolist(),
        "knob_in_base_m": knob_in_base.tolist(),
        "ee_in_base_m": ee_in_base.tolist(),
        "pregrasp_pos_base_m": pregrasp_pos_base.tolist(),
        "pregrasp_quat_w": knob_quat.tolist(),
        "approach_offset_m": args_cli.approach_offset_m,
    }


def _isaac_pos_to_dsr_posx_mm(pos_base_m: np.ndarray, quat_wxyz: np.ndarray) -> list[float]:
    pos_dsr_m = pos_base_m.copy()
    pos_dsr_m[2] += DSR_BASE_Z_OFFSET_M
    r = _wxyz_to_rot(quat_wxyz)
    rx, ry, rz = _rot_to_euler_xyz_deg(r)
    return [
        float(pos_dsr_m[0] * 1000.0),
        float(pos_dsr_m[1] * 1000.0),
        float(pos_dsr_m[2] * 1000.0),
        rx,
        ry,
        rz,
    ]


def _call_dsr_fkin(joint_deg: list[float], namespace: str) -> list[float] | None:
    try:
        import rclpy
        from dsr_msgs2.srv import Fkin
        from rclpy.node import Node
    except ImportError:
        print("[WARN] rclpy/dsr_msgs2 not available — source doosan-robot2 install/setup.bash")
        return None

    if not rclpy.ok():
        rclpy.init()

    node = Node("export_pregrasp_fkin")
    client = node.create_client(Fkin, f"/{namespace}/motion/fkin")
    if not client.wait_for_service(timeout_sec=3.0):
        print(f"[WARN] /{namespace}/motion/fkin not ready — launch dsr_bringup2 virtual mode.")
        node.destroy_node()
        return None

    req = Fkin.Request()
    req.pos = [float(x) for x in joint_deg]
    req.ref = 0
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
    if not future.done():
        node.destroy_node()
        return None
    resp = future.result()
    node.destroy_node()
    if resp is None or not resp.success:
        return None
    return list(resp.conv_posx)


def _call_dsr_ikin(posx: list[float], sol_space: int, namespace: str) -> list[float] | None:
    try:
        import rclpy
        from dsr_msgs2.srv import Ikin
        from rclpy.node import Node
    except ImportError:
        return None

    if not rclpy.ok():
        rclpy.init()

    node = Node("export_pregrasp_ikin")
    client = node.create_client(Ikin, f"/{namespace}/motion/ikin")
    if not client.wait_for_service(timeout_sec=3.0):
        node.destroy_node()
        return None

    req = Ikin.Request()
    req.pos = [float(x) for x in posx]
    req.sol_space = int(sol_space)
    req.ref = 0
    future = client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=10.0)
    if not future.done():
        node.destroy_node()
        return None
    resp = future.result()
    node.destroy_node()
    if resp is None or not resp.success:
        return None
    return list(resp.conv_posj)


def _joint_dict_from_arm_deg(arm_deg: list[float]) -> dict[str, float]:
    out = dict(RL_HOME_JOINT_DEG)
    for i, deg in enumerate(arm_deg[:6], start=1):
        out[f"joint_{i}"] = float(deg)
    for g in GRIPPER_JOINTS:
        out[g] = 0.0
    return out


def _isaac_joint_search(env, seed_deg: dict[str, float]) -> dict[str, float]:
    print("[INFO] Isaac-only joint refinement (DSR ikin unavailable).")
    x0 = np.array([seed_deg[n] for n in ARM_JOINTS], dtype=np.float64)
    best_dist = float("inf")
    best_x = x0.copy()
    rng = np.random.default_rng(42)

    for trial in range(600):
        if trial == 0:
            x = x0.copy()
        else:
            x = best_x + rng.normal(0.0, 8.0 if trial < 200 else 3.0, size=6)
        jd = _joint_dict_from_arm_deg(x.tolist())
        _set_robot_joints(env, jd)
        dist = _ee_knob_dist_m(env)
        if dist < best_dist:
            best_dist = dist
            best_x = x.copy()
            print(f"  trial {trial:4d}  dist={dist:.4f} m  joints={[round(v, 1) for v in x]}")

    best_jd = _joint_dict_from_arm_deg(best_x.tolist())
    _set_robot_joints(env, best_jd)
    return best_jd, best_dist


def main() -> None:
    print("[INFO] export_open_drawer_pregrasp starting", flush=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    _set_robot_joints(env, RL_HOME_JOINT_DEG)
    probe_home = _probe_scene(env)
    home_dist = _ee_knob_dist_m(env)
    print(f"[INFO] home ee-knob dist = {home_dist:.4f} m")

    pregrasp_pos = np.array(probe_home["pregrasp_pos_base_m"], dtype=np.float64)
    pregrasp_quat = np.array(probe_home["pregrasp_quat_w"], dtype=np.float64)
    target_posx = _isaac_pos_to_dsr_posx_mm(pregrasp_pos, pregrasp_quat)
    print(f"[INFO] target DSR posx (mm, deg): {[round(v, 2) for v in target_posx]}")

    candidates: list[tuple[str, dict[str, float], float]] = []

    if not args_cli.skip_ros:
        home_arm_deg = [RL_HOME_JOINT_DEG[n] for n in ARM_JOINTS]
        fkin_home = _call_dsr_fkin(home_arm_deg, args_cli.dsr_namespace)
        if fkin_home is not None:
            print(f"[INFO] DSR fkin(home): {[round(v, 2) for v in fkin_home]}")
            ee_home = np.array(probe_home["ee_in_base_m"], dtype=np.float64)
            ee_home_dsr = np.array(fkin_home[:3], dtype=np.float64) / 1000.0
            ee_home_dsr[2] -= DSR_BASE_Z_OFFSET_M
            delta = ee_home - ee_home_dsr
            print(f"[INFO] pos calibration delta (m): {delta.round(4).tolist()}")
            corrected = pregrasp_pos + delta
            target_posx = _isaac_pos_to_dsr_posx_mm(corrected, pregrasp_quat)
            print(f"[INFO] calibrated posx: {[round(v, 2) for v in target_posx]}")

        for sol in range(8):
            joints = _call_dsr_ikin(target_posx, sol, args_cli.dsr_namespace)
            if joints is None:
                continue
            jd = _joint_dict_from_arm_deg(joints)
            _set_robot_joints(env, jd)
            dist = _ee_knob_dist_m(env)
            print(
                f"[INFO] ikin sol={sol}  dist={dist:.4f} m  "
                f"joints={[round(joints[i], 1) for i in range(6)]}"
            )
            candidates.append((f"ikin_sol{sol}", jd, dist))

    if not candidates:
        jd, dist = _isaac_joint_search(env, RL_HOME_JOINT_DEG)
        candidates.append(("isaac_search", jd, dist))

    best_name, best_jd, best_dist = min(candidates, key=lambda c: c[2])
    print(f"\n[INFO] best candidate: {best_name}  ee-knob dist={best_dist:.4f} m")

    payload = {
        "joint_deg": {k: round(float(v), 2) for k, v in best_jd.items()},
        "meta": {
            "source": best_name,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "ee_handle_dist_m": float(best_dist),
            "home_ee_handle_dist_m": float(home_dist),
            "approach_offset_m": args_cli.approach_offset_m,
            "target_posx_mm_deg": target_posx,
            "probe": probe_home,
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_YAML), exist_ok=True)
    with open(OUTPUT_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"[INFO] wrote {OUTPUT_YAML}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
