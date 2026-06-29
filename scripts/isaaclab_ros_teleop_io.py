"""Teleop I/O for Isaac Lab demo collector (Python 3.11 — no system rclpy).

Action Graph (OmniGraph) subscribes ``isaac/joint_command`` from ``leader_to_isaac``.
T1 records 7D actions from articulation sim state — no ROS Python client in T1.
"""

from __future__ import annotations

import os
import threading
import time

import torch
from sensor_msgs.msg import JointState

JSON_JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)] + ["rh_r1"]
FEEDBACK_JOINTS = JSON_JOINT_NAMES + ["rh_l1"]
ARM_ACTION_DIM = 6
GRIPPER_ACTION_INDEX = 6


class ActionGraphSimTeleop:
    """Read 7D demo actions from sim joints (Action Graph drives via ROS)."""

    def __init__(self, robot, home: torch.Tensor, stale_sec: float = 0.5):
        self._robot = robot
        self._home = home
        self.device = home.device
        self.stale_sec = stale_sec
        self._last: torch.Tensor | None = None
        self._last_applied: torch.Tensor | None = None
        self._last_move_stamp = 0.0
        self._last_warn = 0.0
        print(
            "[INFO] Teleop source: sim joints (Action Graph ROS → no rclpy in T1)",
            flush=True,
        )

    def _read_sim_joints(self) -> torch.Tensor:
        vals = []
        for name in JSON_JOINT_NAMES:
            idx = self._robot.joint_names.index(name)
            vals.append(float(self._robot.data.joint_pos[0, idx].item()))
        return torch.tensor(vals, device=self.device, dtype=torch.float32)

    def is_fresh(self) -> bool:
        if self._last is None:
            return False
        moved = torch.max(torch.abs(self._last - self._home)).item() > 0.01
        recent = (time.time() - self._last_move_stamp) <= self.stale_sec
        return moved and recent

    def warn_if_stale(self) -> None:
        if self.is_fresh():
            return
        now = time.time()
        if now - self._last_warn < 3.0:
            return
        self._last_warn = now
        print(
            "[WARN] Sim joints not moving — check T2/T3 leader_to_isaac "
            f"(ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '71')}), "
            "Action Graph Play, and try --disable_fabric if arm frozen",
            flush=True,
        )

    def read(self, fallback: torch.Tensor) -> torch.Tensor:
        cur = self._read_sim_joints()
        self._last = cur
        if torch.max(torch.abs(cur - self._home)).item() > 0.01:
            self._last_move_stamp = time.time()
        return cur

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


class RosJointCommandTeleop:
    """Subscribe ``isaac/joint_command``; publish ``isaac/joint_states`` from sim."""

    def __init__(
        self,
        device: torch.device,
        stale_sec: float = 0.5,
        cmd_topic: str = "isaac/joint_command",
        feedback_topic: str = "isaac/joint_states",
        domain_id: int = 71,
    ):
        self.device = device
        self.stale_sec = stale_sec
        self.cmd_topic = cmd_topic
        self.feedback_topic = feedback_topic
        self._lock = threading.Lock()
        self._last: torch.Tensor | None = None
        self._last_applied: torch.Tensor | None = None
        self._last_stamp = 0.0
        self._last_warn = 0.0
        self._node = None
        self._feedback_pub = None
        self._ok = False

        os.environ.setdefault("ROS_DOMAIN_ID", str(domain_id))
        try:
            from isaaclab_ros_action_graph import ensure_ros2_teleop_extensions  # noqa: WPS433

            ensure_ros2_teleop_extensions()
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy

            if not rclpy.ok():
                rclpy.init()

            outer = self

            class _RosTeleopNode(Node):
                def __init__(self) -> None:
                    super().__init__("isaaclab_demo_teleop_io")
                    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
                    self.create_subscription(JointState, outer.cmd_topic, outer._on_cmd, qos)
                    outer._feedback_pub = self.create_publisher(JointState, outer.feedback_topic, 10)

            self._rclpy = rclpy
            self._node = _RosTeleopNode()
            self._ok = True
            print(
                f"[INFO] ROS teleop: sub {cmd_topic}, pub {feedback_topic} (domain={domain_id})",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[WARN] ROS teleop unavailable ({exc}). "
                "Use --teleop_source json + T4 ros_joint_command_bridge.py",
                flush=True,
            )

    def spin_once(self) -> None:
        if self._ok and self._node is not None:
            self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def shutdown(self) -> None:
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        self._feedback_pub = None

    def publish_feedback(self, robot) -> None:
        """Publish sim joint positions for leader_to_isaac joint_4 hold."""
        if not self._ok or self._feedback_pub is None:
            return
        names: list[str] = []
        positions: list[float] = []
        for name in FEEDBACK_JOINTS:
            if name not in robot.joint_names:
                continue
            idx = robot.joint_names.index(name)
            names.append(name)
            positions.append(float(robot.data.joint_pos[0, idx].item()))
        if not names:
            return
        msg = JointState()
        msg.name = names
        msg.position = positions
        msg.velocity = [0.0] * len(names)
        msg.effort = [0.0] * len(names)
        self._feedback_pub.publish(msg)

    def _on_cmd(self, msg: JointState) -> None:
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        vals: list[float] = []
        for name in JSON_JOINT_NAMES:
            if name not in name_to_idx:
                return
            vals.append(float(msg.position[name_to_idx[name]]))
        tensor = torch.tensor(vals, device=self.device, dtype=torch.float32)
        with self._lock:
            self._last = tensor
            self._last_stamp = time.time()

    def is_fresh(self) -> bool:
        with self._lock:
            return self._last is not None and (time.time() - self._last_stamp) <= self.stale_sec

    def warn_if_stale(self) -> None:
        if self.is_fresh():
            return
        now = time.time()
        if now - self._last_warn < 3.0:
            return
        self._last_warn = now
        print(
            f"[WARN] No fresh {self.cmd_topic} — run T2 leader USB + T3 leader_to_isaac "
            f"(ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '?')})",
            flush=True,
        )

    def read(self, fallback: torch.Tensor) -> torch.Tensor:
        self.spin_once()
        with self._lock:
            if self._last is None:
                return fallback
            if time.time() - self._last_stamp > self.stale_sec:
                return self._last if self._last is not None else fallback
            return self._last.clone()

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
