#!/usr/bin/env python3
"""Bridge ROS ``isaac/joint_command`` → JSON for RL teleop reward monitor.

Also publishes ``isaac/joint_states`` from a feedback JSON written by
``teleop_reward_monitor.py`` so ``joint_teleop_gui.py`` can sync sliders.

Run with system Python 3.10 + ROS Humble (NOT env_isaaclab):

    export ROS_DOMAIN_ID=71
    source /opt/ros/humble/setup.bash
    python3 scripts/ros_joint_command_bridge.py

Then in another terminal run ``joint_teleop_gui.py`` (same ROS_DOMAIN_ID).
The RL monitor reads the JSON file each sim step.

Output format (/tmp/isaac_teleop_joints.json by default):
    {"joint_1": <rad>, ..., "joint_6": <rad>, "rh_r1": <rad>, "stamp_sec": <float>}

Feedback input (/tmp/isaac_sim_joint_states.json from teleop_reward_monitor.py):
    same joint keys in rad — republished as isaac/joint_states @ 20 Hz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

if sys.version_info[:2] != (3, 10):
    sys.exit(
        "ros_joint_command_bridge.py requires Python 3.10 (ROS Humble).\n"
        f"Current: {sys.version.split()[0]}\n"
        "  conda deactivate && source /opt/ros/humble/setup.bash"
    )

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

ARM_JOINTS = [f"joint_{i}" for i in range(1, 7)]
GRIPPER_JOINT = "rh_r1"
GRIPPER_MIMIC = "rh_l1"
TRACKED = ARM_JOINTS + [GRIPPER_JOINT]
FEEDBACK_JOINTS = ARM_JOINTS + [GRIPPER_JOINT, GRIPPER_MIMIC]
FEEDBACK_HZ = 20.0


class JointCommandBridge(Node):
    def __init__(self, output_path: str, feedback_path: str):
        super().__init__("joint_command_bridge")
        self._output_path = output_path
        self._feedback_path = feedback_path
        self._latest: dict[str, float] = {}
        self._feedback_pub = self.create_publisher(JointState, "isaac/joint_states", 10)
        self.create_subscription(JointState, "isaac/joint_command", self._on_cmd, 10)
        self.create_timer(1.0 / FEEDBACK_HZ, self._publish_feedback)
        self.get_logger().info(f"Writing {TRACKED} → {output_path}")
        self.get_logger().info(f"Reading sim feedback {FEEDBACK_JOINTS} ← {feedback_path} → isaac/joint_states")

    def _on_cmd(self, msg: JointState) -> None:
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        out: dict[str, float] = {"stamp_sec": time.time()}
        for name in TRACKED:
            if name in name_to_idx:
                out[name] = float(msg.position[name_to_idx[name]])
        if len(out) <= 1:
            return
        self._latest = out
        tmp = self._output_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(out, f)
        os.replace(tmp, self._output_path)

    def _publish_feedback(self) -> None:
        if not os.path.isfile(self._feedback_path):
            return
        try:
            with open(self._feedback_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        names: list[str] = []
        positions: list[float] = []
        for name in FEEDBACK_JOINTS:
            if name in data:
                names.append(name)
                positions.append(float(data[name]))
        if not names:
            return

        msg = JointState()
        msg.name = names
        msg.position = positions
        msg.velocity = [0.0] * len(names)
        msg.effort = [0.0] * len(names)
        self._feedback_pub.publish(msg)


def main() -> None:
    parser = argparse.ArgumentParser(description="ROS joint_command → JSON bridge for RL teleop monitor.")
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/isaac_teleop_joints.json",
        help="JSON file written on each isaac/joint_command message.",
    )
    parser.add_argument(
        "--feedback",
        type=str,
        default="/tmp/isaac_sim_joint_states.json",
        help="Sim joint feedback JSON → republished as isaac/joint_states.",
    )
    args = parser.parse_args()

    rclpy.init()
    node = JointCommandBridge(args.output, args.feedback)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
