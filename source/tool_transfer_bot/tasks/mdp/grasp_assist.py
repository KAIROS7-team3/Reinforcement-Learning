"""Kinematic grasp assist — table lock + finger weld (matches teleop demo collection)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import isaaclab.utils.math as math_utils

from tool_transfer_bot.assets.environments import (
    RETURN_TOOL_DRAWER_FLOOR_X_MAX,
    RETURN_TOOL_DRAWER_FLOOR_X_MIN,
    RETURN_TOOL_DRAWER_FLOOR_Y_MAX,
    RETURN_TOOL_DRAWER_FLOOR_Y_MIN,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_GRIPPER_FINGER_PAIRS = (
    ("rh_p12_rn_r1", "rh_p12_rn_l1"),
    ("rh_p12_rn_r2", "rh_p12_rn_l2"),
)

# Defaults aligned with collect_demos_teleop.py
DEFAULT_GRASP_DIST_M = 0.10
DEFAULT_GRASP_CLOSE_RAD = 0.35
DEFAULT_GRASP_RELEASE_RAD = 0.20
DEFAULT_GRASP_HOLD_FRAMES = 1
DEFAULT_GRASP_SNAP_INWARD_M = 0.03


def _ee_body_name(robot) -> str:
    if "rh_p12_rn_base" in robot.body_names:
        return "rh_p12_rn_base"
    return "link_6"


def _ee_body_index(robot) -> int:
    return robot.body_names.index(_ee_body_name(robot))


def _finger_midpoint_world(robot) -> torch.Tensor | None:
    for r_name, l_name in _GRIPPER_FINGER_PAIRS:
        if r_name in robot.body_names and l_name in robot.body_names:
            ri = robot.body_names.index(r_name)
            li = robot.body_names.index(l_name)
            return 0.5 * (robot.data.body_pos_w[:, ri] + robot.data.body_pos_w[:, li])
    return None


def _gripper_grasp_anchor_pos(robot) -> torch.Tensor:
    mid = _finger_midpoint_world(robot)
    if mid is not None:
        return mid
    ee_idx = _ee_body_index(robot)
    return robot.data.body_pos_w[:, ee_idx]


def _gripper_grasp_frame(robot) -> tuple[torch.Tensor, torch.Tensor]:
    ee_idx = _ee_body_index(robot)
    return _gripper_grasp_anchor_pos(robot), robot.data.body_quat_w[:, ee_idx]


def _tool_half_height_z(tool) -> float:
    spawn = tool.cfg.spawn
    if hasattr(spawn, "size"):
        return float(spawn.size[2]) * 0.5
    return 0.02


def write_tool_root(raw, tool, pos_w: torch.Tensor, quat_w: torch.Tensor) -> None:
    root_state = tool.data.root_state_w.clone()
    root_state[:, :3] = pos_w
    root_state[:, 3:7] = quat_w
    root_state[:, 7:] = 0.0
    tool.write_root_state_to_sim(root_state)
    raw.scene.write_data_to_sim()


class GraspAssist:
    """Kinematic weld + table lock: cube fixed on staging until engage; follows fingers until release."""

    def __init__(
        self,
        *,
        dist_m: float = DEFAULT_GRASP_DIST_M,
        close_rad: float = DEFAULT_GRASP_CLOSE_RAD,
        release_rad: float = DEFAULT_GRASP_RELEASE_RAD,
        hold_frames: int = DEFAULT_GRASP_HOLD_FRAMES,
        snap_inward_m: float = DEFAULT_GRASP_SNAP_INWARD_M,
        table_lock: bool = True,
        place_radius_m: float = 0.12,
        place_z_band_m: float = 0.10,
        place_snap: bool = False,
    ) -> None:
        self.dist_m = dist_m
        self.close_rad = close_rad
        self.release_rad = release_rad
        self.hold_frames = max(1, hold_frames)
        self.snap_inward_m = snap_inward_m
        self.table_lock = table_lock
        self.place_radius_m = place_radius_m
        self.place_z_band_m = place_z_band_m
        self.place_snap = place_snap
        self.active = False
        self._released = False
        self._placed = False
        self._near_frames = 0
        self._tool_pos_grasp: torch.Tensor | None = None
        self._tool_quat_grasp: torch.Tensor | None = None
        self._lock_pos: torch.Tensor | None = None
        self._lock_quat: torch.Tensor | None = None
        self._placed_pos: torch.Tensor | None = None
        self._placed_quat: torch.Tensor | None = None
        self._logged_active = False

    def capture_staging_pose(self, robot, raw) -> None:
        tool = raw.scene["tool"]
        self._lock_pos = tool.data.root_pos_w.clone()
        self._lock_quat = tool.data.root_quat_w.clone()
        self._released = False

    def reset(self, robot=None, raw=None) -> None:
        if self.active:
            print("[INFO] grasp assist: released (episode reset)", flush=True)
        self.active = False
        self._near_frames = 0
        self._tool_pos_grasp = None
        self._tool_quat_grasp = None
        self._logged_active = False
        self._placed = False
        self._placed_pos = None
        self._placed_quat = None
        if robot is not None and raw is not None:
            self.capture_staging_pose(robot, raw)
        else:
            self._released = False

    def _hold_lock(self, raw, tool) -> None:
        if not self.table_lock or self._lock_pos is None or self._lock_quat is None:
            return
        write_tool_root(raw, tool, self._lock_pos, self._lock_quat)

    def _drawer_place_pose(self, raw) -> torch.Tensor | None:
        try:
            frame = raw.scene["target_frame"]
            names = frame.data.target_frame_names
            idx = names.index("place_target") if "place_target" in names else 0
            return frame.data.target_pos_w[:, idx].clone()
        except (KeyError, ValueError, AttributeError):
            return None

    def _tool_xy_in_drawer(self, raw, tool_pos: torch.Tensor) -> bool:
        try:
            frame = raw.scene["target_frame"]
            tool = raw.scene["tool"]
            tool_local, _ = math_utils.subtract_frame_transforms(
                frame.data.source_pos_w,
                frame.data.source_quat_w,
                tool_pos,
                tool.data.root_quat_w,
            )
            margin = _tool_half_height_z(tool)
            return bool(
                (tool_local[0, 0] >= RETURN_TOOL_DRAWER_FLOOR_X_MIN + margin)
                and (tool_local[0, 0] <= RETURN_TOOL_DRAWER_FLOOR_X_MAX - margin)
                and (tool_local[0, 1] >= RETURN_TOOL_DRAWER_FLOOR_Y_MIN + margin)
                and (tool_local[0, 1] <= RETURN_TOOL_DRAWER_FLOOR_Y_MAX - margin)
            )
        except (KeyError, ValueError, AttributeError):
            return False

    def _settle_on_drawer(
        self,
        raw,
        tool,
        tool_pos: torch.Tensor,
        tool_quat: torch.Tensor,
        *,
        snap_xy_to_marker: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            frame = raw.scene["target_frame"]
        except KeyError:
            return tool_pos.clone(), tool_quat.clone()
        src_pos = frame.data.source_pos_w
        src_quat = frame.data.source_quat_w
        tool_local, _ = math_utils.subtract_frame_transforms(
            src_pos, src_quat, tool_pos, tool_quat
        )
        settled_local = tool_local.clone()
        settled_local[:, 2] = _tool_half_height_z(tool)
        if snap_xy_to_marker:
            place_local = frame.data.target_pos_source[..., 0, :]
            settled_local[:, 0] = place_local[:, 0]
            settled_local[:, 1] = place_local[:, 1]
        upright_local = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]],
            device=tool_pos.device,
            dtype=tool_pos.dtype,
        ).expand(tool_pos.shape[0], -1)
        return math_utils.combine_frame_transforms(
            src_pos, src_quat, settled_local, upright_local
        )

    def _try_snap_to_drawer(
        self,
        raw,
        tool,
        tool_pos: torch.Tensor,
        tool_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        place_pos = self._drawer_place_pose(raw)
        if place_pos is None:
            return None
        xy_dist = float(torch.norm(tool_pos[0, :2] - place_pos[0, :2]).item())
        z_diff = abs(float(tool_pos[0, 2].item() - place_pos[0, 2].item()))
        if xy_dist > self.place_radius_m or z_diff > self.place_z_band_m:
            return None
        return self._settle_on_drawer(raw, tool, tool_pos, tool_quat, snap_xy_to_marker=True)

    def _release(self, robot, raw) -> None:
        tool = raw.scene["tool"]
        tool_pos = tool.data.root_pos_w.clone()
        tool_quat = tool.data.root_quat_w.clone()
        if self.place_snap:
            snapped = self._try_snap_to_drawer(raw, tool, tool_pos, tool_quat)
            if snapped is not None:
                tool_pos, tool_quat = snapped
                self._placed = True
                self._placed_pos = tool_pos.clone()
                self._placed_quat = tool_quat.clone()
                print(
                    "[INFO] grasp assist: placed on drawer (snap → place_target, kinematic lock)",
                    flush=True,
                )
            else:
                print(
                    "[INFO] grasp assist: released in place (outside place_snap radius)",
                    flush=True,
                )
        elif self._tool_xy_in_drawer(raw, tool_pos):
            tool_pos, tool_quat = self._settle_on_drawer(raw, tool, tool_pos, tool_quat)
            self._placed = True
            self._placed_pos = tool_pos.clone()
            self._placed_quat = tool_quat.clone()
            half_h = _tool_half_height_z(tool)
            print(
                f"[INFO] grasp assist: released in drawer "
                f"(XY hold, floor z={half_h:.3f}m, upright)",
                flush=True,
            )
        else:
            print("[INFO] grasp assist: released in place (zero velocity)", flush=True)
        write_tool_root(raw, tool, tool_pos, tool_quat)

    def _engage(self, robot, raw, gripper_rad: float, dist: float) -> None:
        tool = raw.scene["tool"]
        tool_pos = tool.data.root_pos_w.clone()
        tool_quat = tool.data.root_quat_w

        if self.snap_inward_m > 0.0:
            anchor = _gripper_grasp_anchor_pos(robot)
            delta = anchor[0] - tool_pos[0]
            dist_to_anchor = float(torch.norm(delta).item())
            if dist_to_anchor > 1e-6:
                step = min(self.snap_inward_m, dist_to_anchor)
                tool_pos[0] = tool_pos[0] + delta / dist_to_anchor * step
                write_tool_root(raw, tool, tool_pos, tool_quat)

        grasp_pos, grasp_quat = _gripper_grasp_frame(robot)
        pos_grasp, quat_grasp = math_utils.subtract_frame_transforms(
            grasp_pos, grasp_quat, tool_pos, tool_quat
        )
        self._tool_pos_grasp = pos_grasp.clone()
        self._tool_quat_grasp = quat_grasp.clone()
        self.active = True
        if not self._logged_active:
            self._logged_active = True
            mid = _finger_midpoint_world(robot)
            mid_dist = (
                float(torch.norm(mid[0] - tool_pos[0]).item()) if mid is not None else float("nan")
            )
            print(
                f"[INFO] grasp assist: ENGAGED (finger-mid weld, "
                f"finger_mid→tool={mid_dist * 1000:.0f}mm dist={dist:.3f}m "
                f"rh_r1={math.degrees(gripper_rad):.1f}°)",
                flush=True,
            )

    def update(self, robot, raw, gripper_rad: float) -> None:
        tool = raw.scene["tool"]

        if self.active:
            if gripper_rad <= self.release_rad:
                self.active = False
                self._near_frames = 0
                self._tool_pos_grasp = None
                self._tool_quat_grasp = None
                self._logged_active = False
                self._released = True
                self._lock_pos = None
                self._lock_quat = None
                self._release(robot, raw)
                return

            pos_grasp, quat_grasp = self._tool_pos_grasp, self._tool_quat_grasp
            if pos_grasp is None or quat_grasp is None:
                return
            grasp_pos, grasp_quat = _gripper_grasp_frame(robot)
            new_pos, new_quat = math_utils.combine_frame_transforms(
                grasp_pos, grasp_quat, pos_grasp, quat_grasp
            )
            write_tool_root(raw, tool, new_pos, new_quat)
            return

        if self._placed and self._placed_pos is not None and self._placed_quat is not None:
            write_tool_root(raw, tool, self._placed_pos, self._placed_quat)
            return

        if self._released:
            return

        self._hold_lock(raw, tool)

        lock_pos = self._lock_pos if self._lock_pos is not None else tool.data.root_pos_w
        anchor_pos = _gripper_grasp_anchor_pos(robot)
        dist = float(torch.norm(anchor_pos[0] - lock_pos[0]).item())

        if gripper_rad >= self.close_rad and dist <= self.dist_m:
            self._near_frames += 1
        else:
            self._near_frames = 0

        if self._near_frames < self.hold_frames:
            return

        self._engage(robot, raw, gripper_rad, dist)

    def reassert_tool(self, robot, raw) -> None:
        """Re-apply kinematic tool pose after sim.step (prevents drawer bounce / drift)."""
        if self._released and not self._placed:
            return
        tool = raw.scene["tool"]
        if self.active:
            pos_grasp, quat_grasp = self._tool_pos_grasp, self._tool_quat_grasp
            if pos_grasp is None or quat_grasp is None:
                return
            grasp_pos, grasp_quat = _gripper_grasp_frame(robot)
            new_pos, new_quat = math_utils.combine_frame_transforms(
                grasp_pos, grasp_quat, pos_grasp, quat_grasp
            )
            write_tool_root(raw, tool, new_pos, new_quat)
        elif self._placed and self._placed_pos is not None and self._placed_quat is not None:
            write_tool_root(raw, tool, self._placed_pos, self._placed_quat)
        elif not self._released:
            self._hold_lock(raw, tool)
