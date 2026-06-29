"""Concrete environment config for Isaac-OpenDrawer-Teacher-v0.

Wires in:
  - Doosan e0509 robot + RH-P12-RN gripper
  - Toolbox with 2 drawers (drawer_joint = Task-1 target)
  - drawer_frame → drawer link + knob offset (FurnitureKnob_01 / drawer_handle_top)
  - task_success termination at 15 cm open (75% of 0.2 m travel)
"""

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass

from tool_transfer_bot.assets import DOOSAN_E0509_CFG, TOOLBOX_CFG
from tool_transfer_bot.assets.doosan_e0509 import (
    DOOSAN_E0509_TELEOP_ACTUATORS,
    MAX_DEPENETRATION_VELOCITY,
)
from tool_transfer_bot.assets.pregrasp_waypoints import OPEN_DRAWER_PREGRASP_JOINT_DEG
from tool_transfer_bot.assets.environments import _DRAWER_HANDLE_OFFSET
from tool_transfer_bot.tasks.base_env_cfg import FRAME_MARKER_SMALL_CFG, JointActionsCfg, TeleopActionsCfg, ToolTransferBaseEnvCfg
from tool_transfer_bot.tasks import mdp


@configclass
class OpenDrawerTeacherEnvCfg(ToolTransferBaseEnvCfg):
    """Teacher environment for Task 1: open the top drawer."""

    def __post_init__(self):
        super().__post_init__()

        # ---- Concrete assets ----
        self.scene.robot = DOOSAN_E0509_CFG.replace(prim_path="{ENV_REGEX_NS}/e0509")
        self.scene.toolbox = TOOLBOX_CFG.replace(prim_path="{ENV_REGEX_NS}/toolbox_with_handle")

        self.sim.use_fabric = True

        # PhysX drawer body + offset → knob center (reward / debug marker aim point)
        self.scene.drawer_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/toolbox_with_handle/toolbox/toolbox/toolbox/drawer",
            debug_vis=True,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/OpenDrawerHandleFrame"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/toolbox_with_handle/toolbox/toolbox/toolbox/drawer",
                    name="drawer_handle",
                    offset=OffsetCfg(pos=_DRAWER_HANDLE_OFFSET),
                ),
            ],
        )

        # ---- Success: drawer open ≥ 15 cm ----
        self.terminations.task_success = DoneTerm(
            func=mdp.task_success,
            params={
                "threshold": 0.15,
                "asset_cfg": SceneEntityCfg("toolbox", joint_names=["drawer_joint"]),
            },
        )

        # ---- Pre-grasp bootstrap (Method A: offline DSR ikin waypoint) ----
        # Replaces home-pose reset so RL starts near the knob (grasp + pull only).
        self.events.reset_robot_joints = None
        self.events.reset_robot_pregrasp = EventTerm(
            func=mdp.reset_robot_to_pregrasp,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        "joint_1",
                        "joint_2",
                        "joint_3",
                        "joint_4",
                        "joint_5",
                        "joint_6",
                        "rh_r1",
                        "rh_r2",
                        "rh_l1",
                        "rh_l2",
                    ],
                ),
                "joint_positions_deg": OPEN_DRAWER_PREGRASP_JOINT_DEG,
                "position_range": (-0.02, 0.02),  # ± ~1° arm DR (reduced vs home ±3°)
            },
        )

        # ---- Reduce envs for play/eval mode (override externally if needed) ----
        # Training uses num_envs=4096 from base; that is the default.


@configclass
class OpenDrawerTeacherEnvCfg_PLAY(OpenDrawerTeacherEnvCfg):
    """Smaller eval version for visualisation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0
        self.scene.drawer_frame.debug_vis = True


@configclass
class OpenDrawerTeacherEnvCfg_DEMO(OpenDrawerTeacherEnvCfg_PLAY):
    """Joint-action env for SO ARM demo recording (7D action, mimic sync on)."""

    def __post_init__(self):
        super().__post_init__()
        self.actions = JointActionsCfg()
        self.episode_length_s = 3600.0
        self.decimation = 1
        self.events.reset_robot_pregrasp.params["position_range"] = (0.0, 0.0)
        self.scene.robot = DOOSAN_E0509_CFG.replace(
            prim_path="{ENV_REGEX_NS}/e0509",
            actuators=DOOSAN_E0509_TELEOP_ACTUATORS,
            spawn=DOOSAN_E0509_CFG.spawn.replace(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
                ),
            ),
        )


# TeleopActionsCfg lives in base_env_cfg (shared with ReturnTool demo collector).
@configclass
class OpenDrawerTeacherEnvCfg_TELEOP(OpenDrawerTeacherEnvCfg_PLAY):
    """Play env with joint-position actions instead of DiffIK (teleop / reward monitor)."""

    def __post_init__(self):
        super().__post_init__()
        self.actions = TeleopActionsCfg()
        # Teleop: long session, deterministic reset (no ±3° DR that looks like random snaps)
        self.episode_length_s = 3600.0
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.decimation = 1
        # Teleop joint_action drives mimic fingers; interval sync would fight direct PD
        self.events.sync_gripper_mimic = None
        # No gravity sag while holding joint targets during manual teleop
        self.scene.robot = DOOSAN_E0509_CFG.replace(
            prim_path="{ENV_REGEX_NS}/e0509",
            actuators=DOOSAN_E0509_TELEOP_ACTUATORS,
            spawn=DOOSAN_E0509_CFG.spawn.replace(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
                ),
            ),
        )
