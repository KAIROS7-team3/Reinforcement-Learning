"""Concrete environment config for Isaac-OpenDrawer-Teacher-v0.

Wires in:
  - Doosan e0509 robot + RH-P12-RN gripper
  - Toolbox with 2 drawers (drawer_joint = Task-1 target)
  - drawer_frame → FurnitureKnob_01 on drawer 1
  - task_success termination at 15 cm open (75% of 0.2 m travel)
"""

from dataclasses import MISSING

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass

from tool_transfer_bot.assets import DOOSAN_E0509_CFG, TOOLBOX_CFG
from tool_transfer_bot.tasks.base_env_cfg import ToolTransferBaseEnvCfg
from tool_transfer_bot.tasks import mdp


@configclass
class OpenDrawerTeacherEnvCfg(ToolTransferBaseEnvCfg):
    """Teacher environment for Task 1: open the top drawer."""

    def __post_init__(self):
        super().__post_init__()

        # ---- Concrete assets ----
        self.scene.robot = DOOSAN_E0509_CFG.replace(prim_path="{ENV_REGEX_NS}/e0509")
        self.scene.toolbox = TOOLBOX_CFG.replace(prim_path="{ENV_REGEX_NS}/toolbox_with_handle")

        # ---- Drawer handle frame (FurnitureKnob_01 on drawer 1) ----
        # Prim hierarchy confirmed from with_camera.usda:
        # {ENV_REGEX_NS}/toolbox_with_handle/toolbox/toolbox/drawer/FurnitureKnob_01
        self.scene.drawer_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/toolbox_with_handle/toolbox/toolbox/toolbox",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/toolbox_with_handle/toolbox/toolbox/drawer",
                    name="drawer_handle",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
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

        # ---- Reduce envs for play/eval mode (override externally if needed) ----
        # Training uses num_envs=4096 from base; that is the default.


@configclass
class OpenDrawerTeacherEnvCfg_PLAY(OpenDrawerTeacherEnvCfg):
    """Smaller eval version for visualisation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.env_spacing = 3.0
