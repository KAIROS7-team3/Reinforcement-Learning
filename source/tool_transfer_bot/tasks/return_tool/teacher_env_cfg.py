"""ReturnTool env: pick from staging → place on open toolbox."""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
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
from tool_transfer_bot.assets.environments import (
    RETURN_TOOL_DRAWER_OPEN_JOINT,
    RETURN_TOOL_PLACE_OFFSET,
    return_tool_home_joint_pos_rad,
    return_tool_staging_cube_cfg,
)
from tool_transfer_bot.tasks.base_env_cfg import (
    FRAME_MARKER_SMALL_CFG,
    EventCfg,
    JointActionsCfg,
    TeleopActionsCfg,
    ToolTransferBaseEnvCfg,
    ToolTransferSceneCfg,
)
from tool_transfer_bot.tasks import mdp

# 5 cm cube spawn: center z = 0.025 m → lift gate 0.04 m ≈ 1.5 cm above table.
_RETURN_TOOL_LIFT_MIN_Z = 0.04


@configclass
class ReturnToolRewardsCfg:
    """Isaac Lab Lift-style: reach → lift → gated place tracking → success."""

    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        weight=1.0,
        params={"std": 0.1},
    )
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        weight=15.0,
        params={"minimal_height": _RETURN_TOOL_LIFT_MIN_Z},
    )
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_tracking,
        weight=16.0,
        params={"std": 0.3, "minimal_height": _RETURN_TOOL_LIFT_MIN_Z},
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_tracking,
        weight=5.0,
        params={"std": 0.05, "minimal_height": _RETURN_TOOL_LIFT_MIN_Z},
    )
    success_bonus = RewTerm(
        func=mdp.success_bonus,
        weight=10.0,
        params={"threshold": 0.05},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-6]"])},
    )


@configclass
class ReturnToolSceneCfg(ToolTransferSceneCfg):
    """Open drawer + staging tool + place target frame."""

    tool: RigidObjectCfg = MISSING
    target_frame: FrameTransformerCfg = MISSING


@configclass
class ReturnToolObservationsCfg:
    """Pick-and-place GT observations (minimal — no redundant pose/action terms).

    Policy (19D): joint_pos(7) + object_pos(3) + target_pos(3) + target_tool_id(6)
    Critic (+7D): joint_vel(7) → 26D
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-6]", "rh_r1"])},
        )
        object_pos = ObsTerm(func=mdp.object_pos_w)
        target_pos = ObsTerm(func=mdp.target_pos_w)
        target_tool_id = ObsTerm(func=mdp.target_tool_id)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-6]", "rh_r1"])},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ReturnToolEventCfg(EventCfg):
    set_target_tool = EventTerm(
        func=mdp.set_target_tool_id_onehot,
        mode="reset",
        params={"tool_index": 0},  # screwdriver (TOOL_IDS[0])
    )
    reset_tool = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("tool"),
        },
    )
    # TOOLBOX_CFG default drawer_joint=0 (closed); bias to fully open for ReturnTool.
    reset_drawer = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("toolbox", joint_names=["drawer_joint"]),
            "position_range": (RETURN_TOOL_DRAWER_OPEN_JOINT, RETURN_TOOL_DRAWER_OPEN_JOINT),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class ReturnToolTeacherEnvCfg(ToolTransferBaseEnvCfg):
    """Task 3: staging → toolbox place (drawer already open)."""

    scene: ReturnToolSceneCfg = ReturnToolSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ReturnToolObservationsCfg = ReturnToolObservationsCfg()
    rewards: ReturnToolRewardsCfg = ReturnToolRewardsCfg()
    events: ReturnToolEventCfg = ReturnToolEventCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = DOOSAN_E0509_CFG.replace(prim_path="{ENV_REGEX_NS}/e0509")
        self.scene.robot.init_state.joint_pos = return_tool_home_joint_pos_rad()
        self.scene.toolbox = TOOLBOX_CFG.replace(prim_path="{ENV_REGEX_NS}/toolbox_with_handle")
        self.scene.toolbox.init_state.joint_pos = {"drawer_joint": RETURN_TOOL_DRAWER_OPEN_JOINT}

        # Match teleop demos + BC (5 cm cube). Screw_Driver USD is for later sim2real.
        self.scene.tool = return_tool_staging_cube_cfg(prim_path="{ENV_REGEX_NS}/tool")

        self.scene.target_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/toolbox_with_handle/toolbox/toolbox/toolbox/drawer",
            debug_vis=True,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ReturnToolPlaceFrame"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/toolbox_with_handle/toolbox/toolbox/toolbox/drawer",
                    name="place_target",
                    offset=OffsetCfg(pos=RETURN_TOOL_PLACE_OFFSET),
                ),
            ],
        )

        # Required by base scene; unused by ReturnTool obs/rewards.
        self.scene.drawer_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/toolbox_with_handle/toolbox/toolbox/toolbox/drawer",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/toolbox_with_handle/toolbox/toolbox/toolbox/drawer",
                    name="drawer_handle",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
            ],
        )

        self.terminations.task_success = DoneTerm(
            func=mdp.return_tool_success,
            params={"dist_threshold": 0.05},
        )

        self.events.reset_all = EventTerm(
            func=mdp.reset_scene_to_default,
            mode="reset",
            params={"reset_joint_targets": True},
        )

        self.episode_length_s = 12.0
        self.sim.use_fabric = True


@configclass
class ReturnToolTeacherEnvCfg_PLAY(ReturnToolTeacherEnvCfg):
    """BC checkpoint replay — match teleop demo physics (cube, no gravity sag, 7D joint targets)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0
        # Match BC teleop demos (cube @ staging), not Screw_Driver USD.
        self.scene.tool = return_tool_staging_cube_cfg(prim_path="{ENV_REGEX_NS}/tool")
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.decimation = 1
        self.sim.use_fabric = False
        # Same PD stack as demo collection (disable_gravity avoids sag vs recorded obs).
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
        self.scene.robot.init_state.joint_pos = return_tool_home_joint_pos_rad()


@configclass
class ReturnToolTeacherEnvCfg_DEMO(ReturnToolTeacherEnvCfg_PLAY):
    """SO ARM demo recording — 7D joint action for BC HDF5."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 3600.0


@configclass
class ReturnToolTeacherEnvCfg_TELEOP(ReturnToolTeacherEnvCfg_PLAY):
    """JSON teleop debug — 10D joint action (same PD stack as OpenDrawer teleop monitor)."""

    def __post_init__(self):
        super().__post_init__()
        self.actions = TeleopActionsCfg()
        self.episode_length_s = 3600.0
        self.decimation = 1
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.sync_gripper_mimic = None
        self.sim.use_fabric = False
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
        self.scene.robot.init_state.joint_pos = return_tool_home_joint_pos_rad()
