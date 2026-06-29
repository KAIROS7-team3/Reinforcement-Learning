"""Base ManagerBasedRLEnvCfg for all 4 tool-transfer sub-tasks.

Each task inherits this and overrides:
  - scene.robot / scene.toolbox  (filled with concrete DOOSAN_E0509_CFG / TOOLBOX_CFG)
  - scene.drawer_frame target_frames  (knob center offset on drawer link; not prim ``handle``)
  - rewards  (task-specific weights / joint names)
  - terminations.task_success  (task-specific threshold)
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass

from tool_transfer_bot.paths import asset

from . import mdp

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.05, 0.05, 0.05)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@configclass
class ToolTransferSceneCfg(InteractiveSceneCfg):
    """Scene: robot + workbench table + toolbox + EE/drawer frames + lights.

    Layout matches /home/user/Desktop/with_camera.usda (desk, robot, toolbox positions).
    Robot and toolbox ArticulationCfg are filled by each task cfg.
    """

    # ---- assets (filled by task cfg) ----
    robot: ArticulationCfg = MISSING
    toolbox: ArticulationCfg = MISSING

    # ---- EE FrameTransformer ----
    # Control body remains rh_p12_rn_base for DiffIK, but reward/observation TCP is
    # shifted 160 mm along the gripper-base local +Z axis toward the fingertips.
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/e0509/base_link",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EEFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/e0509/rh_p12_rn_base",
                name="ee_tcp",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.16)),
            ),
            # Left proximal finger tip
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/e0509/rh_p12_rn_l2",
                name="finger_left",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.04)),
            ),
            # Right proximal finger tip
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/e0509/rh_p12_rn_r2",
                name="finger_right",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.04)),
            ),
        ],
    )

    # ---- Drawer handle FrameTransformer (target overridden per task) ----
    drawer_frame: FrameTransformerCfg = MISSING

    # Gripper ↔ arm link contacts (self-collision penalty when EE path cuts through links).
    robot_self_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/e0509/rh_p12_rn_.*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/e0509/link_[1-5]"],
        history_length=2,
    )

    # ---- Environment static assets ----
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )
    # Workbench table — mm→m scale (0.001) baked in table_standalone.usda; assets/table.usdz is local copy
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=asset("table_standalone.usda"),
        ),
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

@configclass
class JointActionsCfg:
    """Absolute joint targets (rad): arm 6-DOF + rh_r1. Finger mimic via sync_gripper_mimic event."""

    joint_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "rh_r1",
        ],
        preserve_order=True,
        use_default_offset=False,
        scale=1.0,
    )


@configclass
class TeleopActionsCfg:
    """Absolute joint targets (rad) for SO ARM teleop — all gripper joints driven (no mimic event)."""

    joint_action: JointPositionActionCfg = JointPositionActionCfg(
        asset_name="robot",
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
        preserve_order=True,
        use_default_offset=False,
        scale=1.0,
    )


@configclass
class DiffIKActionsCfg:
    """Legacy: delta_pos(3) + delta_quat(4) via IK-relative DLS; gripper binary."""

    arm_action: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["joint_[1-6]"],
        body_name="rh_p12_rn_base",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        scale=0.5,
    )

    gripper_action: mdp.BinaryJointPositionActionCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["rh_r1"],
        open_command_expr={"rh_r1": 0.0},            # open  = 0 deg
        close_command_expr={"rh_r1": math.radians(60.0)},  # close = 60 deg
    )


# Back-compat alias (DiffIK was the original default).
ActionsCfg = DiffIKActionsCfg


# ---------------------------------------------------------------------------
# Observations  (asymmetric actor-critic)
# ---------------------------------------------------------------------------

@configclass
class ObservationsCfg:
    """Actor (PolicyCfg) and Critic (CriticCfg) observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations — no joint velocities."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-6]", "rh_r1"])},
        )
        ee_pos = ObsTerm(func=mdp.ee_pos_w)
        ee_quat = ObsTerm(func=mdp.ee_quat_w)
        gripper_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["rh_r1"])},
        )
        rel_ee_handle = ObsTerm(
            func=mdp.rel_ee_handle_distance,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "handle_frame_cfg": SceneEntityCfg("drawer_frame"),
            },
        )
        drawer_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("toolbox", joint_names=["drawer_joint"])},
        )
        target_tool_id = ObsTerm(func=mdp.target_tool_id)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations — adds joint velocities (asymmetric actor-critic)."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-6]", "rh_r1"])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-6]", "rh_r1"])},
        )
        ee_pos = ObsTerm(func=mdp.ee_pos_w)
        ee_quat = ObsTerm(func=mdp.ee_quat_w)
        gripper_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["rh_r1"])},
        )
        rel_ee_handle = ObsTerm(
            func=mdp.rel_ee_handle_distance,
            params={
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "handle_frame_cfg": SceneEntityCfg("drawer_frame"),
            },
        )
        drawer_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("toolbox", joint_names=["drawer_joint"])},
        )
        target_tool_id = ObsTerm(func=mdp.target_tool_id)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ---------------------------------------------------------------------------
# Events (Domain Randomization — physics)
# ---------------------------------------------------------------------------

@configclass
class EventCfg:
    """Physics DR applied on reset and startup."""

    # startup: material randomization
    robot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )
    toolbox_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("toolbox", body_names=".*"),
            "static_friction_range": (0.8, 1.5),
            "dynamic_friction_range": (0.8, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    # reset: scene + joint noise
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-6]"]),
            "position_range": (-0.05, 0.05),  # ± 0.05 rad (≈ ± 3°)
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_drawer = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("toolbox", joint_names=["drawer_joint"]),
            "position_range": (0.0, 0.0),  # fully closed (limits: 0 closed, -0.2 m open)
            "velocity_range": (0.0, 0.0),
        },
    )

    # Every control step: set mimic finger targets = rh_r1 (same sign)
    sync_gripper_mimic = EventTerm(
        func=mdp.sync_gripper_mimic_targets,
        mode="interval",
        interval_range_s=(0.0, 0.0),
    )


# ---------------------------------------------------------------------------
# Rewards  (drawer open — Task 1 / Task 4 pattern)
# Subclasses override as needed.
# ---------------------------------------------------------------------------

# Module-level only — must NOT live inside @configclass (would become reward terms).
_GRIPPER_CLOSED_GATE_CFG = SceneEntityCfg("robot", joint_names=["rh_r1"])
_DRAWER_GRIPPER_CLOSE_THRESHOLD_RAD = math.radians(30.0)


@configclass
class RewardsCfg:
    """Multi-stage reward for drawer open/close. Weight values follow RL_strategy.md."""

    # stage 1: EE → handle approach
    approach_ee_handle = RewTerm(
        func=mdp.approach_ee_handle,
        weight=2.0,
        params={"threshold": 0.12, "max_distance": 0.28},
    )
    align_ee_handle = RewTerm(
        func=mdp.align_ee_handle,
        weight=0.5,
        params={"max_distance": 0.28},
    )

    # stage 2: fingertip pinch around knob (drawer_frame; not prim ``handle``)
    approach_gripper_handle = RewTerm(
        func=mdp.approach_gripper_handle,
        weight=5.0,
        params={"z_tol": 0.02, "y_tol": 0.03, "x_tol": 0.025},
    )
    align_grasp_around_handle = RewTerm(
        func=mdp.align_grasp_around_handle,
        weight=0.5,
        params={"z_sigma": 0.015, "y_min_sep": 0.008},
    )
    grasp_handle = RewTerm(
        func=mdp.grasp_handle,
        weight=0.5,
        params={
            "threshold": 0.06,
            "grasp_align_threshold": 0.3,
            "open_joint_pos": 0.0,
            "close_joint_pos": math.radians(60.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["rh_r1"]),
        },
    )

    # stage 3: open drawer
    open_drawer_bonus = RewTerm(
        func=mdp.open_drawer_bonus,
        weight=7.5,
        params={
            "asset_cfg": SceneEntityCfg("toolbox", joint_names=["drawer_joint"]),
            "gripper_asset_cfg": _GRIPPER_CLOSED_GATE_CFG,
            "close_threshold": _DRAWER_GRIPPER_CLOSE_THRESHOLD_RAD,
        },
    )
    multi_stage_open_drawer = RewTerm(
        func=mdp.multi_stage_open_drawer,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("toolbox", joint_names=["drawer_joint"]),
            "gripper_asset_cfg": _GRIPPER_CLOSED_GATE_CFG,
            "close_threshold": _DRAWER_GRIPPER_CLOSE_THRESHOLD_RAD,
        },
    )

    # penalties
    robot_self_collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.05,
        params={"sensor_cfg": SceneEntityCfg("robot_self_contacts"), "threshold": 1.0},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint_[1-6]"])},
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # task_success is added by each sub-task cfg


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@configclass
class ToolTransferBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Abstract base for all 4 tool-transfer tasks. Must not be instantiated directly."""

    scene: ToolTransferSceneCfg = ToolTransferSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: JointActionsCfg = JointActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0
        self.viewer.eye = (-1.5, 1.5, 1.5)
        self.viewer.lookat = (0.4, 0.5, 0.3)
        self.sim.dt = 1.0 / 120.0  # 120Hz physics, 60Hz policy (decimation=2)
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
        # Pre-grasp + gripper/drawer contacts scale with num_envs; default 5*2**15 overflows ~2k envs.
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15
        self.scene.robot_self_contacts.update_period = self.sim.dt
