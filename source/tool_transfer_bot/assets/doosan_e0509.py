import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from tool_transfer_bot.paths import asset

# Self-contained robot USD (composes parts in assets/robot/).
_ROBOT_USD = asset("e0509_rl.usda")

DOOSAN_E0509_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_ROBOT_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=16,
        ),
    ),
    prim_path="{ENV_REGEX_NS}/e0509",
    # Home position matching with_camera.usda: base on table (z=0.033), j3=90°, j5=90°
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.033),
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": math.radians(90.0),
            "joint_4": 0.0,
            "joint_5": math.radians(90.0),
            "joint_6": 0.0,
            "rh_r1": 0.0,  # gripper open
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # J1–J3: higher stiffness/damping from USDA
        "arm_lower": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-3]"],
            effort_limit_sim=11400.0,
            stiffness=316.6667,
            damping=31.6667,
        ),
        # J4–J6: lower stiffness/damping from USDA
        "arm_upper": ImplicitActuatorCfg(
            joint_names_expr=["joint_[4-6]"],
            effort_limit_sim=2400.0,
            stiffness=66.6667,
            damping=6.6667,
        ),
        # RH-P12-RN gripper: driven joint
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["rh_r1"],
            effort_limit_sim=1000.0,
            stiffness=30.0,
            damping=8.0,
        ),
        # Mimic joints — passive, driven by PhysX mimic feature
        "gripper_mimic": ImplicitActuatorCfg(
            joint_names_expr=["rh_r2", "rh_l1", "rh_l2"],
            effort_limit_sim=1000.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Doosan e0509 with RH-P12-RN gripper ArticulationCfg."""
