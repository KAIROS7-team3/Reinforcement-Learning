import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from tool_transfer_bot.paths import asset

# Self-contained robot USD (composes parts in assets/robot/).
_ROBOT_USD = asset("e0509_rl.usda")

# PhysX tuning for pinch / drawer contact (reduces tunneling vs effort=1000, depen=5).
MAX_DEPENETRATION_VELOCITY = 0.75
GRIPPER_EFFORT_LIMIT_SIM = 150.0
GRIPPER_STIFFNESS_RL = 0.0
GRIPPER_DAMPING_RL = 0.0

# Training / teleop home pose (degrees). Used by init_state and teleop scripts.
# Arm pose tuned so EE starts farther from the drawer knob (reduces idle approach reward).
RL_HOME_JOINT_DEG: dict[str, float] = {
    "joint_1": -19.11,
    "joint_2": 45.8,
    "joint_3": 111.8,
    "joint_4": 71.7,
    "joint_5": 96.5,
    "joint_6": -70.0,
    "rh_r1": 0.0,
    "rh_r2": 0.0,
    "rh_l1": 0.0,
    "rh_l2": 0.0,
}


DOOSAN_E0509_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_ROBOT_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=16,
        ),
    ),
    prim_path="{ENV_REGEX_NS}/e0509",
    # Home pose (degrees): see RL_HOME_JOINT_DEG
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.033),
        joint_pos={
            name: math.radians(deg) if name.startswith("joint_") else deg
            for name, deg in RL_HOME_JOINT_DEG.items()
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
        # RH-P12-RN: all finger joints PD-driven (same joint-space target as rh_r1)
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["rh_r1", "rh_r2", "rh_l1", "rh_l2"],
            effort_limit_sim=GRIPPER_EFFORT_LIMIT_SIM,
            stiffness=GRIPPER_STIFFNESS_RL,
            damping=GRIPPER_DAMPING_RL,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Doosan e0509 with RH-P12-RN gripper ArticulationCfg."""

DOOSAN_E0509_TELEOP_ACTUATORS = {
    "arm_lower": ImplicitActuatorCfg(
        joint_names_expr=["joint_[1-3]"],
        effort_limit_sim=11400.0,
        stiffness=316.6667,
        damping=31.6667,
    ),
    # Higher damping for stable joint-position teleop (reduces wrist oscillation).
    "arm_upper": ImplicitActuatorCfg(
        joint_names_expr=["joint_[4-6]"],
        effort_limit_sim=2400.0,
        stiffness=200.0,
        damping=40.0,
    ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["rh_r1", "rh_r2", "rh_l1", "rh_l2"],
        effort_limit_sim=GRIPPER_EFFORT_LIMIT_SIM,
        stiffness=GRIPPER_STIFFNESS_RL,
        damping=GRIPPER_DAMPING_RL,
    ),
}
"""Teleop actuators: stiffer/damped arm; gripper PD matches RL (zero stiffness/damping)."""

DOOSAN_E0509_ACTIONGRAPH_ACTUATORS = {
    "arm_lower": ImplicitActuatorCfg(
        joint_names_expr=["joint_[1-3]"],
        effort_limit_sim=11400.0,
        stiffness=0.0,
        damping=0.0,
    ),
    "arm_upper": ImplicitActuatorCfg(
        joint_names_expr=["joint_[4-6]"],
        effort_limit_sim=2400.0,
        stiffness=0.0,
        damping=0.0,
    ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["rh_r1", "rh_r2", "rh_l1", "rh_l2"],
        effort_limit_sim=GRIPPER_EFFORT_LIMIT_SIM,
        stiffness=0.0,
        damping=0.0,
    ),
}
"""Zero PD — ROS Action Graph ArticulationController drives joints (leader_to_isaac path)."""
