import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets.rigid_object import RigidObjectCfg

from tool_transfer_bot.assets.doosan_e0509 import MAX_DEPENETRATION_VELOCITY
from tool_transfer_bot.assets.tool_spawn import ToolUsdFileCfg
from tool_transfer_bot.paths import asset, tool_model

# Self-contained flattened toolbox (meshes inlined, textures in assets/textures/).
_TOOLBOX_USD = asset("toolbox_rl_flat.usda")
# with_camera.usda /World/toolbox_with_handle xformOp:translate (USD spawn root)
_TOOLBOX_SPAWN_POS = (0.3877008091166735, 0.56212, 0.058999998658895464)
# Knob center in PhysX drawer body frame (``drawer_frame`` / teleop marker aim point).
# USD note: prim ``handle`` = toolbox top carry handle (unrelated). Drawer-front pull
# appearance is ``drawer/drawer`` mesh + texture, not a separate prim.
# offset = knob_center_in_drawer_link - localPos0
_DRAWER_HANDLE_OFFSET = (-5.090332e-05, -0.11836937, 0.022230722)

# ---------------------------------------------------------------------------
# Toolbox (top drawer: drawer_joint; bottom drawer merged as static geometry)
# Joint axis: Y, limits: 0 (closed) → -0.2 m (fully open)
# ---------------------------------------------------------------------------
TOOLBOX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_TOOLBOX_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=True,  # fixes toolbox body to world (non-kinematic root link)
        ),
    ),
    prim_path="{ENV_REGEX_NS}/toolbox_with_handle",
    articulation_root_prim_path="/toolbox/toolbox/toolbox",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=_TOOLBOX_SPAWN_POS,
        rot=(1.0, 0.0, 0.0, 0.0),  # w, x, y, z — no rotation
        joint_pos={
            "drawer_joint": 0.0,     # closed
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "drawers": ImplicitActuatorCfg(
            joint_names_expr=["drawer_joint"],
            effort_limit_sim=100.0,
            stiffness=0.0,    # passive — robot must push/pull
            damping=10.0,
        ),
    },
)
"""공구함 ArticulationCfg. drawer_joint lowerLimit=-0.2 m (open), upperLimit=0 (closed)."""

# ---------------------------------------------------------------------------
# 6종 공구 RigidObjectCfg (Task 2/3용)
# prim_path는 호출 시 .replace()로 지정
# ---------------------------------------------------------------------------
_TOOL_MASS_KG = {
    "Screw_Driver": 0.08,
    "Paper_Cutter": 0.07,
    "Husky_Socket_Wrench": 0.25,
    "Allen_Key_Tool_Assembly": 0.18,
    "Spanner_16mm": 0.12,
    "socket": 0.09,
}

_TOOL_USDZ = {
    "Screw_Driver": tool_model("Screw Driver.usdz"),
    "Paper_Cutter": tool_model("Paper Cutter.usdz"),
    "Husky_Socket_Wrench": tool_model("Husky Socket Wrench.usdz"),
    "Allen_Key_Tool_Assembly": tool_model("Allen Key Tool Assembly.usdz"),
    "Spanner_16mm": tool_model("Spanner 16mm.usdz"),
    "socket": tool_model("socket.usdz"),
}

TOOL_CFGS: dict[str, RigidObjectCfg] = {
    name: RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Tools/{name}",
        spawn=ToolUsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
                linear_damping=0.1,
                angular_damping=0.2,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=_TOOL_MASS_KG[name]),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    for name, usd_path in _TOOL_USDZ.items()
}
"""6종 공구 RigidObjectCfg dict. key = prim name (e.g. 'Screw_Driver')."""

# Convenience: 설계 문서의 tool_id → USD prim 이름 매핑
TOOL_ID_TO_PRIM = {
    "screwdriver": "Screw_Driver",
    "utility_knife": "Paper_Cutter",
    "ratchet_wrench": "Husky_Socket_Wrench",
    "multi_tool": "Allen_Key_Tool_Assembly",
    "spanner_16mm": "Spanner_16mm",
    "socket_19mm": "socket",
}
TOOL_IDS = list(TOOL_ID_TO_PRIM.keys())  # 길이 6, One-hot 순서

# ReturnTool MVP layout (table-local, meters). Tune in sim if needed.
# Staging on table surface (sim-tuned from robot-side default).
# Tool USDZ bottom is normalized to z=0 → place root ON table top, not in the air.
RETURN_TOOL_STAGING_SURFACE_Z = 0.0
RETURN_TOOL_STAGING_POS = (0.62, -0.08, RETURN_TOOL_STAGING_SURFACE_Z)
# 90° yaw on table (world Z) — handle easier to side-grasp from staging approach.
_STAGING_YAW_RAD = math.pi / 2.0
RETURN_TOOL_STAGING_ROT = (
    math.cos(_STAGING_YAW_RAD / 2.0),
    0.0,
    0.0,
    math.sin(_STAGING_YAW_RAD / 2.0),
)
# Place target offset in open drawer link frame (drawer fully open, 20 cm).
RETURN_TOOL_PLACE_OFFSET = (0.0, 0.06, 0.04)
RETURN_TOOL_DRAWER_OPEN_JOINT = -0.2

# Top-drawer floor playable XY (drawer link frame), from cavity_collision/floor in toolbox_rl_flat.usda.
# Unit-cube floor collision: scale (0.408, 0.172, 0.004), translate (0, 0.077, 0.066).
RETURN_TOOL_DRAWER_FLOOR_X_MIN = -0.2038
RETURN_TOOL_DRAWER_FLOOR_X_MAX = 0.2038
RETURN_TOOL_DRAWER_FLOOR_Y_MIN = -0.0090
RETURN_TOOL_DRAWER_FLOOR_Y_MAX = 0.1634
# Default inset so cube center stays inside walls (5 cm cube → half 2.5 cm).
RETURN_TOOL_DRAWER_TOOL_XY_MARGIN = 0.025

# ReturnTool reset / teleop home (degrees). Arm: J1 J2 J3 J4 J5 J6 = 0 0 90 0 90 0
RETURN_TOOL_HOME_JOINT_DEG: dict[str, float] = {
    "joint_1": 0.0,
    "joint_2": 0.0,
    "joint_3": 90.0,
    "joint_4": 0.0,
    "joint_5": 90.0,
    "joint_6": 0.0,
    "rh_r1": 0.0,
    "rh_r2": 0.0,
    "rh_l1": 0.0,
    "rh_l2": 0.0,
}


def return_tool_home_joint_pos_rad() -> dict[str, float]:
    import math

    return {
        name: math.radians(deg) if name.startswith("joint_") else deg
        for name, deg in RETURN_TOOL_HOME_JOINT_DEG.items()
    }


# BC teleop demos default (--tool_asset cube in collect_demos_teleop.py).
RETURN_TOOL_DEMO_CUBE_SIZE = 0.05


def return_tool_staging_cube_cfg(
    prim_path: str = "{ENV_REGEX_NS}/tool",
    edge: float = RETURN_TOOL_DEMO_CUBE_SIZE,
) -> RigidObjectCfg:
    """5 cm staging cube — matches ReturnTool BC demo collection."""
    cube_pos = (
        RETURN_TOOL_STAGING_POS[0],
        RETURN_TOOL_STAGING_POS[1],
        edge * 0.5,
    )
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=(edge, edge, edge),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=min(0.12, MAX_DEPENETRATION_VELOCITY),
                linear_damping=0.6,
                angular_damping=0.8,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.55, 0.85)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=cube_pos, rot=RETURN_TOOL_STAGING_ROT),
    )
