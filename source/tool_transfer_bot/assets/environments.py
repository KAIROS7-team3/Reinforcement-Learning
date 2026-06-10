import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets.rigid_object import RigidObjectCfg

from tool_transfer_bot.paths import asset, tool_model

# Self-contained flattened toolbox (meshes inlined, textures in assets/textures/).
_TOOLBOX_USD = asset("toolbox_rl_flat.usda")
# with_camera.usda /World/toolbox_with_handle xformOp:translate (USD spawn root)
_TOOLBOX_SPAWN_POS = (0.3877008091166735, 0.56212, 0.058999998658895464)
# Composed world pose of /toolbox/toolbox/toolbox (articulation root link / visuals)
TOOLBOX_BODY_WORLD_POS = (-0.653738, 0.741824, 0.060604)

# ---------------------------------------------------------------------------
# Toolbox (2 prismatic-joint drawers: drawer_joint, drawer_02_joint)
# Joint axis: Y, limits: 0 (closed) → -0.2 m (fully open)
# ---------------------------------------------------------------------------
TOOLBOX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_TOOLBOX_USD,
        activate_contact_sensors=False,
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
            "drawer_02_joint": 0.0,  # closed
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "drawers": ImplicitActuatorCfg(
            joint_names_expr=["drawer_joint", "drawer_02_joint"],
            effort_limit_sim=100.0,
            stiffness=0.0,    # passive — robot must push/pull
            damping=10.0,
        ),
    },
)
"""공구함 + 2-drawer ArticulationCfg. drawer_joint lowerLimit=-0.2 m (open), upperLimit=0 (closed)."""

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
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.1,
                angular_damping=0.2,
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
