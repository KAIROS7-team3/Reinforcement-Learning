"""ROS2 Action Graph teleop for Isaac Lab (same path as leader_to_isaac.md).

Creates OnPlaybackTick → ROS2SubscribeJointState → ArticulationController so
``isaac/joint_command`` from ``leader_to_isaac.py`` drives the robot without
Isaac Lab PD / kinematic overrides fighting the command.
"""

from __future__ import annotations

_ACTION_GRAPH_EXTENSIONS = (
    "isaacsim.ros2.bridge",
    "isaacsim.core.nodes",
)


def ensure_ros2_teleop_extensions() -> None:
    """Load OmniGraph node types required for leader_to_isaac Action Graph."""
    from isaacsim.core.utils.extensions import enable_extension

    for name in _ACTION_GRAPH_EXTENSIONS:
        ok = enable_extension(name)
        if not ok:
            raise RuntimeError(
                f"Failed to enable extension '{name}'. "
                "Isaac Sim ROS2 bridge may be missing from this install."
            )
        print(f"[INFO] Enabled extension: {name}", flush=True)

    # Give Kit a few ticks to register OmniGraph node types.
    import omni.kit.app

    app = omni.kit.app.get_app()
    for _ in range(5):
        app.update()


def resolve_robot_controller_prim(robot) -> str:
    """Return USD path for IsaacArticulationController (…/e0509/root_joint)."""
    import isaaclab.sim as sim_utils
    import omni.usd

    first = sim_utils.find_first_matching_prim(robot.cfg.prim_path)
    if first is None:
        raise RuntimeError(f"Failed to resolve robot prim: {robot.cfg.prim_path!r}")
    base = first.GetPath().pathString
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage not available")
    for suffix in ("/root_joint", ""):
        path = f"{base}{suffix}"
        if stage.GetPrimAtPath(path).IsValid():
            return path
    raise RuntimeError(f"No articulation root under {base}")


def setup_ros2_joint_teleop_graph(
    target_prim: str,
    graph_path: str = "/ActionGraph",
    ros_domain_id: int = 71,
    sub_topic: str = "isaac/joint_command",
    pub_topic: str = "isaac/joint_states",
    force: bool = False,
    publish_joint_states: bool = False,
) -> str:
    """Create or reuse world Action Graph (leader_to_isaac compatible).

    Isaac Lab + fabric: keep ``publish_joint_states=False`` (default) and publish
    ``isaac/joint_states`` from Python instead — avoids PhysX device -1 errors in
    ``ROS2PublishJointState``.

    Returns the graph USD path.
    """
    ensure_ros2_teleop_extensions()

    import omni.graph.core as og
    import omni.usd
    import usdrt.Sdf

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("USD stage not available")

    if stage.GetPrimAtPath(graph_path).IsValid():
        if not force:
            print(f"[INFO] Action Graph already exists: {graph_path}", flush=True)
            return graph_path
        og.Controller.destroy_graph(graph_path)
        print(f"[INFO] Removed old Action Graph: {graph_path}", flush=True)

    keys = og.Controller.Keys
    nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
        ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
        ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
    ]
    set_values = [
        ("ROS2Context.inputs:domain_id", int(ros_domain_id)),
        ("SubscribeJointState.inputs:topicName", sub_topic),
        (
            "ArticulationController.inputs:targetPrim",
            [usdrt.Sdf.Path(target_prim)],
        ),
    ]
    connect = [
        ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
        ("ROS2Context.outputs:context", "SubscribeJointState.inputs:context"),
        (
            "SubscribeJointState.outputs:positionCommand",
            "ArticulationController.inputs:positionCommand",
        ),
        (
            "SubscribeJointState.outputs:velocityCommand",
            "ArticulationController.inputs:velocityCommand",
        ),
        (
            "SubscribeJointState.outputs:effortCommand",
            "ArticulationController.inputs:effortCommand",
        ),
        (
            "SubscribeJointState.outputs:jointNames",
            "ArticulationController.inputs:jointNames",
        ),
    ]
    if publish_joint_states:
        nodes[1:1] = [("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime")]
        nodes.insert(2, ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"))
        set_values.extend(
            [
                ("PublishJointState.inputs:topicName", pub_topic),
                (
                    "PublishJointState.inputs:targetPrim",
                    [usdrt.Sdf.Path(target_prim)],
                ),
            ]
        )
        connect = [
            ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
            *connect,
            ("ROS2Context.outputs:context", "PublishJointState.inputs:context"),
            ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
        ]

    edit_spec = {
        keys.CREATE_NODES: nodes,
        keys.SET_VALUES: set_values,
        keys.CONNECT: connect,
    }

    import omni.kit.app

    app = omni.kit.app.get_app()
    last_err: Exception | None = None
    for attempt in range(12):
        try:
            og.Controller.edit({"graph_path": graph_path, "evaluator_name": "execution"}, edit_spec)
            last_err = None
            break
        except og.OmniGraphError as exc:
            last_err = exc
            if "unrecognized type" not in str(exc).lower() or attempt >= 11:
                raise
            app.update()
    if last_err is not None:
        raise last_err
    print(
        f"[INFO] Action Graph teleop ready: {graph_path} → {target_prim} "
        f"(domain={ros_domain_id}, sub={sub_topic}, "
        f"pub_joint_states={'graph' if publish_joint_states else 'python'})",
        flush=True,
    )
    return graph_path
