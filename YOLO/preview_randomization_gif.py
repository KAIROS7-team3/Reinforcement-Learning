#!/usr/bin/env python3
"""Isaac Sim에서 공구 domain randomization 프리뷰 — PNG N장 저장 (GIF는 직접 합성)."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import time
from pathlib import Path

YOLO_DIR = Path(__file__).resolve().parent
if str(YOLO_DIR) not in sys.path:
    sys.path.insert(0, str(YOLO_DIR))

from generate_realscene_replicator_dataset import (  # noqa: E402
    DEFAULT_CAMERA_PATH,
    DEFAULT_DRAWER_JOINT,
    DEFAULT_DRAWER_OPEN_RANGE,
    DEFAULT_ROBOT_PATH,
    DEFAULT_SCENE_USD,
    DEFAULT_TABLE_SURFACE_PATH,
    DEFAULT_TOOLBOX_ROOT,
    DEFAULT_TOOLS_ROOT,
    DEFAULT_UP_AXIS,
    DEFAULT_PLANE_U_AXIS,
    DEFAULT_PLANE_V_AXIS,
    PER_FRAME_APP_UPDATES,
    PRE_CAPTURE_APP_UPDATES,
    RENDER_RT_SUBFRAMES,
    PlacementAxes,
    apply_semantic_labels,
    capture_tool_base_matrices,
    cleanup_conflicting_replicator_prims,
    compute_staging_bounds_from_table,
    configure_renderer_for_still_images,
    create_flat_capture_camera,
    estimate_staging_surface_height,
    hide_prim,
    prepare_scene_for_offline_capture,
    randomize_dome_light,
    randomize_drawer,
    randomize_tool_layout,
    reset_render_accumulation,
    resolve_camera_path,
    setup_drawer_controller,
    setup_tools,
    wait_for_stage_ready,
)

DEFAULT_OUTPUT = YOLO_DIR.parent / "docs" / "assets" / "sdg_randomization_preview"
DEFAULT_INTERVAL_S = 0.5
DEFAULT_NUM_FRAMES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="공구 랜덤화 프리뷰 — PNG 저장")
    parser.add_argument("--scene-usd", type=Path, default=DEFAULT_SCENE_USD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_S,
        help="프레임 간 대기 (초). GUI에서 변화 보기용",
    )
    parser.add_argument("--resolution", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-tools", type=int, default=1)
    parser.add_argument("--max-tools", type=int, default=6)
    parser.add_argument("--tight-cluster-prob", type=float, default=0.5)
    parser.add_argument("--drawer-open-prob", type=float, default=0.35)
    parser.add_argument("--tool-min-gap", type=float, default=0.003)
    parser.add_argument("--tool-spread-pad", type=float, default=0.03)
    parser.add_argument("--tight-max-gap", type=float, default=0.012)
    parser.add_argument("--settle-updates", type=int, default=PER_FRAME_APP_UPDATES)
    parser.add_argument("--hide-toolbox", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--headless", action="store_true", help="헤드리스 (기본: GUI)")
    return parser.parse_args()


def focus_viewport_on_camera(camera_path: str) -> None:
    try:
        import omni.kit.viewport.utility as vp_utils

        viewport = vp_utils.get_active_viewport()
        if viewport is not None:
            viewport.camera_path = camera_path
            print(f"✓ viewport camera → {camera_path}")
    except Exception as exc:
        print(f"⚠ viewport camera 설정 실패: {exc}")


def setup_rgb_writer(rep, camera_path: str, work_dir: Path, resolution: int):
    render_product = rep.create.render_product(camera_path, (resolution, resolution))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(work_dir), rgb=True)
    writer.attach(render_product)
    return writer


def pump_until(simulation_app, deadline: float) -> None:
    while time.monotonic() < deadline and simulation_app.is_running():
        simulation_app.update()


def copy_latest_rgb(work_dir: Path, out_path: Path) -> None:
    rgb_files = sorted(work_dir.glob("rgb_*.png"))
    if not rgb_files:
        raise RuntimeError("Replicator RGB capture failed")
    shutil.copy2(rgb_files[-1], out_path)


def main() -> None:
    args = parse_args()
    args.scene_usd = args.scene_usd.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.scene_usd.is_file():
        raise FileNotFoundError(f"Scene USD not found: {args.scene_usd}")

    random.seed(args.seed)
    work_dir = args.output_dir / "_capture_tmp"
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    work_dir.mkdir(parents=True)

    placement_axes = PlacementAxes(DEFAULT_UP_AXIS, DEFAULT_PLANE_U_AXIS, DEFAULT_PLANE_V_AXIS)

    print("=" * 56)
    print("SDG Randomization Preview → PNG")
    print("=" * 56)
    print(f"Scene:  {args.scene_usd}")
    print(f"Frames: {args.num_frames}")
    print(f"Output: {args.output_dir}")
    print()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {"headless": args.headless, "width": 1600, "height": 900}
    )

    import carb
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Usd, UsdGeom

    rep.orchestrator.set_capture_on_play(False)
    configure_renderer_for_still_images(carb)

    omni.usd.get_context().open_stage(str(args.scene_usd))
    wait_for_stage_ready(simulation_app)
    stage = omni.usd.get_context().get_stage()
    cleanup_conflicting_replicator_prims(stage)
    wait_for_stage_ready(simulation_app, max_updates=60)

    camera_prim_path = resolve_camera_path(stage, DEFAULT_CAMERA_PATH)
    camera_prim_path = create_flat_capture_camera(stage, camera_prim_path)

    tool_paths = setup_tools(stage, DEFAULT_TOOLS_ROOT)
    staging_bounds = compute_staging_bounds_from_table(stage, DEFAULT_TABLE_SURFACE_PATH, placement_axes)
    surface_height = estimate_staging_surface_height(
        stage, DEFAULT_TABLE_SURFACE_PATH, "/World/table", placement_axes
    )
    tool_base_matrices = capture_tool_base_matrices(stage, tool_paths, UsdGeom)

    hide_prim(stage, DEFAULT_ROBOT_PATH)
    if args.hide_toolbox:
        hide_prim(stage, DEFAULT_TOOLBOX_ROOT)
    apply_semantic_labels(stage, tool_paths)
    prepare_scene_for_offline_capture(stage, tool_paths)
    drawer_controller = setup_drawer_controller(stage, DEFAULT_DRAWER_JOINT)

    for _ in range(PRE_CAPTURE_APP_UPDATES):
        simulation_app.update()

    writer = setup_rgb_writer(rep, camera_prim_path, work_dir, args.resolution)

    if not args.headless:
        for _ in range(20):
            simulation_app.update()
        focus_viewport_on_camera(camera_prim_path)

    interval_s = max(0.1, args.interval)
    saved: list[Path] = []

    for frame_idx in range(args.num_frames):
        loop_start = time.monotonic()

        num_visible = random.randint(args.min_tools, args.max_tools)
        visible_tools = random.sample(list(tool_paths.keys()), k=num_visible)
        tight_cluster = num_visible >= 2 and random.random() < args.tight_cluster_prob

        randomize_tool_layout(
            stage,
            tool_paths,
            visible_tools,
            staging_bounds,
            surface_height,
            tool_base_matrices,
            placement_axes,
            Usd,
            UsdGeom,
            min_gap=args.tool_min_gap,
            spread_pad=args.tool_spread_pad,
            tight_cluster=tight_cluster,
            tight_max_gap=args.tight_max_gap,
        )
        randomize_drawer(drawer_controller, args.drawer_open_prob, DEFAULT_DRAWER_OPEN_RANGE)
        randomize_dome_light(stage)

        reset_render_accumulation(carb)
        for _ in range(args.settle_updates):
            simulation_app.update()
        rep.orchestrator.step(rt_subframes=RENDER_RT_SUBFRAMES)
        rep.orchestrator.wait_until_complete()
        for _ in range(8):
            simulation_app.update()

        out_path = args.output_dir / f"frame_{frame_idx:02d}.png"
        copy_latest_rgb(work_dir, out_path)
        saved.append(out_path)

        layout_tag = "tight" if tight_cluster else "spread"
        print(
            f"  [{frame_idx + 1:02d}/{args.num_frames}] "
            f"tools={num_visible} layout={layout_tag} → {out_path.name}"
        )

        pump_until(simulation_app, loop_start + interval_s)

    writer.detach()
    rep.orchestrator.wait_until_complete()
    shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\n✓ Saved {len(saved)} images → {args.output_dir}")
    for path in saved:
        print(f"    {path.name}")

    if args.keep_open and not args.headless:
        print("\nGUI 유지 중 — 창 닫으면 종료")
        while simulation_app.is_running():
            simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()
