#!/usr/bin/env python3
"""
실제 workbench USD(with_camera_backup.usd)를 로드해 Replicator 합성 데이터를 생성합니다.

기존 ivory 테이블 generator와 달리:
  - 실제 table.usdz + toolbox + RSD455 카메라 구도 유지
  - staging area에 1~6개 공구 랜덤 배치 (항상 6개 아님)
  - 로봇은 캡처 시 숨김
  - 공구 semantic label 런타임 부착 (instance segmentation용)
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_USD = Path("/home/user/Desktop/with_camera_backup.usd")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "YOLO" / "replicator_output" / "realscene_topview_rsd455"
DEFAULT_CAMERA_PATH = "/World/table/Realsense/RSD455/Camera_OmniVision_OV9782_Color"
DEFAULT_TOOLS_ROOT = "/World/Tools"
DEFAULT_ROBOT_PATH = "/World/e0509"
DEFAULT_DRAWER_JOINT = "/World/toolbox_with_handle/toolbox/toolbox/drawer_joint"
DEFAULT_TOOLBOX_ROOT = "/World/toolbox_with_handle"
DRAWER_JOINT_FALLBACKS = (
    "/World/toolbox_with_handle/toolbox/toolbox/drawer_joint",
    "/World/toolbox_with_handle/toolbox/toolbox/toolbox/drawer_joint",
)
DEFAULT_DRAWER_OPEN_RANGE = (-0.18, -0.05)  # closed=0, full open=-0.2 m (Y prismatic)
DEFAULT_TABLE_SURFACE_PATH = "/World/table/MeshInstance"
# 왼쪽 스테이션(회색)은 오른쪽 책상 상판보다 약간 낮음
DEFAULT_STAGING_SURFACE_OFFSET = -0.02
FLAT_CAPTURE_CAMERA_PATH = "/World/YOLOCaptureCamera"
RENDER_RT_SUBFRAMES = 16
PRE_CAPTURE_APP_UPDATES = 60
PER_FRAME_APP_UPDATES = 24
WRITER_WARMUP_FRAMES = 12
RGB_NOISE_STD_RANGE = (0.0, 0.018)
RGB_BRIGHTNESS_RANGE = (0.88, 1.14)
RGB_CONTRAST_RANGE = (0.90, 1.12)
RGB_GAMMA_RANGE = (0.92, 1.10)
DOME_LIGHT_INTENSITY_RANGE = (120.0, 560.0)
DOME_LIGHT_COLOR_RANGE = (0.96, 1.04)
EXPOSURE_RANGE = (-0.25, 0.35)
AXIS_TO_INDEX = {"x": 0, "y": 1, "z": 2}
DEFAULT_UP_AXIS = "z"
DEFAULT_PLANE_U_AXIS = "x"
DEFAULT_PLANE_V_AXIS = "y"
SURFACE_CLEARANCE_M = 0.002
# 잔상 방지: 숨긴 공구를 스테이징 영역 밖으로 물리 이동
TOOL_STASH_U = -8.0
TOOL_STASH_V = -8.0
TOOL_STASH_HEIGHT = -0.35
RSD455_CAMERA_INTRINSICS = {
    "focalLength": 1.93,
    "horizontalAperture": 3.896,
    "verticalAperture": 2.453,
    "clippingRange": (0.01, 10_000.0),
}

# with_camera_backup.usd의 /World/Tools/* prim 이름 -> synthetic class label
SCENE_TOOLS = [
    {"prim_name": "Allen_Key_Tool_Assembly", "label": "allen_key_tool_assembly"},
    {"prim_name": "Husky_Socket_Wrench", "label": "husky_socket_wrench"},
    {"prim_name": "Paper_Cutter", "label": "paper_cutter"},
    {"prim_name": "Screw_Driver", "label": "screw_driver"},
    {"prim_name": "Spanner_16mm", "label": "spanner_16mm"},
    {"prim_name": "socket", "label": "socket"},
]


@dataclass(frozen=True)
class PlacementAxes:
    up_axis: str
    u_axis: str
    v_axis: str

    def __post_init__(self) -> None:
        axes = {self.up_axis, self.u_axis, self.v_axis}
        if len(axes) != 3:
            raise ValueError("--up-axis, --plane-u-axis, --plane-v-axis must be distinct")

    @property
    def up_idx(self) -> int:
        return AXIS_TO_INDEX[self.up_axis]

    @property
    def u_idx(self) -> int:
        return AXIS_TO_INDEX[self.u_axis]

    @property
    def v_idx(self) -> int:
        return AXIS_TO_INDEX[self.v_axis]

# staging area (A4 + central table) — manual override용.
DEFAULT_STAGING_BOUNDS = {
    "min_x": -0.17,
    "max_x": 0.42,
    "min_y": -0.18,
    "max_y": 0.22,
}
DEFAULT_TOOL_MIN_GAP = 0.003
DEFAULT_TOOL_SPREAD_PAD = 0.03
DEFAULT_TIGHT_CLUSTER_PROB = 0.35
DEFAULT_TIGHT_MAX_GAP = 0.012


def resolve_output_dir(output_dir: Path) -> Path:
    if output_dir.is_absolute():
        resolved = output_dir
    else:
        resolved = (PROJECT_ROOT / output_dir).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def wait_for_stage_ready(simulation_app, max_updates: int = 600) -> None:
    """payload/texture 로드 완료까지 SimulationApp을 pump합니다."""
    try:
        from isaacsim.core.utils.stage import is_stage_loading
    except ImportError:
        from omni.isaac.core.utils.stage import is_stage_loading

    for _ in range(max_updates):
        if not is_stage_loading():
            break
        simulation_app.update()
    for _ in range(30):
        simulation_app.update()


def cleanup_conflicting_replicator_prims(stage) -> None:
    """GUI 세션에서 저장된 Replicator graph만 제거. viewport RenderProduct는 Hydra가 참조하므로 유지."""
    paths = [
        "/Replicator",
        "/WriterOrchestrator",
        "/Orchestrator",
        "/Render/OmniverseKit/HydraTextures/Replicator",
    ]
    removed = []
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            stage.RemovePrim(path)
            removed.append(path)
    if removed:
        print("✓ removed conflicting stage prims:")
        for path in removed:
            print(f"    {path}")


def setup_replicator_writer(camera_path: str, output_dir: Path, resolution: int):
    """Replicator writer를 stage에 직접 연결 (동작 확인된 ivory generator와 동일 패턴)."""
    import omni.replicator.core as rep

    render_product = rep.create.render_product(camera_path, (resolution, resolution))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=str(output_dir),
        rgb=True,
        bounding_box_2d_tight=True,
        semantic_segmentation=True,
        instance_segmentation=True,
        camera_params=True,
    )
    writer.attach(render_product)
    return writer


def warn_if_other_isaac_processes() -> None:
    import subprocess

    try:
        result = subprocess.run(
            ["pgrep", "-af", "isaacsim|Isaac-Sim|omni.kit"],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return
    lines = [line for line in result.stdout.splitlines() if "generate_realscene_replicator_dataset" not in line]
    if lines:
        print("⚠ 다른 Isaac Sim/Kit 프로세스가 실행 중입니다. GUI Isaac Sim을 종료한 뒤 재시도하세요.")
        for line in lines[:3]:
            print(f"    {line}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-scene USD Replicator dataset generator")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-frames", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scene-usd", type=Path, default=DEFAULT_SCENE_USD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--camera-path", type=str, default=DEFAULT_CAMERA_PATH)
    parser.add_argument("--tools-root", type=str, default=DEFAULT_TOOLS_ROOT)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--min-tools", type=int, default=1, help="프레임당 최소 visible 공구 수")
    parser.add_argument("--max-tools", type=int, default=6, help="프레임당 최대 visible 공구 수")
    parser.add_argument(
        "--tool-min-gap",
        type=float,
        default=DEFAULT_TOOL_MIN_GAP,
        help="공구 bbox 간 최소 간격(m). 겹침 방지용",
    )
    parser.add_argument(
        "--tool-spread-pad",
        type=float,
        default=DEFAULT_TOOL_SPREAD_PAD,
        help="흩어진 배치에서 후보 위치 샘플링 반경(m)",
    )
    parser.add_argument(
        "--tight-cluster-prob",
        type=float,
        default=DEFAULT_TIGHT_CLUSTER_PROB,
        help="프레임마다 공구를 서로 거의 붙게 배치할 확률 (2개 이상일 때)",
    )
    parser.add_argument(
        "--tight-max-gap",
        type=float,
        default=DEFAULT_TIGHT_MAX_GAP,
        help="tight cluster 배치 시 공구 간 최대 간격(m)",
    )
    parser.add_argument("--drawer-joint-path", type=str, default=DEFAULT_DRAWER_JOINT, help="서랍 prismatic joint prim 경로")
    parser.add_argument("--drawer-open-prob", type=float, default=0.35, help="서랍 열림 확률")
    parser.add_argument(
        "--drawer-open-range",
        type=float,
        nargs=2,
        default=DEFAULT_DRAWER_OPEN_RANGE,
        metavar=("MIN_M", "MAX_M"),
        help="열릴 때 joint 위치(m). 기본 음수(-0.18~-0.05). 양수 입력 시 자동으로 음수 변환",
    )
    parser.add_argument("--hide-robot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--hide-toolbox",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="공구함(toolbox_with_handle) 숨김 — 책상+공구만 보이게",
    )
    parser.add_argument("--staging-min-x", type=float, default=DEFAULT_STAGING_BOUNDS["min_x"])
    parser.add_argument("--staging-max-x", type=float, default=DEFAULT_STAGING_BOUNDS["max_x"])
    parser.add_argument("--staging-min-y", type=float, default=DEFAULT_STAGING_BOUNDS["min_y"], help="평면 두 번째 축 범위 최소값")
    parser.add_argument("--staging-max-y", type=float, default=DEFAULT_STAGING_BOUNDS["max_y"], help="평면 두 번째 축 범위 최대값")
    parser.add_argument(
        "--auto-staging-bounds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="테이블 bbox에서 staging 영역 자동 계산 (기본 ON)",
    )
    parser.add_argument(
        "--staging-surface-offset",
        type=float,
        default=DEFAULT_STAGING_SURFACE_OFFSET,
        help="오른쪽 책상 상판 대비 왼쪽 스테이션 Z 오프셋 (m). 기본 -0.02",
    )
    parser.add_argument(
        "--flat-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="RSD455 world pose를 /World/YOLOCaptureCamera로 복사 (중첩 scale 회피)",
    )
    parser.add_argument(
        "--writer-warmup-frames",
        type=int,
        default=WRITER_WARMUP_FRAMES,
        help="writer attach 직후 버릴 warmup capture 수 (검은 프레임 방지)",
    )
    parser.add_argument(
        "--settle-updates",
        type=int,
        default=PER_FRAME_APP_UPDATES,
        help="공구/조명 랜덤화 후 캡처 전 SimulationApp update 횟수 (잔상 완화)",
    )
    parser.add_argument("--table-path", type=str, default="/World/table")
    parser.add_argument("--table-surface-path", type=str, default=DEFAULT_TABLE_SURFACE_PATH)
    parser.add_argument("--up-axis", choices=("x", "y", "z"), default=DEFAULT_UP_AXIS)
    parser.add_argument("--plane-u-axis", choices=("x", "y", "z"), default=DEFAULT_PLANE_U_AXIS)
    parser.add_argument("--plane-v-axis", choices=("x", "y", "z"), default=DEFAULT_PLANE_V_AXIS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = resolve_output_dir(args.output_dir)
    args.scene_usd = args.scene_usd.resolve()
    random.seed(args.seed)

    if not args.scene_usd.is_file():
        raise FileNotFoundError(f"Scene USD not found: {args.scene_usd}")
    if args.min_tools < 1 or args.max_tools > len(SCENE_TOOLS) or args.min_tools > args.max_tools:
        raise ValueError("Invalid --min-tools / --max-tools range")
    if args.tool_min_gap < 0.0:
        raise ValueError("--tool-min-gap must be >= 0")
    if args.tight_max_gap < args.tool_min_gap:
        raise ValueError("--tight-max-gap must be >= --tool-min-gap")
    if not 0.0 <= args.tight_cluster_prob <= 1.0:
        raise ValueError("--tight-cluster-prob must be in [0, 1]")
    placement_axes = PlacementAxes(args.up_axis, args.plane_u_axis, args.plane_v_axis)

    staging_bounds = {
        "min_u": args.staging_min_x,
        "max_u": args.staging_max_x,
        "min_v": args.staging_min_y,
        "max_v": args.staging_max_y,
    }
    if args.auto_staging_bounds:
        staging_bounds = None  # resolved after stage load

    print("=" * 50)
    print("Real-scene Replicator dataset generator")
    print("=" * 50)
    print(f"Scene USD: {args.scene_usd}")
    print(f"Camera: {args.camera_path}")
    print(f"Frames: {args.num_frames}")
    print(f"Tools/frame: {args.min_tools}~{args.max_tools}")
    print(
        f"Tool layout: spread_pad={args.tool_spread_pad:.3f} m, min_gap={args.tool_min_gap:.3f} m, "
        f"tight_prob={args.tight_cluster_prob:.2f}, tight_max_gap={args.tight_max_gap:.3f} m"
    )
    print(f"Auto staging bounds: {args.auto_staging_bounds}")
    print(f"Flat capture camera: {args.flat_camera}")
    print(f"Writer warmup frames: {args.writer_warmup_frames}")
    print(f"Settle updates before capture: {args.settle_updates}")
    print(f"Placement axes: plane=({placement_axes.u_axis}, {placement_axes.v_axis}), up={placement_axes.up_axis}")
    if staging_bounds:
        print(f"Staging bounds: {staging_bounds}")
    print(f"Output: {args.output_dir}")
    print()
    warn_if_other_isaac_processes()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "width": 1280,
            "height": 720,
        }
    )

    import carb
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Usd, UsdGeom

    rep.orchestrator.set_capture_on_play(False)
    configure_renderer_for_still_images(carb)

    print("[Step 1] Scene USD 로드...")
    omni.usd.get_context().open_stage(str(args.scene_usd))
    wait_for_stage_ready(simulation_app)
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Failed to open scene stage")

    cleanup_conflicting_replicator_prims(stage)
    wait_for_stage_ready(simulation_app, max_updates=60)

    camera_prim_path = resolve_camera_path(stage, args.camera_path)
    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    if not camera_prim.IsValid():
        raise RuntimeError(f"Camera prim not found: {camera_prim_path}")
    print(f"✓ source camera: {camera_prim_path}")
    log_camera_world_pose(stage, camera_prim_path)

    if args.flat_camera:
        camera_prim_path = create_flat_capture_camera(stage, camera_prim_path)
        print(f"✓ capture camera: {camera_prim_path}")

    tool_paths = setup_tools(stage, args.tools_root)
    if args.auto_staging_bounds:
        staging_bounds = compute_staging_bounds_from_table(stage, args.table_surface_path, placement_axes)
        print(f"✓ auto staging bounds: {staging_bounds}")
    table_surface_height = estimate_staging_surface_height(
        stage, args.table_surface_path, args.table_path, placement_axes,
        surface_offset=args.staging_surface_offset,
    )
    print(f"✓ staging surface {placement_axes.up_axis.upper()} ≈ {table_surface_height:.4f} m")
    tool_base_matrices = capture_tool_base_matrices(stage, tool_paths, UsdGeom)

    if args.hide_robot:
        hide_prim(stage, DEFAULT_ROBOT_PATH)
    if args.hide_toolbox:
        hide_prim(stage, DEFAULT_TOOLBOX_ROOT)

    apply_semantic_labels(stage, tool_paths)
    print(f"✓ {len(tool_paths)} tools ready with semantic labels")

    drawer_controller = setup_drawer_controller(stage, args.drawer_joint_path)

    prepare_scene_for_offline_capture(stage, tool_paths)
    print("✓ physics disabled for offline capture")

    print("[Step 2] Render/material warmup (writer attach 전)...")
    for _ in range(PRE_CAPTURE_APP_UPDATES):
        simulation_app.update()

    print("[Step 3] Replicator writer 설정...")
    writer = setup_replicator_writer(camera_prim_path, args.output_dir, args.resolution)
    print("✓ Replicator writer attached")

    warmup = max(0, args.writer_warmup_frames)
    if warmup > 0:
        print(f"[Step 4] Writer warmup ({warmup} frames, 출력에서 제외)...")
        warmup_visible = list(tool_paths.keys())
        randomize_tool_layout(
            stage,
            tool_paths,
            warmup_visible,
            staging_bounds,
            table_surface_height,
            tool_base_matrices,
            placement_axes,
            Usd,
            UsdGeom,
            min_gap=args.tool_min_gap,
            spread_pad=args.tool_spread_pad,
            tight_cluster=False,
            tight_max_gap=args.tight_max_gap,
        )
        randomize_dome_light(stage)
        for _ in range(PRE_CAPTURE_APP_UPDATES):
            simulation_app.update()
        for _ in range(warmup):
            reset_render_accumulation(carb)
            for _ in range(args.settle_updates):
                simulation_app.update()
            rep.orchestrator.step(rt_subframes=RENDER_RT_SUBFRAMES)
            rep.orchestrator.wait_until_complete()
            for _ in range(3):
                simulation_app.update()

    print(f"\n[Capture] {args.num_frames} frames...")
    for frame_idx in range(args.num_frames):
        num_visible = random.randint(args.min_tools, args.max_tools)
        visible_tools = random.sample(list(tool_paths.keys()), k=num_visible)
        tight_cluster = len(visible_tools) >= 2 and random.random() < args.tight_cluster_prob
        randomize_tool_layout(
            stage,
            tool_paths,
            visible_tools,
            staging_bounds,
            table_surface_height,
            tool_base_matrices,
            placement_axes,
            Usd,
            UsdGeom,
            min_gap=args.tool_min_gap,
            spread_pad=args.tool_spread_pad,
            tight_cluster=tight_cluster,
            tight_max_gap=args.tight_max_gap,
        )
        randomize_drawer(drawer_controller, args.drawer_open_prob, args.drawer_open_range)
        randomize_dome_light(stage)
        reset_render_accumulation(carb)
        for _ in range(args.settle_updates):
            simulation_app.update()
        rep.orchestrator.step(rt_subframes=RENDER_RT_SUBFRAMES)
        rep.orchestrator.wait_until_complete()
        for _ in range(3):
            simulation_app.update()
        layout_tag = "tight" if tight_cluster else "spread"
        print(
            f"  frame {frame_idx + 1}/{args.num_frames} "
            f"(visible={num_visible}, layout={layout_tag})",
            end="\r",
        )

    print()
    print("Saving...")
    rep.orchestrator.wait_until_complete()
    try:
        writer.detach()
    except Exception:
        pass
    if warmup > 0:
        trim_warmup_frames(args.output_dir, warmup, args.num_frames)
    add_camera_noise_to_rgb(args.output_dir, args.seed)
    print_output_summary(args.output_dir, args.num_frames)

    print(f"\n✓ Done: {args.output_dir}")
    simulation_app.close()


def log_camera_world_pose(stage, camera_path: str) -> None:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(camera_path)
    if not prim.IsValid():
        return
    cache = UsdGeom.XformCache()
    world_xf = cache.GetLocalToWorldTransform(prim)
    t = world_xf.ExtractTranslation()
    print(f"  camera world position: ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) m")


def create_flat_capture_camera(stage, source_camera_path: str) -> str:
    """RSD455 중첩 xform(scale 1000 등)을 피하기 위해 world pose를 flat camera로 복사."""
    from pxr import UsdGeom

    source = stage.GetPrimAtPath(source_camera_path)
    if not source.IsValid() or not source.IsA(UsdGeom.Camera):
        raise RuntimeError(f"Invalid source camera: {source_camera_path}")

    cache = UsdGeom.XformCache()
    world_xf = cache.GetLocalToWorldTransform(source)

    capture_path = FLAT_CAPTURE_CAMERA_PATH
    existing = stage.GetPrimAtPath(capture_path)
    if existing.IsValid():
        stage.RemovePrim(capture_path)

    cam = UsdGeom.Camera.Define(stage, capture_path)
    src_cam = UsdGeom.Camera(source)

    focal = src_cam.GetFocalLengthAttr().Get()
    cam.GetFocalLengthAttr().Set(focal if focal else RSD455_CAMERA_INTRINSICS["focalLength"])

    h_ap = src_cam.GetHorizontalApertureAttr().Get()
    v_ap = src_cam.GetVerticalApertureAttr().Get()
    cam.GetHorizontalApertureAttr().Set(h_ap if h_ap else RSD455_CAMERA_INTRINSICS["horizontalAperture"])
    cam.GetVerticalApertureAttr().Set(v_ap if v_ap else RSD455_CAMERA_INTRINSICS["verticalAperture"])

    clip = src_cam.GetClippingRangeAttr().Get()
    cam.GetClippingRangeAttr().Set(clip if clip else RSD455_CAMERA_INTRINSICS["clippingRange"])

    xformable = UsdGeom.Xformable(cam.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(world_xf)

    log_camera_world_pose(stage, capture_path)
    return capture_path


def compute_staging_bounds_from_table(stage, table_path: str, axes: PlacementAxes) -> dict[str, float]:
    """테이블 surface prim bbox에서 공구 배치 영역 추정."""
    from pxr import Usd, UsdGeom

    table_prim = stage.GetPrimAtPath(table_path)
    if not table_prim.IsValid():
        return {
            "min_u": DEFAULT_STAGING_BOUNDS["min_x"],
            "max_u": DEFAULT_STAGING_BOUNDS["max_x"],
            "min_v": DEFAULT_STAGING_BOUNDS["min_y"],
            "max_v": DEFAULT_STAGING_BOUNDS["max_y"],
        }

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    aligned = bbox_cache.ComputeWorldBound(table_prim).ComputeAlignedBox()
    min_pt, max_pt = aligned.GetMin(), aligned.GetMax()
    u_min, u_max = float(min_pt[axes.u_idx]), float(max_pt[axes.u_idx])
    v_min, v_max = float(min_pt[axes.v_idx]), float(max_pt[axes.v_idx])
    u_size = u_max - u_min
    v_size = v_max - v_min
    margin = 0.08
    side_surface = stage.GetPrimAtPath("/World/table/MeshInstance_6")
    if side_surface.IsValid():
        side_box = bbox_cache.ComputeWorldBound(side_surface).ComputeAlignedBox()
        side_min = side_box.GetMin()
        # The white board/toolbox support starts at a larger plane-v value.
        # Keep tools on the gray desk side only.
        if float(side_min[axes.v_idx]) > v_min:
            v_max = min(v_max, float(side_min[axes.v_idx]) - margin)
    return {
        "min_u": u_min + margin,
        "max_u": u_max - margin,
        "min_v": v_min + margin,
        "max_v": v_max,
    }


def trim_warmup_frames(output_dir: Path, warmup_count: int, expected_frames: int) -> None:
    """Writer warmup으로 생성된 앞쪽 프레임을 제외하고 0부터 재번호.

    Isaac/Replicator writer는 모든 orchestrator step을 항상 같은 수의 output index로
    flush하지 않을 수 있다. 그래서 앞 warmup_count개를 고정 삭제하지 않고, 실제로
    저장된 index 중 마지막 expected_frames개를 최종 capture로 유지한다.
    """
    import re

    if warmup_count <= 0 or expected_frames <= 0 or not output_dir.exists():
        return

    frame_pattern = re.compile(r"^(.+)_(\d{4})(.*)$")

    def frame_index(path: Path) -> int | None:
        match = frame_pattern.match(path.name)
        return int(match.group(2)) if match else None

    indices = sorted(
        {
            idx
            for path in output_dir.iterdir()
            if path.is_file() and (idx := frame_index(path)) is not None
        }
    )
    if not indices:
        return

    keep_indices = set(indices[-expected_frames:])
    deleted = 0
    for path in list(output_dir.iterdir()):
        if not path.is_file():
            continue
        idx = frame_index(path)
        if idx is not None and idx not in keep_indices:
            path.unlink()
            deleted += 1

    remaining_indices = sorted(keep_indices)
    remap = {old: new for new, old in enumerate(remaining_indices)}

    pending_renames: list[tuple[Path, Path]] = []
    for path in sorted(output_dir.iterdir(), key=lambda p: p.name):
        if not path.is_file():
            continue
        idx = frame_index(path)
        if idx is None or idx not in remap:
            continue
        new_idx = remap[idx]
        if new_idx == idx:
            continue
        new_name = frame_pattern.sub(
            lambda m, n=new_idx: f"{m.group(1)}_{n:04d}{m.group(3)}",
            path.name,
            count=1,
        )
        final_path = path.parent / new_name
        tmp_path = path.parent / f".__trim_tmp_{new_idx:04d}_{path.name}"
        path.rename(tmp_path)
        pending_renames.append((tmp_path, final_path))

    for tmp_path, final_path in pending_renames:
        tmp_path.rename(final_path)

    if deleted:
        print(
            f"✓ kept last {len(remaining_indices)} frame index(es), "
            f"trimmed {deleted} warmup file(s), renumbered from rgb_0000"
        )
    if len(remaining_indices) < expected_frames:
        print(
            f"  ! writer saved only {len(remaining_indices)} frame index(es); "
            f"expected {expected_frames}"
        )


def resolve_camera_path(stage, preferred_path: str) -> str:
    """RSD455 color camera 경로를 확인하고, 없으면 stage에서 탐색합니다."""
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(preferred_path)
    if prim.IsValid() and prim.IsA(UsdGeom.Camera):
        return preferred_path

    candidates = []
    for p in stage.Traverse():
        if not p.IsA(UsdGeom.Camera):
            continue
        path = p.GetPath().pathString
        if "OmniVision_OV9782_Color" in path or path.endswith("Camera_OmniVision_OV9782_Color"):
            candidates.append(path)
        elif "/World/table/Realsense" in path and "Camera" in path:
            candidates.append(path)

    if candidates:
        candidates.sort(key=lambda s: ("OmniVision_OV9782_Color" not in s, s))
        print(f"! preferred camera missing ({preferred_path}), using {candidates[0]}")
        return candidates[0]

    raise RuntimeError(
        f"Camera prim not found: {preferred_path}. "
        "Isaac Sim에서 RSD455 reference가 로드됐는지 확인하세요."
    )


def setup_tools(stage, tools_root: str) -> dict[str, str]:
    tool_paths: dict[str, str] = {}
    missing = []
    for tool in SCENE_TOOLS:
        path = f"{tools_root}/{tool['prim_name']}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            missing.append(path)
            continue
        tool_paths[tool["prim_name"]] = path
    if missing:
        raise RuntimeError(f"Missing tool prims in scene: {missing}")
    if not tool_paths:
        raise RuntimeError(f"No tools found under {tools_root}")
    return tool_paths


def _get_add_update_semantics():
    """Isaac Sim 버전별 semantics API를 통일해서 반환합니다."""
    try:
        from isaacsim.core.utils.semantics import add_update_semantics as fn

        return fn
    except ImportError:
        pass
    try:
        from omni.isaac.core.utils.semantics import add_update_semantics as fn

        return fn
    except ImportError:
        pass

    import omni.usd

    def _fallback(prim, label: str) -> None:
        omni.usd.semantics.add_semantic_label(prim.GetPath().pathString, label)

    return _fallback


def apply_semantic_labels(stage, tool_paths: dict[str, str]) -> None:
    add_update_semantics = _get_add_update_semantics()
    label_by_name = {t["prim_name"]: t["label"] for t in SCENE_TOOLS}
    for prim_name, prim_path in tool_paths.items():
        prim = stage.GetPrimAtPath(prim_path)
        add_update_semantics(prim, label_by_name[prim_name])


def _collect_flat_surface_tops(stage, root_path: str, axes: PlacementAxes) -> list[float]:
    """테이블 하위 얇은 평면 mesh의 up-axis 최대값 후보."""
    from pxr import Usd, UsdGeom

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    root = stage.GetPrimAtPath(root_path)
    candidates: list[float] = []
    if not root.IsValid():
        return candidates

    for prim in Usd.PrimRange(root):
        if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Xform)):
            continue
        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        bbox_min, bbox_max = aligned.GetMin(), aligned.GetMax()
        up_size = float(bbox_max[axes.up_idx] - bbox_min[axes.up_idx])
        u_size = float(bbox_max[axes.u_idx] - bbox_min[axes.u_idx])
        v_size = float(bbox_max[axes.v_idx] - bbox_min[axes.v_idx])
        top = float(bbox_max[axes.up_idx])
        if up_size <= 0.08 and u_size >= 0.05 and v_size >= 0.05 and -0.05 <= top <= 0.20:
            candidates.append(top)
    return candidates


def estimate_desk_surface_height(stage, fallback_table_path: str, axes: PlacementAxes) -> float | None:
    """오른쪽 책상 상판 높이 (여러 평면 중 가장 높은 값)."""
    candidates = _collect_flat_surface_tops(stage, fallback_table_path, axes)
    if candidates:
        return max(candidates)

    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(fallback_table_path)
    if prim.IsValid():
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bbox_max = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox().GetMax()
        return float(bbox_max[axes.up_idx])
    return None


def estimate_staging_surface_height(
    stage,
    table_surface_path: str,
    fallback_table_path: str,
    axes: PlacementAxes,
    *,
    surface_offset: float = DEFAULT_STAGING_SURFACE_OFFSET,
) -> float:
    """왼쪽 스테이션(회색) 표면 높이. 책상 상판보다 surface_offset만큼 낮게 배치."""
    from pxr import Usd, UsdGeom

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    desk_height = estimate_desk_surface_height(stage, fallback_table_path, axes)

    staging_prim = stage.GetPrimAtPath(table_surface_path)
    if staging_prim.IsValid():
        staging_top = float(
            bbox_cache.ComputeWorldBound(staging_prim).ComputeAlignedBox().GetMax()[axes.up_idx]
        )
        if desk_height is None or staging_top < desk_height - 1e-4:
            return staging_top
        # MeshInstance bbox가 책상과 같으면 측정된 오프셋 적용
        return desk_height + surface_offset

    if desk_height is not None:
        return desk_height + surface_offset

    return 0.0


def capture_tool_base_matrices(stage, tool_paths: dict[str, str], UsdGeom) -> dict[str, object]:
    base_matrices = {}
    for prim_name, prim_path in tool_paths.items():
        prim = stage.GetPrimAtPath(prim_path)
        xform = UsdGeom.Xformable(prim)
        local_xf = xform.GetLocalTransformation()
        if isinstance(local_xf, tuple):
            local_xf = local_xf[0]
        base_matrices[prim_name] = local_xf
    return base_matrices


def hide_prim(stage, prim_path: str) -> None:
    from pxr import UsdGeom

    resolved = resolve_stage_prim_path(stage, prim_path) or prim_path
    prim = stage.GetPrimAtPath(resolved)
    if prim.IsValid():
        UsdGeom.Imageable(prim).MakeInvisible()


def prepare_scene_for_offline_capture(stage, tool_paths: dict[str, str]) -> None:
    """물리 시뮬레이션 없이 정적 렌더만 수행 (검은 프레임/카메라 튐 방지)."""
    import carb
    import omni.timeline
    from pxr import UsdPhysics

    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        timeline.stop()

    settings = carb.settings.get_settings()
    settings.set("/physics/updateEnabled", False)

    robot = stage.GetPrimAtPath(DEFAULT_ROBOT_PATH)
    if robot.IsValid() and robot.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI(robot).CreateArticulationEnabledAttr(False)

    for prim_path in tool_paths.values():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rb = UsdPhysics.RigidBodyAPI(prim)
        rb.CreateRigidBodyEnabledAttr(False)

    disable_toolbox_drawer_physics(stage)


def randomize_tool_layout(
    stage,
    tool_paths: dict[str, str],
    visible_tools: list[str],
    bounds: dict[str, float],
    surface_height: float,
    tool_base_matrices: dict[str, object],
    axes: PlacementAxes,
    Usd,
    UsdGeom,
    *,
    min_gap: float = DEFAULT_TOOL_MIN_GAP,
    spread_pad: float = DEFAULT_TOOL_SPREAD_PAD,
    tight_cluster: bool = False,
    tight_max_gap: float = DEFAULT_TIGHT_MAX_GAP,
) -> None:
    # 사라지는 공구 잔상 방지: 전부 화면 밖으로 이동 후 visible만 재배치
    for prim_name, prim_path in tool_paths.items():
        stash_tool_offscreen(
            stage, prim_path, prim_name, tool_base_matrices, axes, Usd, UsdGeom
        )

    if tight_cluster and len(visible_tools) >= 2:
        _place_tools_tight_cluster(
            stage,
            tool_paths,
            visible_tools,
            bounds,
            surface_height,
            tool_base_matrices,
            axes,
            Usd,
            UsdGeom,
            min_gap=min_gap,
            tight_max_gap=tight_max_gap,
            spread_pad=spread_pad,
        )
    else:
        _place_tools_spread(
            stage,
            tool_paths,
            visible_tools,
            bounds,
            surface_height,
            tool_base_matrices,
            axes,
            Usd,
            UsdGeom,
            min_gap=min_gap,
            spread_pad=spread_pad,
        )


def stash_tool_offscreen(
    stage,
    prim_path: str,
    prim_name: str,
    tool_base_matrices: dict[str, object],
    axes: PlacementAxes,
    Usd,
    UsdGeom,
) -> None:
    """숨긴 공구를 스테이징 밖으로 옮겨 RTX temporal buffer 잔상을 줄임."""
    set_tool_world_pose(
        stage,
        prim_path,
        TOOL_STASH_U,
        TOOL_STASH_V,
        TOOL_STASH_HEIGHT,
        0.0,
        tool_base_matrices[prim_name],
        axes,
        Usd,
        UsdGeom,
    )
    set_visibility(stage, prim_path, False)


def stash_all_tools(
    stage,
    tool_paths: dict[str, str],
    tool_base_matrices: dict[str, object],
    axes: PlacementAxes,
    Usd,
    UsdGeom,
) -> None:
    for prim_name, prim_path in tool_paths.items():
        stash_tool_offscreen(
            stage, prim_path, prim_name, tool_base_matrices, axes, Usd, UsdGeom
        )


def _place_tools_spread(
    stage,
    tool_paths: dict[str, str],
    visible_tools: list[str],
    bounds: dict[str, float],
    surface_height: float,
    tool_base_matrices: dict[str, object],
    axes: PlacementAxes,
    Usd,
    UsdGeom,
    *,
    min_gap: float,
    spread_pad: float,
) -> None:
    placed_bboxes: list[tuple[float, float, float, float]] = []
    slots = _sample_non_overlapping_positions(
        len(visible_tools), bounds, spread_pad, min_gap=min_gap
    )

    for prim_name, (u, v, yaw) in zip(visible_tools, slots):
        prim_path = tool_paths[prim_name]
        set_tool_world_pose(
            stage,
            prim_path,
            u,
            v,
            surface_height,
            yaw,
            tool_base_matrices[prim_name],
            axes,
            Usd,
            UsdGeom,
        )
        shift_tool_bbox_into_bounds(stage, prim_path, bounds, axes, Usd, UsdGeom)
        placed_bboxes.append(get_tool_plane_bbox(stage, prim_path, axes, Usd, UsdGeom))
        set_visibility(stage, prim_path, True)


def _place_tools_tight_cluster(
    stage,
    tool_paths: dict[str, str],
    visible_tools: list[str],
    bounds: dict[str, float],
    surface_height: float,
    tool_base_matrices: dict[str, object],
    axes: PlacementAxes,
    Usd,
    UsdGeom,
    *,
    min_gap: float,
    tight_max_gap: float,
    spread_pad: float,
) -> None:
    """첫 공구는 랜덤, 이후 공구는 기존 bbox에 거의 붙게 배치 (겹치지 않음)."""
    ordered = list(visible_tools)
    random.shuffle(ordered)
    placed: list[tuple[str, tuple[float, float, float, float]]] = []

    first_name = ordered[0]
    first_path = tool_paths[first_name]
    u = random.uniform(bounds["min_u"], bounds["max_u"])
    v = random.uniform(bounds["min_v"], bounds["max_v"])
    yaw = random.uniform(-180.0, 180.0)
    set_tool_world_pose(
        stage,
        first_path,
        u,
        v,
        surface_height,
        yaw,
        tool_base_matrices[first_name],
        axes,
        Usd,
        UsdGeom,
    )
    shift_tool_bbox_into_bounds(stage, first_path, bounds, axes, Usd, UsdGeom)
    placed.append((first_name, get_tool_plane_bbox(stage, first_path, axes, Usd, UsdGeom)))
    set_visibility(stage, first_path, True)

    for prim_name in ordered[1:]:
        prim_path = tool_paths[prim_name]
        placed_bbox = _place_tool_near_cluster(
            stage,
            prim_path,
            prim_name,
            placed,
            bounds,
            surface_height,
            tool_base_matrices[prim_name],
            axes,
            Usd,
            UsdGeom,
            min_gap=min_gap,
            tight_max_gap=tight_max_gap,
            spread_pad=spread_pad,
        )
        placed.append((prim_name, placed_bbox))
        set_visibility(stage, prim_path, True)


def _place_tool_near_cluster(
    stage,
    prim_path: str,
    prim_name: str,
    placed: list[tuple[str, tuple[float, float, float, float]]],
    bounds: dict[str, float],
    surface_height: float,
    base_matrix,
    axes: PlacementAxes,
    Usd,
    UsdGeom,
    *,
    min_gap: float,
    tight_max_gap: float,
    spread_pad: float,
) -> tuple[float, float, float, float]:
    sides = ("right", "left", "top", "bottom")
    placed_bboxes = [bbox for _, bbox in placed]

    for _attempt in range(120):
        anchor_bbox = random.choice(placed_bboxes)
        side = random.choice(sides)
        gap = random.uniform(min_gap, tight_max_gap)
        yaw = random.uniform(-180.0, 180.0)
        anchor_u = (anchor_bbox[0] + anchor_bbox[2]) * 0.5
        anchor_v = (anchor_bbox[1] + anchor_bbox[3]) * 0.5

        set_tool_world_pose(
            stage,
            prim_path,
            anchor_u,
            anchor_v,
            surface_height,
            yaw,
            base_matrix,
            axes,
            Usd,
            UsdGeom,
        )
        _snap_tool_to_neighbor(stage, prim_path, anchor_bbox, side, gap, axes, Usd, UsdGeom)
        shift_tool_bbox_into_bounds(stage, prim_path, bounds, axes, Usd, UsdGeom)
        candidate = get_tool_plane_bbox(stage, prim_path, axes, Usd, UsdGeom)
        if all(not _bboxes_overlap(candidate, other, min_gap) for other in placed_bboxes):
            return candidate

    # tight 배치 실패 시 spread fallback
    u = random.uniform(bounds["min_u"], bounds["max_u"])
    v = random.uniform(bounds["min_v"], bounds["max_v"])
    yaw = random.uniform(-180.0, 180.0)
    set_tool_world_pose(
        stage, prim_path, u, v, surface_height, yaw, base_matrix, axes, Usd, UsdGeom
    )
    shift_tool_bbox_into_bounds(stage, prim_path, bounds, axes, Usd, UsdGeom)
    candidate = get_tool_plane_bbox(stage, prim_path, axes, Usd, UsdGeom)
    for _attempt in range(40):
        if all(not _bboxes_overlap(candidate, other, min_gap) for other in placed_bboxes):
            return candidate
        u = random.uniform(bounds["min_u"], bounds["max_u"])
        v = random.uniform(bounds["min_v"], bounds["max_v"])
        yaw = random.uniform(-180.0, 180.0)
        set_tool_world_pose(
            stage, prim_path, u, v, surface_height, yaw, base_matrix, axes, Usd, UsdGeom
        )
        shift_tool_bbox_into_bounds(stage, prim_path, bounds, axes, Usd, UsdGeom)
        candidate = get_tool_plane_bbox(stage, prim_path, axes, Usd, UsdGeom)
    return candidate


def _snap_tool_to_neighbor(
    stage,
    prim_path: str,
    anchor_bbox: tuple[float, float, float, float],
    side: str,
    gap: float,
    axes: PlacementAxes,
    Usd,
    UsdGeom,
) -> None:
    tool_bbox = get_tool_plane_bbox(stage, prim_path, axes, Usd, UsdGeom)
    tu0, tv0, tu1, tv1 = tool_bbox
    au0, av0, au1, av1 = anchor_bbox
    delta = [0.0, 0.0, 0.0]

    if side == "right":
        delta[axes.u_idx] = (au1 + gap) - tu0
    elif side == "left":
        delta[axes.u_idx] = (au0 - gap) - tu1
    elif side == "top":
        delta[axes.v_idx] = (av1 + gap) - tv0
    elif side == "bottom":
        delta[axes.v_idx] = (av0 - gap) - tv1

    if any(abs(value) > 1e-9 for value in delta):
        apply_translation_delta(stage, prim_path, delta, UsdGeom)


def _sample_non_overlapping_positions(
    count: int,
    bounds: dict[str, float],
    spread_pad: float,
    *,
    min_gap: float,
) -> list[tuple[float, float, float]]:
    results: list[tuple[float, float, float]] = []
    local_bboxes: list[tuple[float, float, float, float]] = []
    pad = max(spread_pad, min_gap)
    for _ in range(count):
        for _attempt in range(80):
            u = random.uniform(bounds["min_u"], bounds["max_u"])
            v = random.uniform(bounds["min_v"], bounds["max_v"])
            yaw = random.uniform(-180.0, 180.0)
            bbox = (u - pad, v - pad, u + pad, v + pad)
            if not any(_bboxes_overlap(bbox, other, min_gap) for other in local_bboxes):
                results.append((u, v, yaw))
                local_bboxes.append(bbox)
                break
        else:
            results.append((u, v, yaw))
            local_bboxes.append((u - pad, v - pad, u + pad, v + pad))
    return results


def set_tool_world_pose(
    stage,
    prim_path: str,
    u: float,
    v: float,
    surface_height: float,
    yaw_deg: float,
    base_matrix,
    axes: PlacementAxes,
    Usd,
    UsdGeom,
) -> None:
    """원래 자세/스케일은 보존하고 bbox 기준으로 책상 표면에 배치."""
    from pxr import Gf

    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)

    yaw_mat = Gf.Matrix4d(1.0)
    yaw_mat.SetRotate(Gf.Rotation(_axis_vec(axes.up_axis, Gf), yaw_deg))
    mat = Gf.Matrix4d(base_matrix)
    mat.SetTranslateOnly(Gf.Vec3d(0.0, 0.0, 0.0))
    mat = mat * yaw_mat

    initial_t = [0.0, 0.0, 0.0]
    initial_t[axes.u_idx] = u
    initial_t[axes.v_idx] = v
    initial_t[axes.up_idx] = surface_height + 0.05
    mat.SetTranslateOnly(Gf.Vec3d(*initial_t))

    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(mat)
    align_tool_bbox(stage, prim_path, u, v, surface_height, axes, Usd, UsdGeom)


def align_tool_bbox(
    stage, prim_path: str, u: float, v: float, surface_height: float, axes: PlacementAxes, Usd, UsdGeom
) -> None:
    aligned = get_world_aligned_bbox(stage, prim_path, Usd, UsdGeom)
    bbox_min, bbox_max = aligned.GetMin(), aligned.GetMax()
    center_u = (bbox_min[axes.u_idx] + bbox_max[axes.u_idx]) * 0.5
    center_v = (bbox_min[axes.v_idx] + bbox_max[axes.v_idx]) * 0.5
    bottom = bbox_min[axes.up_idx]
    delta = [0.0, 0.0, 0.0]
    delta[axes.u_idx] = u - center_u
    delta[axes.v_idx] = v - center_v
    delta[axes.up_idx] = surface_height + SURFACE_CLEARANCE_M - bottom
    apply_translation_delta(stage, prim_path, delta, UsdGeom)


def shift_tool_bbox_into_bounds(
    stage, prim_path: str, bounds: dict[str, float], axes: PlacementAxes, Usd, UsdGeom
) -> None:
    aligned = get_world_aligned_bbox(stage, prim_path, Usd, UsdGeom)
    bbox_min, bbox_max = aligned.GetMin(), aligned.GetMax()
    delta = [0.0, 0.0, 0.0]
    if bbox_min[axes.u_idx] < bounds["min_u"]:
        delta[axes.u_idx] = bounds["min_u"] - bbox_min[axes.u_idx]
    elif bbox_max[axes.u_idx] > bounds["max_u"]:
        delta[axes.u_idx] = bounds["max_u"] - bbox_max[axes.u_idx]
    if bbox_min[axes.v_idx] < bounds["min_v"]:
        delta[axes.v_idx] = bounds["min_v"] - bbox_min[axes.v_idx]
    elif bbox_max[axes.v_idx] > bounds["max_v"]:
        delta[axes.v_idx] = bounds["max_v"] - bbox_max[axes.v_idx]
    if any(abs(value) > 1e-8 for value in delta):
        apply_translation_delta(stage, prim_path, delta, UsdGeom)


def apply_translation_delta(stage, prim_path: str, delta: list[float], UsdGeom) -> None:
    from pxr import Gf

    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:transform":
            mat = op.Get()
            t = mat.ExtractTranslation()
            mat.SetTranslateOnly(Gf.Vec3d(t[0] + delta[0], t[1] + delta[1], t[2] + delta[2]))
            op.Set(mat)
            return
        if op.GetOpName() == "xformOp:translate":
            current = op.Get()
            op.Set((current[0] + delta[0], current[1] + delta[1], current[2] + delta[2]))
            return
    xform.AddTranslateOp().Set((delta[0], delta[1], delta[2]))


def get_world_aligned_bbox(stage, prim_path: str, Usd, UsdGeom):
    prim = stage.GetPrimAtPath(prim_path)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    return bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()


def get_tool_plane_bbox(
    stage, prim_path: str, axes: PlacementAxes, Usd, UsdGeom
) -> tuple[float, float, float, float]:
    aligned = get_world_aligned_bbox(stage, prim_path, Usd, UsdGeom)
    bbox_min = aligned.GetMin()
    bbox_max = aligned.GetMax()
    return (bbox_min[axes.u_idx], bbox_min[axes.v_idx], bbox_max[axes.u_idx], bbox_max[axes.v_idx])


def _axis_vec(axis: str, Gf):
    if axis == "x":
        return Gf.Vec3d(1.0, 0.0, 0.0)
    if axis == "y":
        return Gf.Vec3d(0.0, 1.0, 0.0)
    return Gf.Vec3d(0.0, 0.0, 1.0)


def _bboxes_overlap(a, b, pad: float = 0.003) -> bool:
    return (
        a[0] - pad < b[2]
        and a[2] + pad > b[0]
        and a[1] - pad < b[3]
        and a[3] + pad > b[1]
    )


def set_visibility(stage, prim_path: str, visible: bool) -> None:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()


@dataclass
class DrawerJointController:
    joint_path: str
    drawer_path: str
    axis: str
    closed_drawer_matrix: object
    joint_axis_world_scale: float
    lower_limit: float
    upper_limit: float
    pos_attr: object


def _drawer_axis_vector(axis: str):
    from pxr import Gf

    return {
        "X": Gf.Vec3d(1.0, 0.0, 0.0),
        "Y": Gf.Vec3d(0.0, 1.0, 0.0),
        "Z": Gf.Vec3d(0.0, 0.0, 1.0),
    }[axis]


def resolve_stage_prim_path(stage, path: str) -> str | None:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        return path

    if path.startswith("/World/"):
        return None

    candidates = []
    if path.startswith("/"):
        candidates.append(f"/World{path}")
    else:
        candidates.append(f"/World/{path}")

    for candidate in candidates:
        if stage.GetPrimAtPath(candidate).IsValid():
            return candidate
    return None


def disable_toolbox_drawer_physics(stage) -> None:
    """오프라인 캡처 시 USD xform이 렌더에 반영되도록 drawer articulation/rigid body를 끕니다."""
    from pxr import UsdPhysics

    toolbox_root = resolve_stage_prim_path(stage, DEFAULT_TOOLBOX_ROOT)
    if toolbox_root is None:
        return

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(toolbox_root):
            continue
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            UsdPhysics.ArticulationRootAPI(prim).CreateArticulationEnabledAttr(False)
        if prim.GetName() in ("drawer", "drawer_02") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)


def _drawer_joint_axis_world_scale(stage, drawer_path: str, axis: str) -> float:
    from pxr import UsdGeom

    drawer = stage.GetPrimAtPath(drawer_path)
    if not drawer.IsValid():
        return 1.0

    cache = UsdGeom.XformCache()
    parent_xf = cache.GetLocalToWorldTransform(drawer.GetParent())
    axis_world = parent_xf.TransformDir(_drawer_axis_vector(axis))
    scale = axis_world.GetLength()
    return scale if scale > 1e-9 else 1.0


def normalize_drawer_open_range(
    open_range: tuple[float, float],
    lower_limit: float,
    upper_limit: float,
) -> tuple[float, float]:
    lo, hi = float(open_range[0]), float(open_range[1])
    if lo > 0.0 and hi > 0.0:
        lo, hi = -hi, -lo
    if lo > hi:
        lo, hi = hi, lo
    lo = max(lo, lower_limit)
    hi = min(hi, upper_limit)
    if lo > hi:
        lo = hi = upper_limit
    return lo, hi


def _is_drawer_joint_prim(prim) -> bool:
    if not prim or not prim.IsValid():
        return False
    if prim.GetName() != "drawer_joint":
        return False
    return prim.GetAttribute("state:linear:physics:position") is not None


def resolve_drawer_joint_path(stage, preferred_path: str) -> str | None:
    prim = stage.GetPrimAtPath(preferred_path)
    if _is_drawer_joint_prim(prim):
        return preferred_path

    alt_paths: list[str] = []
    if "/toolbox/toolbox/toolbox/drawer_joint" in preferred_path:
        alt_paths.append(
            preferred_path.replace(
                "/toolbox/toolbox/toolbox/drawer_joint",
                "/toolbox/toolbox/drawer_joint",
            )
        )
    elif "/toolbox/toolbox/drawer_joint" in preferred_path:
        alt_paths.append(
            preferred_path.replace(
                "/toolbox/toolbox/drawer_joint",
                "/toolbox/toolbox/toolbox/drawer_joint",
            )
        )

    for alt_path in alt_paths + [p for p in DRAWER_JOINT_FALLBACKS if p != preferred_path]:
        alt_prim = stage.GetPrimAtPath(alt_path)
        if _is_drawer_joint_prim(alt_prim):
            print(f"✓ drawer joint auto-resolved: {alt_path} (requested {preferred_path})")
            return alt_path

    discovered = []
    for candidate in stage.Traverse():
        if _is_drawer_joint_prim(candidate):
            discovered.append(str(candidate.GetPath()))

    if not discovered:
        return None

    discovered.sort(key=lambda path: ("/toolbox_with_handle" not in path, len(path)))
    resolved = discovered[0]
    if resolved != preferred_path:
        suffix = f" ({len(discovered)} candidates)" if len(discovered) > 1 else ""
        print(f"✓ drawer joint auto-discovered: {resolved}{suffix}")
    return resolved


def setup_drawer_controller(stage, joint_path: str) -> DrawerJointController | None:
    from pxr import UsdGeom

    resolved_path = resolve_drawer_joint_path(stage, joint_path)
    if resolved_path is None:
        print(f"⚠ drawer joint not found: {joint_path}")
        return None

    joint_path = resolved_path
    joint = stage.GetPrimAtPath(joint_path)

    pos_attr = joint.GetAttribute("state:linear:physics:position")
    if not pos_attr:
        print(f"⚠ drawer joint missing state:linear:physics:position: {joint_path}")
        return None

    axis = "Y"
    axis_attr = joint.GetAttribute("physics:axis")
    if axis_attr and axis_attr.Get() is not None:
        axis = str(axis_attr.Get())

    lower_attr = joint.GetAttribute("physics:lowerLimit")
    upper_attr = joint.GetAttribute("physics:upperLimit")
    lower_limit = float(lower_attr.Get()) if lower_attr and lower_attr.Get() is not None else -0.2
    upper_limit = float(upper_attr.Get()) if upper_attr and upper_attr.Get() is not None else 0.0

    body1_targets = joint.GetRelationship("physics:body1").GetTargets()
    if body1_targets:
        drawer_path = resolve_stage_prim_path(stage, str(body1_targets[0]))
        if drawer_path is None:
            drawer_path = str(body1_targets[0])
    else:
        drawer_path = resolve_stage_prim_path(
            stage, str(Path(joint_path).parent / "drawer")
        ) or str(Path(joint_path).parent / "drawer")

    drawer = stage.GetPrimAtPath(drawer_path)
    if not drawer.IsValid():
        print(f"⚠ drawer body not found: {drawer_path}")
        return None

    closed_drawer_matrix = UsdGeom.Xformable(drawer).GetLocalTransformation()
    joint_axis_world_scale = _drawer_joint_axis_world_scale(stage, drawer_path, axis)
    pos_attr.Set(upper_limit)
    print(
        f"✓ drawer randomizer: joint={joint_path}, body={drawer_path}, "
        f"axis={axis}, limits=[{lower_limit:.3f}, {upper_limit:.3f}] m, "
        f"axis_scale={joint_axis_world_scale:.4f}"
    )
    return DrawerJointController(
        joint_path=joint_path,
        drawer_path=drawer_path,
        axis=axis,
        closed_drawer_matrix=closed_drawer_matrix,
        joint_axis_world_scale=joint_axis_world_scale,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        pos_attr=pos_attr,
    )


def apply_drawer_joint_position(stage, controller: DrawerJointController, position: float) -> None:
    from pxr import Gf, UsdGeom

    controller.pos_attr.Set(position)
    target_attr = stage.GetPrimAtPath(controller.joint_path).GetAttribute(
        "drive:linear:physics:targetPosition"
    )
    if target_attr:
        target_attr.Set(position)

    drawer = stage.GetPrimAtPath(controller.drawer_path)
    if not drawer.IsValid():
        return

    axis_vec = _drawer_axis_vector(controller.axis)
    # Joint position is meters; parent unitsResolve scale must be compensated in local xform.
    local_offset = position / controller.joint_axis_world_scale
    closed = Gf.Matrix4d(controller.closed_drawer_matrix)
    new_matrix = Gf.Matrix4d(closed)
    new_matrix.SetTranslateOnly(closed.ExtractTranslation() + axis_vec * local_offset)

    transform_attr = drawer.GetAttribute("xformOp:transform")
    if transform_attr:
        transform_attr.Set(new_matrix)
        return

    xformable = UsdGeom.Xformable(drawer)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
            op.Set(new_matrix)
            return

    translate_op = xformable.GetXformOp(UsdGeom.XformOp.TypeTranslate)
    if translate_op:
        translate_op.Set(closed.ExtractTranslation() + axis_vec * local_offset)
        return

    op = xformable.AddTranslateOp()
    op.Set(axis_vec * local_offset)


def randomize_drawer(
    controller: DrawerJointController | None,
    open_prob: float,
    open_range: tuple[float, float],
) -> None:
    if controller is None:
        return

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    lo, hi = normalize_drawer_open_range(
        open_range,
        controller.lower_limit,
        controller.upper_limit,
    )
    if random.random() < open_prob:
        value = random.uniform(lo, hi)
    else:
        value = controller.upper_limit
    apply_drawer_joint_position(stage, controller, value)


def randomize_dome_light(stage) -> None:
    import carb
    from pxr import Gf

    dome = stage.GetPrimAtPath("/DomeLight")
    if not dome.IsValid():
        dome = stage.GetPrimAtPath("/World/DomeLight")
    if dome.IsValid():
        intensity_attr = dome.GetAttribute("inputs:intensity")
        if intensity_attr:
            intensity_attr.Set(random.uniform(*DOME_LIGHT_INTENSITY_RANGE))

        color_attr = dome.GetAttribute("inputs:color")
        if color_attr:
            # Mild white-balance drift: enough for robustness, not enough to change object identity.
            color = Gf.Vec3f(
                random.uniform(*DOME_LIGHT_COLOR_RANGE),
                random.uniform(*DOME_LIGHT_COLOR_RANGE),
                random.uniform(*DOME_LIGHT_COLOR_RANGE),
            )
            color_attr.Set(color)

        enable_temp_attr = dome.GetAttribute("inputs:enableColorTemperature")
        if enable_temp_attr:
            enable_temp_attr.Set(False)

    settings = carb.settings.get_settings()
    for key in ("/rtx/post/tonemap/exposure", "/rtx/post/exposure"):
        try:
            settings.set(key, random.uniform(*EXPOSURE_RANGE))
        except Exception:
            pass


def configure_renderer_for_still_images(carb) -> None:
    settings = carb.settings.get_settings()
    for key, value in {
        "/rtx/post/aa/op": 0,
        "/rtx/post/aa/taa/enabled": False,
        "/rtx/post/aa/temporal/enabled": False,
        "/rtx/post/motionblur/enabled": False,
        "/rtx/post/motionBlur/enabled": False,
        "/rtx/post/dlss/enabled": False,
        "/rtx/post/dlss/execMode": 0,
        "/rtx-transient/dlssg/enabled": False,
        "/rtx/pathtracing/optixDenoiser/enabled": False,
        "/rtx/pathtracing/optixDenoiser/enable": False,
        "/rtx/pathtracing/temporalDenoising/enabled": False,
        "/rtx/pathtracing/temporalDenoising/enable": False,
        "/rtx/raytracing/denoiser/enabled": False,
        "/rtx/raytracing/temporalDenoising/enabled": False,
        "/rtx/hydra/subframe/enabled": False,
    }.items():
        try:
            settings.set(key, value)
        except Exception:
            pass


def reset_render_accumulation(carb) -> None:
    settings = carb.settings.get_settings()
    for key in (
        "/rtx/resetPtAccumulation",
        "/rtx/pathtracing/resetAccumulation",
        "/rtx/raytracing/resetAccumulation",
        "/app/renderer/resetAccumulation",
        "/rtx/hydra/resetAccumulation",
    ):
        try:
            settings.set(key, True)
        except Exception:
            pass
    try:
        import omni.kit.viewport.utility as vp_utils

        viewport = vp_utils.get_active_viewport()
        if viewport is not None and hasattr(viewport, "viewport_api"):
            api = viewport.viewport_api
            for method_name in ("invalidate", "reset_accumulation", "request_reset"):
                method = getattr(api, method_name, None)
                if callable(method):
                    method()
                    break
    except Exception:
        pass


def print_output_summary(output_dir: Path, expected_frames: int) -> None:
    rgb_files = sorted(output_dir.glob("rgb_*.png"))
    seg_files = sorted(output_dir.glob("instance_segmentation_*.png"))
    total_bytes = sum(p.stat().st_size for p in output_dir.iterdir() if p.is_file())
    print("\n[Output summary]")
    print(f"  path: {output_dir}")
    print(f"  rgb: {len(rgb_files)} files (expected {expected_frames})")
    print(f"  instance_segmentation: {len(seg_files)} files")
    print(f"  total size: {total_bytes / 1024 / 1024:.2f} MB")
    for rgb_path in rgb_files[:3]:
        print(f"    {rgb_path.name}: {rgb_path.stat().st_size / 1024:.1f} KB")
    if len(rgb_files) > 3:
        print(f"    ... +{len(rgb_files) - 3} more")
    black = []
    for rgb_path in rgb_files:
        from PIL import Image
        import numpy as np

        arr = np.array(Image.open(rgb_path))
        if arr[..., :3].max() < 10:
            black.append(rgb_path.name)
    if black:
        print(f"  ! nearly-black rgb: {', '.join(black)}")
        print("    → physics/렌더 warmup 부족일 수 있습니다. --num-frames 늘리거나 재실행해 보세요.")
    if len(rgb_files) < expected_frames:
        print("  ! rgb 파일 수가 요청 frame 수보다 적습니다.")


def add_camera_noise_to_rgb(output_dir: Path, seed: int) -> None:
    if not output_dir.exists():
        return
    from PIL import Image
    import numpy as np

    rng = np.random.default_rng(seed + 1000)
    rgb_paths = sorted(output_dir.glob("rgb_*.png"))
    for rgb_path in rgb_paths:
        image = Image.open(rgb_path)
        arr = np.asarray(image).astype(np.float32) / 255.0
        rgb = arr[..., :3]
        alpha = arr[..., 3:4] if arr.shape[-1] == 4 else None

        gamma = rng.uniform(*RGB_GAMMA_RANGE)
        brightness = rng.uniform(*RGB_BRIGHTNESS_RANGE)
        contrast = rng.uniform(*RGB_CONTRAST_RANGE)
        rgb = np.power(np.clip(rgb, 0.0, 1.0), gamma)
        rgb = (rgb - 0.5) * contrast + 0.5
        rgb = rgb * brightness

        noise_std = rng.uniform(*RGB_NOISE_STD_RANGE)
        noisy_rgb = np.clip(rgb + rng.normal(0.0, noise_std, rgb.shape), 0.0, 1.0)
        if alpha is not None:
            noisy = np.concatenate([noisy_rgb, alpha], axis=-1)
        else:
            noisy = noisy_rgb
        Image.fromarray((noisy * 255.0).astype(np.uint8)).save(rgb_path)
    if rgb_paths:
        print(f"✓ RGB brightness/contrast/gamma/noise jitter applied: {len(rgb_paths)} images")


if __name__ == "__main__":
    main()
