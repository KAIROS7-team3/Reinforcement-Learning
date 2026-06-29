#!/usr/bin/env python3
"""SDG domain randomization 비교 이미지 4쌍(8장) + side-by-side 합성."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass
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
    DOME_LIGHT_COLOR_RANGE,
    DOME_LIGHT_INTENSITY_RANGE,
    EXPOSURE_RANGE,
    PRE_CAPTURE_APP_UPDATES,
    RGB_BRIGHTNESS_RANGE,
    RGB_CONTRAST_RANGE,
    RGB_GAMMA_RANGE,
    RGB_NOISE_STD_RANGE,
    WRITER_WARMUP_FRAMES,
    PlacementAxes,
    apply_drawer_joint_position,
    apply_semantic_labels,
    capture_tool_base_matrices,
    cleanup_conflicting_replicator_prims,
    compute_staging_bounds_from_table,
    configure_renderer_for_still_images,
    create_flat_capture_camera,
    estimate_staging_surface_height,
    hide_prim,
    normalize_drawer_open_range,
    prepare_scene_for_offline_capture,
    randomize_tool_layout,
    reset_render_accumulation,
    resolve_camera_path,
    setup_drawer_controller,
    setup_replicator_writer,
    setup_tools,
    wait_for_stage_ready,
)

DEFAULT_OUTPUT = YOLO_DIR.parent / "docs" / "assets" / "sdg_comparisons"
# 비교 캡처는 잔상 제거를 위해 메인 SDG보다 긴 settle 사용
DEFAULT_SETTLE_UPDATES = 96
DEFAULT_SETTLE_PASSES = 2
DEFAULT_RT_SUBFRAMES = 32
DEFAULT_POST_CAPTURE_UPDATES = 24


@dataclass(frozen=True)
class CaptureSpec:
    filename: str
    label: str
    seed: int
    min_tools: int
    max_tools: int
    tight_cluster: bool
    drawer_open: bool | None = None
    dome_intensity: float | None = None
    dome_color: tuple[float, float, float] | None = None
    exposure: float | None = None
    rgb_jitter: str = "none"  # none | heavy
    spread_pad: float = 0.03
    tight_max_gap: float = 0.012


def build_pairs(*, seed_offset: int = 0, maximize_contrast: bool = False) -> list[
    tuple[str, str, CaptureSpec, CaptureSpec]
]:
    """비교 쌍 정의. maximize_contrast=True면 발표용으로 좌우 차이를 극대화."""
    spread_pad = 0.09 if maximize_contrast else 0.03
    tight_gap = 0.004 if maximize_contrast else 0.012
    dome_dim = 80.0 if maximize_contrast else DOME_LIGHT_INTENSITY_RANGE[0]
    dome_bright = 650.0 if maximize_contrast else DOME_LIGHT_INTENSITY_RANGE[1]
    dome_dim_color = (0.90, 0.94, 1.10) if maximize_contrast else (1.0, 1.0, 1.0)
    dome_bright_color = (1.14, 1.0, 0.84) if maximize_contrast else (1.04, 1.0, 0.96)
    exp_low = EXPOSURE_RANGE[0] - (0.18 if maximize_contrast else 0.0)
    exp_high = EXPOSURE_RANGE[1] + (0.22 if maximize_contrast else 0.0)
    layout_tools_lo = 1 if maximize_contrast else 2
    layout_tools_hi = 2 if maximize_contrast else 2

    def s(base: int) -> int:
        return base + seed_offset

    return [
        (
            "01_tool_layout",
            "공구 수 · 배치 · yaw",
            CaptureSpec(
                filename="01a_spread_2tools.png",
                label="Spread · few tools" if maximize_contrast else "Spread · 2 tools",
                seed=s(11),
                min_tools=layout_tools_lo,
                max_tools=layout_tools_hi,
                tight_cluster=False,
                drawer_open=False,
                dome_intensity=300.0,
                dome_color=(1.0, 1.0, 1.0),
                exposure=0.0,
                spread_pad=spread_pad,
                tight_max_gap=tight_gap,
            ),
            CaptureSpec(
                filename="01b_tight_6tools.png",
                label="Tight cluster · 6 tools",
                seed=s(22),
                min_tools=6,
                max_tools=6,
                tight_cluster=True,
                drawer_open=False,
                dome_intensity=300.0,
                dome_color=(1.0, 1.0, 1.0),
                exposure=0.0,
                spread_pad=spread_pad,
                tight_max_gap=tight_gap,
            ),
        ),
        (
            "02_drawer",
            "서랍 열림 / 닫힘",
            CaptureSpec(
                filename="02a_drawer_closed.png",
                label="Drawer closed",
                seed=s(33),
                min_tools=4,
                max_tools=4,
                tight_cluster=False,
                drawer_open=False,
                dome_intensity=300.0,
                dome_color=(1.0, 1.0, 1.0),
                exposure=0.0,
                spread_pad=spread_pad,
            ),
            CaptureSpec(
                filename="02b_drawer_open.png",
                label="Drawer open",
                seed=s(33),
                min_tools=4,
                max_tools=4,
                tight_cluster=False,
                drawer_open=True,
                dome_intensity=300.0,
                dome_color=(1.0, 1.0, 1.0),
                exposure=0.0,
                spread_pad=spread_pad,
            ),
        ),
        (
            "03_dome_light",
            "Dome light intensity / color",
            CaptureSpec(
                filename="03a_dome_dim_neutral.png",
                label="Dim · cool tint" if maximize_contrast else "Dim · neutral",
                seed=s(44),
                min_tools=4,
                max_tools=4,
                tight_cluster=False,
                drawer_open=False,
                dome_intensity=dome_dim,
                dome_color=dome_dim_color,
                exposure=0.0,
                spread_pad=spread_pad,
            ),
            CaptureSpec(
                filename="03b_dome_bright_warm.png",
                label="Bright · warm tint",
                seed=s(44),
                min_tools=4,
                max_tools=4,
                tight_cluster=False,
                drawer_open=False,
                dome_intensity=dome_bright,
                dome_color=dome_bright_color,
                exposure=0.0,
                spread_pad=spread_pad,
            ),
        ),
        (
            "04_exposure_rgb",
            "Exposure · RGB jitter",
            CaptureSpec(
                filename="04a_low_exposure_clean.png",
                label="Low exposure · clean RGB",
                seed=s(55),
                min_tools=4,
                max_tools=4,
                tight_cluster=False,
                drawer_open=False,
                dome_intensity=300.0,
                dome_color=(1.0, 1.0, 1.0),
                exposure=exp_low,
                rgb_jitter="none",
                spread_pad=spread_pad,
            ),
            CaptureSpec(
                filename="04b_high_exposure_noisy.png",
                label="High exposure · noisy RGB",
                seed=s(55),
                min_tools=4,
                max_tools=4,
                tight_cluster=False,
                drawer_open=False,
                dome_intensity=300.0,
                dome_color=(1.0, 1.0, 1.0),
                exposure=exp_high,
                rgb_jitter="heavy",
                spread_pad=spread_pad,
            ),
        ),
    ]


PAIRS = build_pairs()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDG comparison image export")
    parser.add_argument("--scene-usd", type=Path, default=DEFAULT_SCENE_USD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=int, default=1280)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument(
        "--hide-toolbox",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="공구함 숨김 (기본: 보임)",
    )
    parser.add_argument(
        "--settle-updates",
        type=int,
        default=DEFAULT_SETTLE_UPDATES,
        help="캡처 직전 SimulationApp update 횟수 (잔상 완화)",
    )
    parser.add_argument(
        "--settle-passes",
        type=int,
        default=DEFAULT_SETTLE_PASSES,
        help="reset_render_accumulation + settle 반복 횟수",
    )
    parser.add_argument(
        "--rt-subframes",
        type=int,
        default=DEFAULT_RT_SUBFRAMES,
        help="rep.orchestrator.step rt_subframes",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="",
        help="출력 하위 폴더명 (예: v2 → output_dir/v2/). 비우면 output_dir 루트",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="레이아웃 시드 오프셋 (variant마다 다른 배치)",
    )
    parser.add_argument(
        "--maximize-contrast",
        action="store_true",
        help="좌우 차이 극대화 (넓은 spread, 촘촘한 cluster, 극단 조명/노출)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="출력 폴더를 지우지 않고 덮어쓰기",
    )
    return parser.parse_args()


def set_dome_light(stage, intensity: float, color: tuple[float, float, float]) -> None:
    from pxr import Gf

    for path in ("/DomeLight", "/World/DomeLight"):
        dome = stage.GetPrimAtPath(path)
        if not dome.IsValid():
            continue
        intensity_attr = dome.GetAttribute("inputs:intensity")
        if intensity_attr:
            intensity_attr.Set(float(intensity))
        color_attr = dome.GetAttribute("inputs:color")
        if color_attr:
            color_attr.Set(Gf.Vec3f(*color))
        enable_temp_attr = dome.GetAttribute("inputs:enableColorTemperature")
        if enable_temp_attr:
            enable_temp_attr.Set(False)
        return


def set_exposure(carb, exposure: float) -> None:
    settings = carb.settings.get_settings()
    for key in ("/rtx/post/tonemap/exposure", "/rtx/post/exposure"):
        try:
            settings.set(key, float(exposure))
        except Exception:
            pass


def apply_rgb_jitter(rgb_path: Path, mode: str, *, maximize: bool = False) -> None:
    if mode == "none":
        return
    from PIL import Image
    import numpy as np

    image = Image.open(rgb_path)
    arr = np.asarray(image).astype(np.float32) / 255.0
    rgb = arr[..., :3]
    alpha = arr[..., 3:4] if arr.shape[-1] == 4 else None

    if mode == "heavy":
        gamma = RGB_GAMMA_RANGE[1] + (0.08 if maximize else 0.0)
        brightness = RGB_BRIGHTNESS_RANGE[1] + (0.10 if maximize else 0.0)
        contrast = RGB_CONTRAST_RANGE[1] + (0.10 if maximize else 0.0)
        noise_std = RGB_NOISE_STD_RANGE[1] * (2.2 if maximize else 1.0)
    else:
        gamma = 1.0
        brightness = 1.0
        contrast = 1.0
        noise_std = 0.0

    rgb = np.power(np.clip(rgb, 0.0, 1.0), gamma)
    rgb = (rgb - 0.5) * contrast + 0.5
    rgb = rgb * brightness
    rng = np.random.default_rng(99)
    rgb = np.clip(rgb + rng.normal(0.0, noise_std, rgb.shape), 0.0, 1.0)
    out = np.concatenate([rgb, alpha], axis=-1) if alpha is not None else rgb
    Image.fromarray((out * 255.0).astype(np.uint8)).save(rgb_path)


def settle_before_capture(
    simulation_app,
    carb,
    *,
    settle_updates: int,
    settle_passes: int = 1,
) -> None:
    for _ in range(max(1, settle_passes)):
        reset_render_accumulation(carb)
        for _ in range(settle_updates):
            simulation_app.update()


def capture_frame(
    simulation_app,
    rep,
    carb,
    stage,
    work_dir: Path,
    tool_paths,
    staging_bounds,
    surface_height,
    tool_base_matrices,
    placement_axes,
    drawer_controller,
    spec: CaptureSpec,
    Usd,
    UsdGeom,
    *,
    settle_updates: int,
    settle_passes: int,
    rt_subframes: int,
    post_capture_updates: int,
    maximize_contrast: bool = False,
) -> Path:
    random.seed(spec.seed)
    visible_count = random.randint(spec.min_tools, spec.max_tools)
    visible_tools = random.sample(list(tool_paths.keys()), k=visible_count)

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
        min_gap=0.003,
        spread_pad=spec.spread_pad,
        tight_cluster=spec.tight_cluster and visible_count >= 2,
        tight_max_gap=spec.tight_max_gap,
    )

    if drawer_controller is not None and spec.drawer_open is not None:
        lo, _hi = normalize_drawer_open_range(
            DEFAULT_DRAWER_OPEN_RANGE,
            drawer_controller.lower_limit,
            drawer_controller.upper_limit,
        )
        if spec.drawer_open:
            apply_drawer_joint_position(stage, drawer_controller, lo)
        else:
            apply_drawer_joint_position(stage, drawer_controller, drawer_controller.upper_limit)

    if spec.dome_intensity is not None and spec.dome_color is not None:
        set_dome_light(stage, spec.dome_intensity, spec.dome_color)
    if spec.exposure is not None:
        set_exposure(carb, spec.exposure)

    settle_before_capture(
        simulation_app,
        carb,
        settle_updates=settle_updates,
        settle_passes=settle_passes,
    )
    rep.orchestrator.step(rt_subframes=rt_subframes)
    rep.orchestrator.wait_until_complete()
    for _ in range(post_capture_updates):
        simulation_app.update()

    rgb_files = sorted(work_dir.glob("rgb_*.png"))
    if not rgb_files:
        raise RuntimeError("No rgb output from writer")
    latest = rgb_files[-1]
    apply_rgb_jitter(latest, spec.rgb_jitter, maximize=maximize_contrast)
    return latest


def make_side_by_side(
    left_path: Path,
    right_path: Path,
    out_path: Path,
    title: str,
    left_label: str,
    right_label: str,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    left = Image.open(left_path).convert("RGB")
    right = Image.open(right_path).convert("RGB")
    h = max(left.height, right.height)
    w = left.width + right.width
    header = 72
    canvas = Image.new("RGB", (w, h + header), (24, 24, 28))
    canvas.paste(left, (0, header))
    canvas.paste(right, (left.width, header))

    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except OSError:
        font_title = ImageFont.load_default()
        font_label = font_title

    draw.text((20, 12), title, fill=(240, 240, 240), font=font_title)
    draw.text((20, h + header - 34), left_label, fill=(180, 220, 255), font=font_label)
    draw.text((left.width + 20, h + header - 34), right_label, fill=(180, 255, 200), font=font_label)
    draw.line([(left.width, header), (left.width, h + header)], fill=(80, 80, 90), width=3)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    args.scene_usd = args.scene_usd.resolve()
    base_output = args.output_dir.resolve()
    if args.variant:
        args.output_dir = base_output / args.variant
    else:
        args.output_dir = base_output

    pairs = build_pairs(seed_offset=args.seed_offset, maximize_contrast=args.maximize_contrast)

    if args.output_dir.exists() and not args.no_clean:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = args.output_dir / "_capture_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    placement_axes = PlacementAxes(DEFAULT_UP_AXIS, DEFAULT_PLANE_U_AXIS, DEFAULT_PLANE_V_AXIS)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

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

    for _ in range(PRE_CAPTURE_APP_UPDATES):
        simulation_app.update()

    writer = setup_replicator_writer(camera_prim_path, work_dir, args.resolution)

    for _ in range(WRITER_WARMUP_FRAMES):
        random.seed(0)
        randomize_tool_layout(
            stage,
            tool_paths,
            list(tool_paths.keys())[:3],
            staging_bounds,
            surface_height,
            tool_base_matrices,
            placement_axes,
            Usd,
            UsdGeom,
        )
        set_dome_light(stage, 300.0, (1.0, 1.0, 1.0))
        settle_before_capture(
            simulation_app,
            carb,
            settle_updates=args.settle_updates,
            settle_passes=args.settle_passes,
        )
        rep.orchestrator.step(rt_subframes=args.rt_subframes)
        rep.orchestrator.wait_until_complete()

    drawer_controller = setup_drawer_controller(stage, DEFAULT_DRAWER_JOINT)

    print(
        f"Capture timing: settle_updates={args.settle_updates}, "
        f"settle_passes={args.settle_passes}, "
        f"rt_subframes={args.rt_subframes}, post_capture={DEFAULT_POST_CAPTURE_UPDATES}"
    )
    if args.variant:
        print(f"Variant: {args.variant} (seed_offset={args.seed_offset}, maximize={args.maximize_contrast})")

    saved: list[Path] = []
    capture_kwargs = {
        "settle_updates": args.settle_updates,
        "settle_passes": args.settle_passes,
        "rt_subframes": args.rt_subframes,
        "post_capture_updates": DEFAULT_POST_CAPTURE_UPDATES,
        "maximize_contrast": args.maximize_contrast,
    }
    for pair_id, title, spec_a, spec_b in pairs:
        print(f"\n[{pair_id}] {title}")
        path_a = capture_frame(
            simulation_app,
            rep,
            carb,
            stage,
            work_dir,
            tool_paths,
            staging_bounds,
            surface_height,
            tool_base_matrices,
            placement_axes,
            drawer_controller,
            spec_a,
            Usd,
            UsdGeom,
            **capture_kwargs,
        )
        out_a = args.output_dir / spec_a.filename
        shutil.copy2(path_a, out_a)
        saved.append(out_a)
        print(f"  ✓ {out_a.name}")

        path_b = capture_frame(
            simulation_app,
            rep,
            carb,
            stage,
            work_dir,
            tool_paths,
            staging_bounds,
            surface_height,
            tool_base_matrices,
            placement_axes,
            drawer_controller,
            spec_b,
            Usd,
            UsdGeom,
            **capture_kwargs,
        )
        out_b = args.output_dir / spec_b.filename
        shutil.copy2(path_b, out_b)
        saved.append(out_b)
        print(f"  ✓ {out_b.name}")

        combo = args.output_dir / f"{pair_id}_side_by_side.png"
        make_side_by_side(out_a, out_b, combo, title, spec_a.label, spec_b.label)
        print(f"  ✓ {combo.name}")

    writer.detach()
    rep.orchestrator.wait_until_complete()
    shutil.rmtree(work_dir, ignore_errors=True)
    simulation_app.close()

    print(f"\n✓ Saved {len(saved)} comparison images + side-by-side composites → {args.output_dir}")


if __name__ == "__main__":
    main()
