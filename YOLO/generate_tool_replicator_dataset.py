#!/usr/bin/env python3
"""
실습 3 응용: Isaac Sim Replicator - 공구 합성 데이터 생성
책상 위 6종 공구를 랜덤 배치하고 YOLO 학습용 원천 데이터를 생성합니다.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "YOLO" / "replicator_output" / "tool_table_raw"
DEFAULT_TOOLS_DIR = PROJECT_ROOT / "assets" / "tools"
TABLE_WIDTH_M = 0.50
TABLE_DEPTH_M = 0.75
TABLE_GRID_COLS = 2
TABLE_GRID_ROWS = 3
GRID_WIDTH_M = TABLE_WIDTH_M / TABLE_GRID_COLS
GRID_DEPTH_M = TABLE_DEPTH_M / TABLE_GRID_ROWS
TABLE_THICKNESS_M = 0.04
TABLE_TOP_Z_M = 0.0
TOOL_Z_M = TABLE_TOP_Z_M
TABLE_BOUND_MARGIN_M = 0.005
TOOL_SLOT_JITTER_M = 0.008
TOOL_YAW_JITTER_DEG = 8.0
TOOL_FLIP_PROBABILITY = 0.15
TOOL_PLACEMENT_MAX_ATTEMPTS = 80
TOOL_BBOX_OVERLAP_PADDING_M = 0.003
CAMERA_HEIGHT_M = 0.95
CAMERA_HEIGHT_JITTER_M = 0.05
CAMERA_XY_JITTER_M = 0.015
CAMERA_ROLL_PITCH_JITTER_DEG = 2.0
CAMERA_YAW_JITTER_DEG = 2.0
C270_FOCAL_LENGTH = 40.0
C270_HORIZONTAL_APERTURE = 36.284443
C270_VERTICAL_APERTURE = 20.41
C270_CLIPPING_RANGE = (0.03, 2.0)
RSD455_COLOR_FOCAL_LENGTH = 1.93
RSD455_COLOR_HORIZONTAL_APERTURE = 3.896
RSD455_COLOR_VERTICAL_APERTURE = 2.453
RSD455_DEPTH_FOCAL_LENGTH = 1.93
RSD455_DEPTH_HORIZONTAL_APERTURE = 3.896
RSD455_DEPTH_VERTICAL_APERTURE = 2.054531
RSD455_CLIPPING_RANGE = (0.01, 1_000_000.0)
CAMERA_PROFILES = {
    "c270": {
        "focal_length": C270_FOCAL_LENGTH,
        "horizontal_aperture": C270_HORIZONTAL_APERTURE,
        "vertical_aperture": C270_VERTICAL_APERTURE,
        "clipping_range": C270_CLIPPING_RANGE,
    },
    # Isaac Sim 5.1 RSD455 asset의 RGB Camera_OmniVision_OV9782_Color 속성.
    "rsd455": {
        "focal_length": RSD455_COLOR_FOCAL_LENGTH,
        "horizontal_aperture": RSD455_COLOR_HORIZONTAL_APERTURE,
        "vertical_aperture": RSD455_COLOR_VERTICAL_APERTURE,
        "clipping_range": RSD455_CLIPPING_RANGE,
    },
    "rsd455_depth": {
        "focal_length": RSD455_DEPTH_FOCAL_LENGTH,
        "horizontal_aperture": RSD455_DEPTH_HORIZONTAL_APERTURE,
        "vertical_aperture": RSD455_DEPTH_VERTICAL_APERTURE,
        "clipping_range": RSD455_CLIPPING_RANGE,
    },
}
DOME_LIGHT_INTENSITY = 250.0
KEY_LIGHT_INTENSITY = 2200.0
KEY_LIGHT_SIZE_M = 0.35
KEY_LIGHT_HEIGHT_M = 0.85
KEY_LIGHT_XY_JITTER_M = 0.12
KEY_LIGHT_ROLL_PITCH_JITTER_DEG = 12.0
KEY_LIGHT_YAW_JITTER_DEG = 180.0
TABLE_IVORY_COLOR = (0.94, 0.90, 0.82)
TABLE_ROUGHNESS = 0.9
RGB_NOISE_STD_RANGE = (0.0, 0.015)
RENDER_RT_SUBFRAMES = 8

# 학습 대상 공구 6종 정의
TOOLS = [
    {
        "name": "Screw_Driver",
        "label": "screw_driver",
        "file": "Screw Driver.usdz",
        "z": TOOL_Z_M,
    },
    {
        "name": "Paper_Cutter",
        "label": "paper_cutter",
        "file": "Paper Cutter.usdz",
        "z": TOOL_Z_M,
    },
    {
        "name": "Husky_Socket_Wrench",
        "label": "husky_socket_wrench",
        "file": "Husky Socket Wrench.usdz",
        "z": TOOL_Z_M,
    },
    {
        "name": "Allen_Key_Tool_Assembly",
        "label": "allen_key_tool_assembly",
        "file": "Allen Key Tool Assembly.usdz",
        "z": TOOL_Z_M,
    },
    {
        "name": "Spanner_16mm",
        "label": "spanner_16mm",
        "file": "Spanner 16mm.usdz",
        "z": TOOL_Z_M,
    },
    {
        "name": "socket",
        "label": "socket",
        "file": "socket.usdz",
        "z": TOOL_Z_M,
    },
]

# 0.50m x 0.75m 테이블을 2x3으로 나눈 6개 그리드. 그리드당 공구 1개만 배치.
TOOL_PLACEMENT_SLOTS = [
    {
        "pos": (
            -TABLE_WIDTH_M / 2.0 + GRID_WIDTH_M * (col + 0.5),
            TABLE_DEPTH_M / 2.0 - GRID_DEPTH_M * (row + 0.5),
        ),
        "yaw": 0.0 if row % 2 == 0 else 90.0,
        "grid": (row, col),
    }
    for row in range(TABLE_GRID_ROWS)
    for col in range(TABLE_GRID_COLS)
]


def resolve_output_dir(output_dir: Path) -> Path:
    """Replicator Writer는 상대 경로를 ~/omni.replicator_out/ 아래에 저장하므로 절대 경로로 통일."""
    if output_dir.is_absolute():
        resolved = output_dir
    else:
        resolved = (PROJECT_ROOT / output_dir).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def parse_args() -> argparse.Namespace:
    # 커맨드 라인 인자
    parser = argparse.ArgumentParser(description="Isaac Sim Replicator 공구 데이터 생성 실습")
    parser.add_argument("--headless", action="store_true", help="헤드리스 모드")
    parser.add_argument("--num-frames", type=int, default=50, help="생성할 프레임 수")
    parser.add_argument("--seed", type=int, default=7, help="랜덤 시드")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Replicator 원천 데이터 출력 경로")
    parser.add_argument("--tools-dir", type=Path, default=DEFAULT_TOOLS_DIR, help="공구 USDZ 에셋 폴더")
    parser.add_argument("--resolution", type=int, default=640, help="정사각형 이미지 해상도")
    parser.add_argument("--camera-model", choices=sorted(CAMERA_PROFILES), default="c270", help="카메라 intrinsics 프로파일")
    parser.add_argument("--camera-height", type=float, default=CAMERA_HEIGHT_M, help="탑뷰 카메라 기준 높이(m)")
    parser.add_argument("--camera-height-jitter", type=float, default=CAMERA_HEIGHT_JITTER_M, help="카메라 높이 랜덤화 범위(m)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = resolve_output_dir(args.output_dir)
    random.seed(args.seed)

    print("=" * 50)
    print("Isaac Sim Replicator 공구 데이터 생성 실습")
    print("책상 위 공구 랜덤 배치 데이터셋 구축")
    print("=" * 50)
    print(f"프레임 수: {args.num_frames}")
    print(f"헤드리스 모드: {args.headless}")
    print(f"공구 에셋 폴더: {args.tools_dir}")
    print(f"테이블 크기: {TABLE_WIDTH_M:.2f}m x {TABLE_DEPTH_M:.2f}m")
    print(f"그리드: {TABLE_GRID_COLS} x {TABLE_GRID_ROWS}개, 각 {GRID_WIDTH_M:.2f}m x {GRID_DEPTH_M:.2f}m")
    print(f"프레임당 공구 수: {len(TOOLS)}개")
    print(f"공구 배치 최대 재시도: {TOOL_PLACEMENT_MAX_ATTEMPTS}회")
    print(f"카메라 모델: {args.camera_model}")
    print(f"카메라 높이: {args.camera_height:.2f}m ± {args.camera_height_jitter:.2f}m")
    print(f"출력 경로: {args.output_dir}")
    print()

    # ====================================
    # 1. SimulationApp 초기화
    # ====================================
    print("[Step 1] SimulationApp 초기화 중...")
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "width": 1280,
            "height": 720,
            "window_width": 1280,
            "window_height": 720,
        }
    )
    print("✓ SimulationApp 초기화 완료")

    # ====================================
    # 2. 필요한 모듈 임포트
    # ====================================
    print("[Step 2] 필요한 모듈 임포트 중...")
    import carb
    import omni.replicator.core as rep
    import omni.usd
    try:
        from isaacsim.core.utils.prims import create_prim
        from isaacsim.core.utils.stage import add_reference_to_stage
    except ImportError:
        from omni.isaac.core.utils.prims import create_prim
        from omni.isaac.core.utils.stage import add_reference_to_stage

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
    print("✓ 모든 모듈 임포트 완료")

    # ====================================
    # 3. 스테이지 생성 및 기본 설정
    # ====================================
    print("[Step 3] 새로운 스테이지 생성 중...")
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    # Replicator capture on play 비활성화 (수동 제어)
    rep.orchestrator.set_capture_on_play(False)
    configure_renderer_for_still_images(carb)
    print("✓ 스테이지 생성 완료")

    # ====================================
    # 4. 입력 에셋 확인
    # ====================================
    print("[Step 4] 입력 에셋 확인 중...")
    validate_assets(args.tools_dir)
    print("✓ 입력 에셋 확인 완료")

    # ====================================
    # 5. 조명 설정
    # ====================================
    print("[Step 5] 조명 시스템 구성 중...")
    create_lights(stage, UsdGeom, UsdLux, Sdf, Gf)
    print("✓ 조명 설정 완료")

    # ====================================
    # 6. 환경 구성 (책상)
    # ====================================
    print("[Step 6] 기본 환경 구성 중...")
    create_table(stage, UsdGeom, UsdShade, Sdf, Gf)
    print("✓ 환경 구성 완료")

    # ====================================
    # 7. 카메라 생성
    # ====================================
    print("[Step 7] 카메라 설정 중...")
    create_camera(stage, UsdGeom, args.camera_model, args.camera_height)
    print("✓ 카메라 설정 완료")

    # ====================================
    # 8. 학습용 공구 에셋 생성
    # ====================================
    print("[Step 8] 학습용 공구 생성 중...")
    tool_paths = create_tools(stage, args.tools_dir, add_reference_to_stage)
    print(f"✓ {len(tool_paths)}개의 공구 생성 완료")

    # ====================================
    # 9. Render Product 및 Writer 설정
    # ====================================
    print("[Step 9] 데이터 저장 설정 중...")
    render_product = rep.create.render_product("/World/TopCamera", (args.resolution, args.resolution))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=str(args.output_dir),
        rgb=True,
        bounding_box_2d_tight=True,
        semantic_segmentation=True,
        instance_segmentation=True,
        camera_params=True,
        occlusion=True,
    )
    writer.attach(render_product)
    print("✓ Writer 설정 완료")
    print(f"✓ 출력 디렉토리: {args.output_dir}")

    # ====================================
    # 10. 데이터 생성 실행
    # ====================================
    print(f"\n{'='*50}")
    print("[데이터 생성 시작]")
    print("=" * 50)
    print(f"총 {args.num_frames}개의 프레임을 생성합니다...")

    for frame_idx in range(args.num_frames):
        placement_ok = place_tools_without_overlap(stage, tool_paths, Usd, UsdGeom)
        if not placement_ok:
            print(f"\n! 프레임 {frame_idx + 1}: 겹침 없는 배치를 찾지 못해 마지막 배치를 사용합니다.")

        randomize_lights(stage)
        jitter_camera(stage, args.camera_height, args.camera_height_jitter)
        rep.orchestrator.step(rt_subframes=RENDER_RT_SUBFRAMES)
        print(f"프레임 {frame_idx + 1}/{args.num_frames} 생성 중...", end="\r")

    print(f"\n✓ {args.num_frames}개 프레임 생성 완료")

    # Writer 정리
    writer.detach()

    # 데이터 쓰기 완료 대기
    print("데이터 저장 중...")
    rep.orchestrator.wait_until_complete()
    add_camera_noise_to_rgb(args.output_dir, args.seed)

    # ====================================
    # 11. 결과 확인
    # ====================================
    print(f"\n{'='*50}")
    print("[결과 요약]")
    print("=" * 50)
    print(f"✓ 저장 위치: {args.output_dir}")
    print("✓ 생성 어노테이션: RGB, 2D BBox, Semantic Segmentation, Instance Segmentation")

    # ====================================
    # 12. 학습 포인트 요약
    # ====================================
    print(f"\n{'='*50}")
    print("[학습 포인트 요약]")
    print("=" * 50)
    print("""
✓ Replicator 핵심 개념:
  1. Render Product: 카메라와 해상도 연결
  2. Writer: RGB/Segmentation/BBox 데이터 저장
  3. Randomization: 공구 위치/회전/조명/카메라 변화
  4. Orchestrator: 프레임별 캡처 실행

✓ 생성 대상:
  - 책상 위 6종 공구 USDZ 에셋
  - YOLO 학습 변환용 instance segmentation 원천 데이터
""")

    # ====================================
    # 13. 종료
    # ====================================
    print("\n시뮬레이션을 종료합니다...")
    simulation_app.close()
    print("✓ 프로그램 종료")


def validate_assets(tools_dir: Path) -> None:
    """공구 USDZ 파일이 존재하는지 확인"""
    if not tools_dir.is_dir():
        raise FileNotFoundError(f"Tool directory not found: {tools_dir}")

    missing = [tool["file"] for tool in TOOLS if not (tools_dir / tool["file"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing tool USDZ files in {tools_dir}: {missing}")


def configure_renderer_for_still_images(carb) -> None:
    """프레임 간 잔상을 줄이기 위해 temporal/denoising 계열 설정을 약화"""
    settings = carb.settings.get_settings()
    # 설정 키는 Isaac Sim 버전에 따라 일부만 적용될 수 있다. 적용 가능한 키만 조용히 반영한다.
    for key, value in {
        "/rtx/post/aa/op": 0,
        "/rtx/post/motionblur/enabled": False,
        "/rtx/post/motionBlur/enabled": False,
        "/rtx/pathtracing/optixDenoiser/enabled": False,
        "/rtx/pathtracing/optixDenoiser/enable": False,
        "/rtx/raytracing/denoiser/enabled": False,
    }.items():
        try:
            settings.set(key, value)
        except Exception:
            pass


def create_lights(stage, UsdGeom, UsdLux, Sdf, Gf) -> None:
    """Dome Light와 위치/각도 랜덤화가 가능한 Rect Light 생성"""
    dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
    dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(DOME_LIGHT_INTENSITY)

    key_light = stage.DefinePrim("/World/KeyLight", "RectLight")
    rect_light = UsdLux.RectLight(key_light)
    rect_light.CreateIntensityAttr(KEY_LIGHT_INTENSITY)
    rect_light.CreateWidthAttr(KEY_LIGHT_SIZE_M)
    rect_light.CreateHeightAttr(KEY_LIGHT_SIZE_M)
    xform = UsdGeom.Xformable(key_light)
    xform.AddTranslateOp().Set((0.0, 0.0, KEY_LIGHT_HEIGHT_M))
    # RectLight는 local -Z 방향으로 방사하므로 기본 회전은 탑다운 조명이다.
    xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))


def create_table(stage, UsdGeom, UsdShade, Sdf, Gf) -> None:
    """0.50m x 0.75m 아이보리 무광 판자 테이블 생성"""
    table = stage.DefinePrim("/World/Table", "Cube")
    UsdGeom.Cube(table).CreateSizeAttr(1.0)
    UsdGeom.Xformable(table).AddTranslateOp().Set((0.0, 0.0, TABLE_TOP_Z_M - TABLE_THICKNESS_M / 2.0))
    UsdGeom.Xformable(table).AddScaleOp().Set((TABLE_WIDTH_M, TABLE_DEPTH_M, TABLE_THICKNESS_M))
    UsdGeom.Gprim(table).CreateDisplayColorAttr([TABLE_IVORY_COLOR])

    material = UsdShade.Material.Define(stage, "/World/Materials/TableIvoryMatte")
    shader = UsdShade.Shader.Define(stage, "/World/Materials/TableIvoryMatte/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*TABLE_IVORY_COLOR))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(TABLE_ROUGHNESS)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(table).Bind(material)


def create_camera(stage, UsdGeom, camera_model: str, camera_height: float) -> None:
    """선택한 렌즈 속성을 반영한 탑다운 카메라 생성"""
    profile = CAMERA_PROFILES[camera_model]
    camera = stage.DefinePrim("/World/TopCamera", "Camera")
    xform = UsdGeom.Xformable(camera)
    xform.AddTranslateOp().Set((0.0, 0.0, camera_height))
    # USD cameras look along local -Z by default, so no rotation gives a stable top-down view.
    xform.AddRotateXYZOp().Set((0.0, 0.0, 0.0))

    camera_geom = UsdGeom.Camera(camera)
    camera_geom.CreateFocalLengthAttr(profile["focal_length"])
    camera_geom.CreateHorizontalApertureAttr(profile["horizontal_aperture"])
    camera_geom.CreateVerticalApertureAttr(profile["vertical_aperture"])
    camera_geom.CreateClippingRangeAttr(profile["clipping_range"])
    camera_geom.CreateFocusDistanceAttr(camera_height)
    print(
        "✓ 카메라 intrinsics: "
        f"{camera_model}, focal={profile['focal_length']:.3f}, "
        f"FOV={compute_fov_deg(profile['focal_length'], profile['horizontal_aperture']):.1f}° x "
        f"{compute_fov_deg(profile['focal_length'], profile['vertical_aperture']):.1f}°"
    )


def compute_fov_deg(focal_length: float, aperture: float) -> float:
    """USD camera focal/aperture 값에서 FOV(deg)를 계산"""
    return math.degrees(2.0 * math.atan(aperture / (2.0 * focal_length)))


def create_tools(stage, tools_dir: Path, add_reference_to_stage) -> dict[str, str]:
    """meter 단위, pivot, semantic label이 정규화된 6종 공구 USDZ를 wrapper 아래 추가"""
    tool_paths = {}
    for tool in TOOLS:
        prim_path = f"/World/Tools/{tool['name']}"
        model_path = f"{prim_path}/Model"

        stage.DefinePrim(prim_path, "Xform")
        stage.DefinePrim(model_path, "Xform")
        add_reference_to_stage(usd_path=str(tools_dir / tool["file"]), prim_path=model_path)

        tool_paths[tool["name"]] = prim_path
        print(f"Loaded {tool['label']}: {tools_dir / tool['file']}")

    return tool_paths


def place_tools_without_overlap(stage, tool_paths: dict[str, str], Usd, UsdGeom) -> bool:
    """공구들을 테이블 안에 놓고, bbox끼리 겹치면 다시 샘플링"""
    for _ in range(TOOL_PLACEMENT_MAX_ATTEMPTS):
        slots = random.sample(TOOL_PLACEMENT_SLOTS, k=len(TOOL_PLACEMENT_SLOTS))
        bboxes: list[tuple[float, float, float, float]] = []

        for tool, slot in zip(TOOLS, slots):
            prim_path = tool_paths[tool["name"]]
            x = slot["pos"][0] + random.uniform(-TOOL_SLOT_JITTER_M, TOOL_SLOT_JITTER_M)
            y = slot["pos"][1] + random.uniform(-TOOL_SLOT_JITTER_M, TOOL_SLOT_JITTER_M)
            yaw = slot["yaw"] + random.uniform(-TOOL_YAW_JITTER_DEG, TOOL_YAW_JITTER_DEG)
            roll = 180.0 if random.random() < TOOL_FLIP_PROBABILITY else 0.0
            # 낮은 확률로 공구를 뒤집어 뒷면도 학습 데이터에 포함한다.
            set_pose(stage, prim_path, (x, y, tool["z"]), (roll, 0.0, yaw), visible=True)
            snap_tool_to_table(stage, prim_path, Usd, UsdGeom)
            bboxes.append(get_tool_xy_bbox(stage, prim_path, Usd, UsdGeom))

        if not any_bboxes_overlap(bboxes):
            return True

    return False


def get_tool_xy_bbox(stage, prim_path: str, Usd, UsdGeom) -> tuple[float, float, float, float]:
    """공구 world bbox의 XY 범위 반환: (min_x, min_y, max_x, max_y)"""
    prim = stage.GetPrimAtPath(prim_path)
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    aligned_box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    bbox_min = aligned_box.GetMin()
    bbox_max = aligned_box.GetMax()
    return (bbox_min[0], bbox_min[1], bbox_max[0], bbox_max[1])


def any_bboxes_overlap(bboxes: list[tuple[float, float, float, float]]) -> bool:
    """여러 bbox 중 하나라도 서로 겹치는지 확인"""
    for idx, bbox_a in enumerate(bboxes):
        for bbox_b in bboxes[idx + 1 :]:
            if bboxes_overlap(bbox_a, bbox_b):
                return True
    return False


def bboxes_overlap(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> bool:
    """두 bbox가 padding을 포함해 겹치는지 확인"""
    pad = TOOL_BBOX_OVERLAP_PADDING_M
    return (
        bbox_a[0] - pad < bbox_b[2]
        and bbox_a[2] + pad > bbox_b[0]
        and bbox_a[1] - pad < bbox_b[3]
        and bbox_a[3] + pad > bbox_b[1]
    )


def set_pose(
    stage,
    prim_path: str,
    position: tuple[float, float, float],
    rotation_xyz: tuple[float, float, float],
    visible: bool,
) -> None:
    """Prim 위치, 회전, 가시성 설정"""
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    set_or_add_xform_op(xform, "xformOp:translate", position)
    set_or_add_xform_op(xform, "xformOp:rotateXYZ", rotation_xyz)

    imageable = UsdGeom.Imageable(prim)
    imageable.MakeVisible() if visible else imageable.MakeInvisible()


def snap_tool_to_table(stage, prim_path: str, Usd, UsdGeom) -> None:
    """공구 world bbox가 테이블 상단/경계 안에 들어오도록 위치 보정"""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    aligned_box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    bbox_min = aligned_box.GetMin()
    bbox_max = aligned_box.GetMax()

    table_min_x = -TABLE_WIDTH_M / 2.0 + TABLE_BOUND_MARGIN_M
    table_max_x = TABLE_WIDTH_M / 2.0 - TABLE_BOUND_MARGIN_M
    table_min_y = -TABLE_DEPTH_M / 2.0 + TABLE_BOUND_MARGIN_M
    table_max_y = TABLE_DEPTH_M / 2.0 - TABLE_BOUND_MARGIN_M

    x_delta = 0.0
    y_delta = 0.0
    if bbox_min[0] < table_min_x:
        x_delta = table_min_x - bbox_min[0]
    elif bbox_max[0] > table_max_x:
        x_delta = table_max_x - bbox_max[0]

    if bbox_min[1] < table_min_y:
        y_delta = table_min_y - bbox_min[1]
    elif bbox_max[1] > table_max_y:
        y_delta = table_max_y - bbox_max[1]

    z_delta = TABLE_TOP_Z_M - bbox_min[2]

    if abs(x_delta) <= 1e-6 and abs(y_delta) <= 1e-6 and abs(z_delta) <= 1e-6:
        return

    xform = UsdGeom.Xformable(prim)
    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate":
            current = op.Get()
            op.Set((current[0] + x_delta, current[1] + y_delta, current[2] + z_delta))
            return

    xform.AddTranslateOp().Set((x_delta, y_delta, z_delta))


def set_or_add_xform_op(xform, op_name: str, value) -> None:
    """기존 xformOp를 갱신하거나 없으면 새로 생성"""
    for op in xform.GetOrderedXformOps():
        if op.GetOpName() == op_name:
            op.Set(value)
            return

    if op_name == "xformOp:translate":
        xform.AddTranslateOp().Set(value)
    elif op_name == "xformOp:rotateXYZ":
        xform.AddRotateXYZOp().Set(value)
    else:
        raise ValueError(f"Unsupported xform op: {op_name}")


def randomize_lights(stage) -> None:
    """프레임마다 조명 밝기, 위치, 각도를 랜덤화"""
    dome = stage.GetPrimAtPath("/World/DomeLight")
    key = stage.GetPrimAtPath("/World/KeyLight")
    if dome:
        dome.GetAttribute("inputs:intensity").Set(random.uniform(180.0, 320.0))
    if key:
        key.GetAttribute("inputs:intensity").Set(random.uniform(1800.0, 2800.0))
        x = random.uniform(-KEY_LIGHT_XY_JITTER_M, KEY_LIGHT_XY_JITTER_M)
        y = random.uniform(-KEY_LIGHT_XY_JITTER_M, KEY_LIGHT_XY_JITTER_M)
        z = random.uniform(KEY_LIGHT_HEIGHT_M - 0.08, KEY_LIGHT_HEIGHT_M + 0.08)
        roll = random.uniform(-KEY_LIGHT_ROLL_PITCH_JITTER_DEG, KEY_LIGHT_ROLL_PITCH_JITTER_DEG)
        pitch = random.uniform(-KEY_LIGHT_ROLL_PITCH_JITTER_DEG, KEY_LIGHT_ROLL_PITCH_JITTER_DEG)
        yaw = random.uniform(-KEY_LIGHT_YAW_JITTER_DEG, KEY_LIGHT_YAW_JITTER_DEG)
        set_pose(stage, "/World/KeyLight", (x, y, z), (roll, pitch, yaw), visible=True)


def jitter_camera(stage, camera_height: float, camera_height_jitter: float) -> None:
    """프레임마다 탑다운 카메라 높이, 위치, 기울기, 이미지 회전을 랜덤화"""
    camera_path = "/World/TopCamera"
    x = random.uniform(-CAMERA_XY_JITTER_M, CAMERA_XY_JITTER_M)
    y = random.uniform(-CAMERA_XY_JITTER_M, CAMERA_XY_JITTER_M)
    z = random.uniform(camera_height - camera_height_jitter, camera_height + camera_height_jitter)
    roll = random.uniform(-CAMERA_ROLL_PITCH_JITTER_DEG, CAMERA_ROLL_PITCH_JITTER_DEG)
    pitch = random.uniform(-CAMERA_ROLL_PITCH_JITTER_DEG, CAMERA_ROLL_PITCH_JITTER_DEG)
    image_yaw = random.uniform(-CAMERA_YAW_JITTER_DEG, CAMERA_YAW_JITTER_DEG)
    set_pose(stage, camera_path, (x, y, z), (roll, pitch, image_yaw), visible=True)


def add_camera_noise_to_rgb(output_dir: Path, seed: int) -> None:
    """저장된 RGB 이미지에 약한 Gaussian camera noise를 적용"""
    if RGB_NOISE_STD_RANGE[1] <= 0.0:
        return
    if not output_dir.exists():
        print(f"\n! RGB 노이즈 적용 생략: 출력 경로를 찾을 수 없습니다. ({output_dir})")
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
        noise_std = rng.uniform(*RGB_NOISE_STD_RANGE)
        noisy_rgb = np.clip(rgb + rng.normal(0.0, noise_std, rgb.shape), 0.0, 1.0)
        if alpha is not None:
            noisy = np.concatenate([noisy_rgb, alpha], axis=-1)
        else:
            noisy = noisy_rgb
        Image.fromarray((noisy * 255.0).astype(np.uint8)).save(rgb_path)

    print(f"✓ RGB 카메라 노이즈 적용 완료: {len(rgb_paths)}개 이미지")


if __name__ == "__main__":
    main()
