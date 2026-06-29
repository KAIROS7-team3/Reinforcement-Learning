"""Ensure RH-P12 mimic finger joints (rotX instance) reference rh_r1."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from pxr import PhysxSchema, Usd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RL_USDA = ROOT / "assets" / "e0509_rl.usda"
MIMIC_JOINTS = ("rh_r2", "rh_l1", "rh_l2")
REFERENCE_JOINT = "rh_r1"
INSTANCE = "rotX"
GEARING = 1.0


def main() -> None:
    stage = Usd.Stage.Open(str(RL_USDA))
    stage.GetPrimAtPath("/e0509").GetVariantSet("Physics").SetVariantSelection("PhysX")
    stage.Load()

    ref = stage.GetPrimAtPath(f"/e0509/joints/{REFERENCE_JOINT}")
    if not ref.IsValid():
        raise RuntimeError(f"missing reference joint: {REFERENCE_JOINT}")

    for name in MIMIC_JOINTS:
        joint = stage.GetPrimAtPath(f"/e0509/joints/{name}")
        if not joint.IsValid():
            print(f"[WARN] missing joint: {name}")
            continue
        mimic = PhysxSchema.PhysxMimicJointAPI(joint, INSTANCE)
        mimic.GetReferenceJointRel().SetTargets([ref.GetPath()])
        mimic.GetGearingAttr().Set(GEARING)
        print(f"[OK] {joint.GetPath()} -> {ref.GetPath()}")

    stage.GetRootLayer().Export(str(RL_USDA))
    print(f"[INFO] saved {RL_USDA}")


if __name__ == "__main__":
    main()
    app.close()
