"""Play script wrapper — delegates to Isaac Lab's rsl_rl play.py.

Run from the Reinforcement-Learning repo root (not ~).

Usage:
    cd /home/user/Reinforcement-Learning

    # Option A: absolute checkpoint path (--load_run is ignored when --checkpoint is set)
    /home/user/IsaacLab/isaaclab.sh -p scripts/play.py \\
        --task Isaac-OpenDrawer-Teacher-Play-v0 \\
        --num_envs 1 \\
        --checkpoint logs/rsl_rl/open_drawer_teacher/<run_dir>/model_1499.pt

    # Option B: latest checkpoint in a run (omit --checkpoint)
    /home/user/IsaacLab/isaaclab.sh -p scripts/play.py \\
        --task Isaac-OpenDrawer-Teacher-Play-v0 \\
        --num_envs 1 \\
        --load_run 2026-06-17_21-05-15_physx_tune_v1

Pass --disable_fabric if drawer visuals drift from PhysX during play.
"""

import os
import runpy
import sys

# Ensure our package is importable when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import tool_transfer_bot  # noqa: F401 — triggers gym.register() calls

_ISAACLAB_PLAY = os.path.join(
    os.environ.get("ISAACLAB_PATH", "/home/user/IsaacLab"),
    "scripts",
    "reinforcement_learning",
    "rsl_rl",
    "play.py",
)

if not os.path.exists(_ISAACLAB_PLAY):
    raise FileNotFoundError(
        f"Isaac Lab rsl_rl play script not found at {_ISAACLAB_PLAY}. "
        "Set ISAACLAB_PATH environment variable if installed elsewhere."
    )

sys.path.insert(0, os.path.dirname(_ISAACLAB_PLAY))
runpy.run_path(_ISAACLAB_PLAY, run_name="__main__")
