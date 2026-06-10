"""Train script wrapper — delegates to Isaac Lab's rsl_rl train.py.

Usage (run inside Docker via isaaclab.sh):
    ./isaaclab.sh -p /home/user/Reinforcement-Learning/scripts/train.py \
        --task Isaac-OpenDrawer-Teacher-v0 \
        --num_envs 64 --headless
"""

import sys
import os
import runpy

# Ensure our package is importable when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import tool_transfer_bot  # noqa: F401 — triggers gym.register() calls

# Hand off to Isaac Lab's built-in rsl_rl train script
_ISAACLAB_TRAIN = os.path.join(
    os.environ.get("ISAACLAB_PATH", "/home/user/IsaacLab"),
    "scripts",
    "reinforcement_learning",
    "rsl_rl",
    "train.py",
)

if not os.path.exists(_ISAACLAB_TRAIN):
    raise FileNotFoundError(
        f"Isaac Lab rsl_rl train script not found at {_ISAACLAB_TRAIN}. "
        "Set ISAACLAB_PATH environment variable if installed elsewhere."
    )

# Add the train script's directory to sys.path so its local imports (cli_args etc.) resolve.
sys.path.insert(0, os.path.dirname(_ISAACLAB_TRAIN))

# Run as __main__ so the script sees correct __file__ and argv.
runpy.run_path(_ISAACLAB_TRAIN, run_name="__main__")
