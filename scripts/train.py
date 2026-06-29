"""Train script wrapper — delegates to Isaac Lab's rsl_rl train.py.

Usage:
    cd /home/user/Reinforcement-Learning
    ../IsaacLab/isaaclab.sh -p scripts/train.py \\
        --task Isaac-ReturnTool-Teacher-v0 \\
        --num_envs 2048 --headless \\
        --max_iterations 2000 \\
        --run_name return_tool_ppo_v1

    # --iterations is an alias for --max_iterations
"""

import sys
import os
import runpy

# --iterations → --max_iterations (Isaac Lab native flag)
_argv = sys.argv
for _i, _arg in enumerate(list(_argv)):
    if _arg == "--iterations":
        _argv[_i] = "--max_iterations"
    elif _arg.startswith("--iterations="):
        _argv[_i] = "--max_iterations=" + _arg.split("=", 1)[1]

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
