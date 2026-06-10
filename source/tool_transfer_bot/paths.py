"""Centralised filesystem paths for the tool_transfer_bot package.

All asset lookups should go through this module instead of hard-coding
absolute paths. This keeps the project relocatable and makes it trivial to
extend to the remaining tasks/assets later.
"""

import os

# .../source/tool_transfer_bot/paths.py → project root is 3 levels up.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

# In-repo asset directory (USD scene/robot/toolbox files).
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

# Tool USD models. Prefer the in-repo copy (assets/tools); fall back to an
# external directory overridable via the TOOL_MODELS_DIR environment variable.
_REPO_TOOLS_DIR = os.path.join(ASSETS_DIR, "tools")
TOOL_MODELS_DIR = os.environ.get(
    "TOOL_MODELS_DIR",
    _REPO_TOOLS_DIR if os.path.isdir(_REPO_TOOLS_DIR) else "/home/user/Downloads/models_수정",
)


def asset(*parts: str) -> str:
    """Return an absolute path inside the in-repo assets directory."""
    return os.path.join(ASSETS_DIR, *parts)


def tool_model(filename: str) -> str:
    """Return an absolute path to a tool USD model file."""
    return os.path.join(TOOL_MODELS_DIR, filename)
