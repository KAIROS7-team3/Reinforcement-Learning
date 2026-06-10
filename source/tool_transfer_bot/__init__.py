import os
import toml

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# trigger gym.register() calls in all sub-task __init__.py files
from .tasks import *  # noqa: F401, F403
