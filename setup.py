import os
import toml
from setuptools import find_packages, setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

setup(
    name="tool_transfer_bot",
    version=TOML_DATA["package"]["version"],
    description=TOML_DATA["package"]["description"],
    packages=find_packages(where="source"),
    package_dir={"": "source"},
    python_requires=">=3.10",
    zip_safe=False,
)
