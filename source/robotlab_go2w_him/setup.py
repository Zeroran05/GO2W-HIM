"""Installation script for the RobotLab Go2W HIM IsaacLab extension."""

import os

import toml
from setuptools import find_packages, setup


EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

setup(
    name="robotlab_go2w_him",
    packages=find_packages(),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=["gymnasium", "numpy", "scipy", "toml"],
    include_package_data=True,
    python_requires=">=3.10",
    zip_safe=False,
)

