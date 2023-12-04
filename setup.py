import os
import sys

from setuptools import find_packages, setup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "aime"))
__version__ = 1.0

assert sys.version_info.major == 3, (
    "This repo is designed to work with Python 3."
    + "Please install it before proceeding."
)

setup(
    name="aime",
    author="Xingyuan Zhang",
    author_email="xingyuan.zhang@argmax.ai",
    packages=find_packages(),
    version=__version__,
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "einops",
        "dm_control",
        "mujoco",
        "gym",
        "matplotlib",
        "tensorboard",
        "tqdm",
        "moviepy",
        "imageio==2.27",
        "hydra-core",
    ],
    url="https://github.com/argmax-ai/aime",
)
