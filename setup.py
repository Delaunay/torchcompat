#!/usr/bin/env python
from pathlib import Path

from setuptools import setup

with open("torchcompat/core/__init__.py") as file:
    for line in file.readlines():
        if "version" in line:
            version = line.split("=")[1].strip().replace('"', "")
            break

TT_INDEX = "https://pypi.eng.aws.tenstorrent.com/"

# Per-accelerator optional dependencies. Install examples:
#   pip install -e ".[cpu]"
#   pip install -e ".[cuda]"
#   pip install -e ".[xla]"
#   pip install -e ".[tt]" --extra-index-url https://pypi.eng.aws.tenstorrent.com/
#   pip install -e ".[all]" --extra-index-url https://pypi.eng.aws.tenstorrent.com/
extra_requires = {
    "base": ["torch"],
    "cpu": ["torch"],
    "cuda": ["torch"],
    "rocm": ["torch"],
    "xpu": ["torch", "intel-extension-for-pytorch"],
    "gaudi": ["torch", "habana-frameworks-torch"],
    "xla": ["torch", "torch-xla>=2.7"],
    "tt": [
        "torch",
        "torch-xla>=2.7",
        "pjrt-plugin-tt>=1.1.0",
    ],
    "cli": ["argklass>=1.4.4"],
}

extra_requires["plugins"] = sorted(
    {
        "importlib_resources",
        *extra_requires["xla"],
        *extra_requires["tt"],
        *extra_requires["xpu"],
        *extra_requires["gaudi"],
    }
)
extra_requires["all"] = sorted(
    set(sum(extra_requires.values(), [])) | {"argklass>=1.4.4"}
)

if __name__ == "__main__":
    setup(
        name="torchcompat",
        version=version,
        extras_require=extra_requires,
        description="torch compatibility layer",
        long_description=(Path(__file__).parent / "README.rst").read_text(),
        author="Anonymous",
        author_email="anony@mous.com",
        license="BSD 3-Clause License",
        url="https://torchcompat.readthedocs.io",
        classifiers=[
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Operating System :: OS Independent",
        ],
        packages=[
            "torchcompat",
            "torchcompat.cli",
            "torchcompat.core",
            "torchcompat.lazy",
            "torchcompat.utils",
            "torchcompat.plugins",
            "torchcompat.plugins.cuda",
            "torchcompat.plugins.rocm",
            "torchcompat.plugins.xpu",
            "torchcompat.plugins.cpu",
            "torchcompat.plugins.gaudi",
            "torchcompat.plugins.xla",
            "torchcompat.plugins.tt",
        ],
        setup_requires=["setuptools"],
        install_requires=[
            "importlib_resources",
            "argklass>=1.4.4",
        ],
        entry_points={
            "console_scripts": [
                "torchcompat=torchcompat.cli:main",
            ],
        },
        package_data={
            "torchcompat.data": [
                "torchcompat/data",
            ],
        },
        project_urls={
            "TT Index": TT_INDEX,
        },
    )
