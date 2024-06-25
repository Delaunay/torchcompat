#!/usr/bin/env python
from pathlib import Path

from setuptools import setup

with open("torchcompat/core/__init__.py") as file:
    for line in file.readlines():
        if "version" in line:
            version = line.split("=")[1].strip().replace('"', "")
            break

extra_requires = {"plugins": ["importlib_resources"], "base": ["torch"]}
extra_requires["all"] = sorted(set(sum(extra_requires.values(), [])))

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
            "torchcompat.core",
            "torchcompat.plugins",
            "torchcompat.plugins.cuda",
            "torchcompat.plugins.rocm",
            "torchcompat.plugins.xpu",
            "torchcompat.plugins.cpu",
            "torchcompat.plugins.gaudi",
        ],
        setup_requires=["setuptools"],
        install_requires=[
            "importlib_resources",
        ],
        package_data={
            "torchcompat.data": [
                "torchcompat/data",
            ],
        },
    )
