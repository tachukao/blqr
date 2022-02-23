import os
from setuptools import setup
from setuptools import find_packages

setup(
    name="blqr",
    author="",
    version="0.0.1",
    description="Batch LQR",
    license="MIT",
    install_requires=["numpy", "jaxlib", "jax"],
    packages=find_packages(),
)
