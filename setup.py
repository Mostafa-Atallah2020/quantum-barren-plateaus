#!/usr/bin/env python3

from setuptools import find_packages, setup

setup(
    name="qubap",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "qiskit",
        "qiskit-aer",
        "matplotlib",
        "qiskit-algorithms",
        "pylatexenc",  # Required for circuit visualization
        "pillow",  # Required for image handling
    ],
)
