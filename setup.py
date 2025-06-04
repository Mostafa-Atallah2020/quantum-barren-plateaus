#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='qubap',
    packages=find_packages(),
    install_requires=[
        "qiskit[visualization]>=0.46,<1.0",
        "qiskit_aer",
        "numpy>=1.17,<3.0",
        "scipy>=1.5",
        "matplotlib>=3.3",
        "pylatexenc",
        "ipywidgets",        
        "jupyter",           
    ],
)