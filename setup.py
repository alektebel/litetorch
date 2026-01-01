"""
Setup script for LiteTorch package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="litetorch",
    version="0.1.0",
    author="LiteTorch Contributors",
    description="A lightweight implementation of PyTorch and RL algorithms for educational purposes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alektebel/litetorch",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "benchmark": [
            "torch>=2.0.0",
            "stable-baselines3>=2.0.0",
            "gymnasium>=0.28.0",
        ],
    },
)
