# setup.py
from setuptools import setup, find_packages

setup(
    name="rankwalk",
    version="0.1.0",
    description="RankWalk GNN implementation",
    author="Your Name",
    packages=find_packages(include=["rankwalk", "rankwalk.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "torch-geometric",
        "numpy",
        "scikit-learn",
        "node2vec",
        "tqdm",
        "networkx",
    ],
    include_package_data=True,
)