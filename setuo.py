from setuptools import setup, find_packages

setup(
    name="rankwalk",
    version="0.1.0",
    description="Differentiable rank-based random walk GNN layer",
    author="Bastian Pfeifer",
    author_email="bastian.pfeifer@medunigraz.at",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",          # CPU version by default
        "torch-geometric",      # requires PyTorch installed first
    ],
)