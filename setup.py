from setuptools import setup, find_packages

setup(
    name='rankwalk',
    version='0.1.0',
    description='Start-node anchored random walk GNN package',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torch_geometric',
        'networkx',
        'scikit-learn',
        'node2vec',
    ],
)