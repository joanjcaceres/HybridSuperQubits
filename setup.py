from setuptools import setup, find_packages

setup(
    name="HybridSuperQubits",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "scqubits>=3.0",
        "h5py>=3.0",
    ],
)