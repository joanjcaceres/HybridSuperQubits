from setuptools import setup, find_packages

setup(
    name="HybridSuperQubits",
    version="0.1",
    packages=find_packages(include=["HybridSuperQubits", "HybridSuperQubits.*"]),
    install_requires=[
        "scqubits>=3.0",
        "h5py>=3.0",
        "pyyaml",
        "ipywidgets>=8.0",
        "tqdm>=4.0"       
    ],

    author="Joan Caceres",
    description="Package to simulate hybrid superconducting qubits",
    python_requires=">=3.8"
)