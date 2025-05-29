from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="MLOps_Project-1",
    version="0.1",
    author="Raja Sekhar",
    packages=find_packages(),
    install_requires=requirements,
)
