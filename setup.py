from setuptools import setup

with open("./README.md") as f:
    description = f.read()

with open("./requirements.txt", encoding="utf-16") as f:
    requirements = f.readlines()

setup(
    name = "fitting_toolkit",
    version = "1.0.0",
    package_dir={"": "src"},
    packages=[""],
    long_description=description,
    install_requires = requirements
)