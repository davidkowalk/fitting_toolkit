from setuptools import setup, find_packages
#   

with open("./package_description.md") as f:
    description = f.read()

with open("./requirements.txt", encoding="utf-16") as f:
    requirements = f.readlines()

setup(
    name = "fitting_toolkit",
    version = "1.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    long_description=description,
    long_description_content_type="text/markdown",
    install_requires = requirements,
    project_urls = {
        "Documentation": "https://github.com/davidkowalk/fitting_toolkit/blob/Stable/docs/manual.md",
        "Source": "https://github.com/davidkowalk/fitting_toolkit/",
        "Tracker": "https://github.com/davidkowalk/fitting_toolkit/issues"
    },
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Natural Language :: English"
    ],
    license="MIT",
    description="Easy and Flexible Curve Fitting",
    url="https://github.com/davidkowalk/fitting_toolkit/"
)