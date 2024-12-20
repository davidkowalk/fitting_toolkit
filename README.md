![GitHub License](https://img.shields.io/github/license/davidkowalk/fitting_toolkit)
![Version](https://img.shields.io/badge/version-1.0.3-green)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/davidkowalk/fitting_toolkit)
![GitHub Repo stars](https://img.shields.io/github/stars/davidkowalk/fitting_toolkit?style=flat&label=github%20stars)
![PyPI - Downloads](https://img.shields.io/pypi/dm/fitting-toolkit?label=pip%20installs)\
![University](https://img.shields.io/badge/Univeristy_of_Bonn-brown)



# Fitting Toolkit
This toolkit aims at providing flexible and powerful tools for data analysis and modelling, but remain easy to use.

Here, I aim to strike a balance between the two extremes in this field. On one side are toolkits such as Kafe2, which prioritize ease of use and convenience but limit user control over the output. On the other side are data analysis systems like CERN's ROOT, which offer exceptional speed and capability but come with a steep learning curve and often exceed the requirements of most experiments.

This package is aimed primarily at my peers, students of physics at the university of bonn, and to a degree at professionals within my field. It is written for small scale applications, typical for a lab course, however still aims at exceptional performance.

Using the different functions provided by scipy and numpy this toolkit implements curve and distribution (peak) fitting with both least squares and maximum likelyhood estimation methods.

Check out the `docs` folder for documentation and tutorials.

## Quick Introduction

### Installation

There are multiple ways to install this package. The easiest is via pip:
```
pip install fitting-toolkit
```
If you need a specific version (for example due to compatibillity issues) you can specify the version via `fitting-toolkit==version`, e.g:
```
pip install fitting-toolkit==1.0.1
```

You can test, whether the package was installed properly via:
```py
import fitting_toolkit as ft
ft.version()
```

### Alternative Installation Methods

You can find all releases here: 

<a href= "https://github.com/davidkowalk/fitting_toolkit/releases">![Download](./docs/img/download.svg)</a>

To install the current development version ahead of releases check out the development branches.
| Branch          | Purpose
|-----------------|-------------
| development-1.0 | Bug fixes and documentation adding onto version 1.0.1
| development-1.1 | Development of new major features

After downloading the desired version you can find the `fitting_toolkit.py` in the `src` folder and copy it into your project.

To build the project yourself and install it, make sure `setuptools` and `wheel` are installed, then run
```
python3 setup.py sdist bdist_wheel
pip install --no-deps --force-reinstall ./dist/fitting_toolkit-VERSION_NUMBER-py3-none-any.whl 
pip show fitting-toolkit -v
```

### Requirements
This project requires the following modules along with their dependencies:
- numpy
- matplotlib
- scipy

It is highly recommended that the user familiarizes themselves with the functionality of these modules first. A rudimentary understanding of `numpy` and `matplotlib.pyplot` is required.

If you install via pip the dependencies will automatically be installed. However if the project files are used directly you may want to install dependencies manually:

To install the dependencies, first a [virtual environment](https://docs.python.org/3/library/venv.html) should be created. `requirements.txt` lists all necessary packages. Run:
```
pip install -r requirements.txt
```

For an introductory explanation and tutorials please reference the [documentation](./docs/manual%20and%20instructions/README.md).

## How to Support the Fitting Toolkit

The Fitting Toolkit is published on GitHub under the MIT License.
It is built and maintained by volunteers. There are multiple ways to contribute:

1. **Share And Boost The Project**\
The easiest way to support the project is to show your colleges and friends how to use it. One big way to help is to mark the project with a star on GitHub. This helps to stay on the front page in developer forums and to attract new contributors.
2. **Write an Issue**\
The repository is being actively maintained. When you find a bug or miss a feature you can write an [issue](https://github.com/davidkowalk/fitting_toolkit/issues).
3. **Do it yourself**\
The fitting toolkit is always looking for contributors. Fork the repository, make your changes and submit a pull request. A good place to start if you're looking to contribute are the ["good first issue"](https://github.com/davidkowalk/fitting_toolkit/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22) and  ["help wanted"](https://github.com/davidkowalk/fitting_toolkit/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22) sections

## Literature:
[1] Vugrin, K. W., L. P. Swiler, R. M. Roberts, N. J. Stucky-Mack, and S. P. Sullivan (2007), Confidence region estimation techniques for nonlinear regression in groundwater flow: Three case studies, Water Resour. Res., 43, W03423, https://doi.org/10.1029/2005WR004804. \
[2] Dennis D. Boos. "Introduction to the Bootstrap World." Statist. Sci. 18 (2) 168 - 174, May 2003. https://doi.org/10.1214/ss/1063994971