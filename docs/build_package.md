# Instructions for Building Package

When actively contributing to the package or wanting to use the newest features before a public release you may want to build the package locally. You can download the desired branch on [GitHub](https://github.com/davidkowalk/fitting_toolkit/branches) or browse the [forks](https://github.com/davidkowalk/fitting_toolkit/forks).
| Branch          | Purpose
|-----------------|-------------
| development-1.0 | Bug fixes documentation adding onto version 1.0.1
| development-1.1 | Development of new major features.

## Setting Up The Environment
When working with unstable Versions it is highly recommended that you first create a [virtual environment](https://docs.python.org/3/library/venv.html). Virtual environments create isolated spaces for Python projects, ensuring that dependencies installed for one project do not interfere with others.This isolation prevents conflicts between libraries, especially when two projects require different versions of the same library. Furthermore virtual environments make it easy to replicate the same development environment across multiple machines thus ensuring reproducabillity of unit tests.

Locate your local version of the repository and run

*On Linux*:
```
python3 -m venv .venv
```
*On Windows*:
```
python -m venv .venv
```
Note that on windows you may have to manually copy the `tcl` folder from your global Python installation to the root directory of your virtual environment `.venv`

You will now need to activate your virtual environment.\
*On Linux run*:
```
source .venv/bin/activate
```
*On Windows*:
```
.venv\Scripts\activate
```

You can now install the dependencies. Run

```
pip install -r requirements.txt
pip install setuptools wheel
```

The fitting toolkit uses `setuptools` and `wheel` to build the package. These are additionally installed.
If you are using Visual Studio Code make sure to also switch to the virtual environment there.

## Running Unit Tests

When building unstable versions of the package you need to make sure that the version at hand is working correcty. The repository contains a folder `tests` which contains the unit tests. The tests are implemented using the `unittest` package. To run the unit-tests locate the repository root directory, activate your virtual environment and run

*On Windows*:
```
python -m unittest discover -s tests
```
*On Linux*:
```
python3 -m unittest discover -s tests
```

If tests fail it is recommended to run again with `-v` tag for a more detailed output.

## Building The Package

To build the package locate `setup.py` in the project root directory and run it in your virtual environment. This will automatically read the contents of `src/` and package it. Make sure to update `__init__.py`. You will see two new directories. The `build` directory will contain the packaged files, the dist `dist` directory will contan the wheel file and the packaged tarball.

## Installing the Package 
Install the Package using pip:
```sh
pip install --no-deps --force-reinstall ./dist/fitting_toolkit-VERSION_NUMBER-py3-none-any.whl
```

The `--no-deps` makes pip ignore the dependencies. To reinstall the dependencies omit it.
Make sure to replace the version number with the number in `setup.py` 

### Windows
```sh
./.venv/Scripts/activate
python setup.py sdist bdist_wheel
pip install --no-deps --force-reinstall ./dist/fitting_toolkit-VERSION_NUMBER-py3-none-any.whl
pip show fitting-toolkit -v
deactivate
```

### Linux
```sh
source ./.venv/Scripts/activate
python setup.py sdist bdist_wheel
pip install --no-deps --force-reinstall ./dist/fitting_toolkit-VERSION_NUMBER-py3-none-any.whl
pip show fitting-toolkit -v
deactivate
```
