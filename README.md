![GitHub commit activity](https://img.shields.io/github/commit-activity/m/davidkowalk/fitting_toolkit)
![GitHub License](https://img.shields.io/github/license/davidkowalk/fitting_toolkit)
![University](https://img.shields.io/badge/Univeristy_of_Bonn-brown)
![Version](https://img.shields.io/badge/version-open%20beta-red)
![GitHub Repo stars](https://img.shields.io/github/stars/davidkowalk/fitting_toolkit?style=flat&label=github%20stars)



# Fitting Toolkit
This toolkit aims at providing flexible and powerful tools for data analysis and modelling, but remain easy to use.

Here, I aim to strike a balance between the two extremes in this field. On one side are toolkits such as Kafe2, which prioritize ease of use and convenience but limit user control over the output, often resulting in highly specialized graphics that frequently do not meet standards required for publication without considerable effort. On the other side are data analysis systems like CERN's ROOT, which offer exceptional speed and capability but come with a steep learning curve and often exceed the requirements of most experiments.

This toolkit is aimed primarily at my peers, students of physics at the university of bonn, and to a degree at professionals within my field. I am optimizing this toolkit to be used on the scale typical of lab courses and homework assignments but if possible it should be powerful enough to run decently sized datasets on an average laptop.

This toolkit wraps numpy for fast data management and manipulation, scipy for `curve_fit()` and matplotlib for display options.

Check out the `docs` folder for documentation and tutorials.

## Quick Introduction

### Requirements
This project requires the following modules along with their dependencies:
- numpy
- matplotlib
- scipy

It is highly recommended that the user familiarizes themselves with the functionality of these modules first. A rudimentary understanding of `numpy` and `matplotlib.pyplot` is required.

To install the dependencies, first a [virtual environment](https://docs.python.org/3/library/venv.html) should be created. `requirements.txt` lists all necessary packages. Run:
```
pip install -r requirements.txt
```

### Getting Started

To get started find the `fitting_toolkit.py` in the `src` folder and copy it into your project.
You can now import the relevant functions into your code:
```python
from fitting_toolkit import curve_fit, plot_fit 
import numpy as np
```
The `curve_fit` requires numpy-arrays. Therefore numpy has to be imported as well.

We can now start by simply defining our data.
```python
x = np.array((1, 2, 3, 4, 5))
y = np.array((1, 2, 1.75, 2.25, 3))
dy = 0.1*y+0.05
dx = 0.1
```
We chose a simple linear model:
```python
def f(x, a, b):
    return a * x + b
```
We can now fit the model to the data:
```python
params, cov, lower_conf, upper_conf = curve_fit(f, x, y, yerror=dy)
```
This functions returns 4 arrays. First the parameters of the model, the covariance matrix of those parameters and then the lower and upper limits of the confidence interval around the fit. Note that the confidence interval is absolute. To get the error in relation to the fitted function you would need to find the difference at each point.

The resulting fit can now be plotted. This toolkit provides a premade function to generate plots:
```python
from matplotlib import pyplot as plt
fig, ax = plot_fit(x, y, f, params, lower_conf, upper_conf, xerror=dx, yerror=dy)
plt.show()
```
Note that the fitted function is not automatically displayed. Instead the figure and axis-objects are returned.

![Example Graph](./docs/img/example_fit.png)

For a deeper explanation and tutorials please reference the [documentation](./docs/manual.md/).

## Literature:
[1] Vugrin, K. W., L. P. Swiler, R. M. Roberts, N. J. Stucky-Mack, and S. P. Sullivan (2007), Confidence region estimation techniques for nonlinear regression in groundwater flow: Three case studies, Water Resour. Res., 43, W03423, https://doi.org/10.1029/2005WR004804. \
[2] Dennis D. Boos. "Introduction to the Bootstrap World." Statist. Sci. 18 (2) 168 - 174, May 2003. https://doi.org/10.1214/ss/1063994971