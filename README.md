# Fitting Toolkit
This toolkit aims at providing flexible and powerful tools for data analysis and modelling, but remain easy to use.

Here I aim to walk the line between the extremes in this field. On the one side lie toolkits like Kafe2 which are easy and comfortable, however the resulting graphics are highly specified, cluttered and usually unfit for puplication. On the other lie data analysis systems like Cern's ROOT, which are fast and highly capable, however have a steep learning curve and overshoot the requirements for most experiments.

This toolkit is aimed primarily at my peers, students of physics at the university of bonn, and to a degree at professionals within my field. I am optimizing this toolkit to be used on the scale typical of lab courses and homework assignments but if possible it should be powerful enough to run decently sized datasets on an average laptop.

This toolkit wraps numpy for fast data management and manipulation, scipy for `curve_fit()` and matplotlib for display options.

Check out the `docs` folder for documentation and tutorials.

## Quick Introduction

### Requirements
This project requires the following modules:
- numpy
- matplotlib
- scipy

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
dy = 0.1*f+0.05
dx = 0.1
```
For a model we chose a simple linear model:
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
Note that the toolkit does not automatically display the fitted values. Instead the figure and axis-objects are returned.

![Example Graph](./docs/img/example_fit.png)

## Methods Used
The Toolkit uses "Bootstrapping" to generate the confidence interval. The `curve_fit` function takes the `resamples` parameter, which define how many resamples of the parameters are generated. Default is 5000.

`curve_fit` passes this parameter onto `confidence_interval` which generates the defined number of datapoints in the parameter space. For each datapoint the resulting distribution around each point defined by `xdata` is generated and the thresholds are chosen so that 2/3 of the points are between them.