# Manual for the Fitting Toolkit

This toolkit provides all necessarry functions for fitting and displaying data in a scatter plot with errorbars, but we also maintain acess to the basic functionalities.

These instructions provide a basic introduction into the use of the toolkit and later explains how to use it's parts independently of each other.

## Getting Started

This section covers the use of the toolkit functionalities as intended, which will cover most use-cases.

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

![Example Graph](./img/example_fit.png)

## Errors and Uncertainties

Understanding how the tools we use work with and calculate errors and uncertainties is a vital part of judging and quantifying the characteristics of our work. The Toolkit-curve-fit uses scipy's `curve_fit()` function which handles the fitting process as well as the calculation of errors and correlation of the fitted parameters. You can reference the modules [documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) for a detailed explanation.

### Covaraince Matrix and Sigma

SciPy returns a [covariance matrix](https://en.wikipedia.org/wiki/Covariance) along with the optimal parameters of the fitted function. In [error propagation](https://en.wikipedia.org/wiki/Propagation_of_uncertainty) according to gauß we assume that the measured values are uncorrelated. In the real world this is usually not the case.

> Covariance in probability theory and statistics is a measure of the joint variability of two random variables.

The closer the absolute value of the covariance between two values is to the product of their standard deviations the more correlated they are. SciPy automatically finds a numerical approximation for the covariance between fit parameters.

> Note that the relationship between cov and parameter error estimates is derived based on a linear approximation to the model function around the optimum. When this approximation becomes inaccurate, cov may not provide an accurate measure of uncertainty.

Further the documentation notes:

> How the sigma parameter affects the estimated covariance depends on `absolute_sigma` argument, as described above.

The `absolute_sigma` determines the interpretation of and effect the data errors have on the variance and covariance of the fitted parameters. By default the `absolute_sigma = False` which means that the errors are only used as weights, i.e., as hints how important a point should be in the least-squares computation. To compute the covariance matrix, `scipy.optimize.curve_fit` then estimates the actual error from the average defiation from the optimized curve. [(Read More Here)](https://wwwstaff.ari.uni-heidelberg.de/tsapras/pub/pycourse/15.html) When fitting data with absolute errorbars as can be seen in the examples provided the `absolute_sigma` parameter should be set to `True` via the `**kwargs`.

```python
params, cov, lower_conf, upper_conf = curve_fit(model, x, y, yerror=dy, absolute_sigma = True)
```

Also note that only the y-error affects the uncertainty

### Calculation of the Confidence Interval

To estimate the confidence interval the fitted parameters are then resampled using their estimated means and covariance matrix. The number of resamples can be set via the `resamples = 5000` parameter.


![PDF of Example Graph](./img/example_fit_pdf.png)

For each point on the x-axis the fitted function is calculated for each of the resampled parameter sets and using the `numpy.percentile` function an upper and a lower threshold for the confidence interval is estimated. By default this function is set so that exactly 1/6 of the resampled points lie above the interval and 1/6 lie below it. Therefore 2/3 of the resampled points lie between the interval thresholds, which corresponds to a 1-sigma interval.

This method is often referred to as ["bootstrapping"](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) and presents a simple example of this method.

By default the confidence interval is estimated at each x-position provided, however this may cause issues of resolution when few datapoints are provided or the datapoints are non-uniformally distributed along the x-axis, or it may be computanionally expensive for large datasets. When `curve_fit`is provided a `confidence_resolution: int` parameter, the number of points provided are generated between the highest and lowest point on the x-axis and used as the x-axis instead.

```python
resampled_points = np.linspace(min(xdata), max(xdata), confidence_resolution) 
```

Note that the `confidence_resolution` must be provided to both `curve_fit` and `plot_fit`  


## Code Documentation

By separating the fitting functionality from the display options a user can utilize the parts independently of each other.

### Using the Fitting Functionality

To fit a dataset call
```python
curve_fit(model, xdata: np.array, ydata: np.array, yerror = None, resamples = 5000, **kwargs)
```

| Parameters | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| model    | function | Function to be fitted. Must take `xdata` as a first argument and then an arbitrary number of fit parameters.|
| xdata    | np.array | The independent variable where the data is measured. Each element should be float convertible if it is an array like object.
| ydata    | np.array | The dependent data, a length M array - nominally f(xdata, ...)
| yerror   | np.array | (optional) Determines the uncertainty in ydata. Pass absolute values.
| resamples| int      | (optional) Number of samples to be generated in parameter space for bootstrapping.
| **kwargs | any      | (optional) Parameters to be passed on to `scipy.optimize.curve_fit`

| Returns | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| params   | np.array | List of optimal parameters. Can be separated by `p1, p1, ..., pn = params`
| cov      | np.array | Covariance Matrix of parameters as provided by `scipy.optimize.curve_fit`. Standard deviations can be calculated by `sigma = np.sqrt(np.diagonal(cov))`
| lower_conf | np.array | Absolute y-values of lower edge of 1-sigma confidence interval.|
| upper_conf | np.array | Absolute y-values of upper edge of 1-sigma confidence interval.

#### Using Custom Graphics

To generate your own plot you can use the returned values of `curve_fit`. Using a defined model `f(x, *params)`:

```python
params, cov, lower_conf, upper_conf = curve_fit(f, x, y)
#calculate standard deviations for possible later use
standard-deviations = np.sqrt(np.diagonal(cov))

from matplotlib import pyplot as plt
#scatter data
plt.scatter(x, y)
# plots the fitted parameters
plt.plot(x, f(x, *params), color = "black")
#plots the confidence interval
plt.plot(x, lower_conf, color = "red")
plt.plot(x, upper_conf, color = "red")
#Display
plt.show()
```

### Calculate Confidence Interval for an Existing Fit

Given already fitted parameters and a covariance matrix a confidence interval can be calculated using `confidence_interval(model, xdata: np.array, params: np.array, cov: np.array, resamples: int)`

| Parameters | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| model    | function | Model fitted to data
| xdata    | np.array | The independent variable at which the confidence interval is to be calculated.
| params   | np.array | Fitted parameters passed onto `model`.
| cov      | np.array | Covariance matrix of `params`
| resamples| int      | Number of resamples to be calculated.

| Returns | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| lower_conf | np.array | Absolute y-values of lower edge of 1-sigma confidence interval.|
| upper_conf | np.array | Absolute y-values of upper edge of 1-sigma confidence interval.

Example:
Given a model `f(x, *params)`
```python
from scipy.optimize import curve_fit
params, cov = sc_curve_fit(f = f, xdata = xdata, ydata = ydata, sigma = yerror, absolute_sigma=True, **kwargs) 
lower_conf, upper_conf = confidence_interval(model, xdata, params, cov, resamples)
```
Errors relative to the fitted function can be calculated as thus:
```python
sigma_pos = upper_conf - f(x, *params)
sigma_neg = f(x, *params) - lower_conf
```
So that the fitted value `x[i]` are
```python
print(f"f({x[i]:.2e}) = {f(x[i], *params):.2e} (+{sigma_pos[i]:.2e}/-{sigma_neg[i]:.2e})")
```

### Generating Thresholds

Given a bootstrapped distribution generate a custom interval for the confidence interval.

Call `generate_thresholds(data, lower_frac=1/6, upper_frac=5/6)`

| Parameters | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| data     | np.array | At a defined point x give the y value for all points in parameter space.
| lower_frac | float  | Fraction of data below lower threshold
| upper_frac | float  | Fraction of data below upper threshold


| Returns | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| lower_threshold | float | Point defined by `lower_frac` 
| upper_threshold | float | Point defined by `upper_frac`