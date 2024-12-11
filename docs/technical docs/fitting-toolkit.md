
# Code Documentation

By separating the fitting functionality from the display options, a user can utilize the parts independently of each other.
This document describes the primary module functionalities, which can be directly accessed as methods of the `fitting_toolkit` package.

## fitting_toolkit.curve_fit

To fit a dataset, call:
```python
curve_fit(model, xdata: np.array, ydata: np.array, yerror = None, method = "scipy", resamples = 5000, model_resolution: int = None, model_axis = None, nsigma:float = 1, **kwargs)
```

| Parameters | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| model    | function | Function to be fitted. Must take `xdata` as a first argument and then an arbitrary number of fit parameters.|
| xdata    | np.array | The independent variable where the data is measured. Each element should be float convertible if it is an array-like object.
| ydata    | np.array | The dependent data, a length M array - nominally f(xdata, ...)
| yerror   | np.array, optional | Determines the uncertainty in ydata.
| method   | str, optional | Select method used for fitting the model. Must either be "scipy" for scipy's builtin least squares fit or "mle" for maximum likelyhood estimation. By default "scipy" is used.
| resamples| int, optional | Number of samples to be generated in parameter space for bootstrapping.
|model_resolution | int, optional | If specified the confidence interval will be calculated at linearly spaced points along x-axis. Otherwise xdata is used.
| model_axis | numpy.ndarray, optional | If specified this axis is used instead of axis generated via model_resolution.
| nsigma | float, optional | Specifies the number of standard deviations corresponding to the desired confidence interval, when assuming a normal distribution.
| **kwargs | any      | (optional) Parameters to be passed on to `scipy.optimize.curve_fit`

| Parameters for MLE | | |
|----------|-----------------------|-----------------|
| **Name** |        **Type**       | **Description** |
| xerror   | np.ndarray (optional) | Determines the uncertainty in ydata.
| theta_0  | np.ndarray (optional) | Initial guess for parameters

| Returns  | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| fit      | fitting_toolkit.Fit | Wrapper object containing the fitted model, fit results and confidence interval. 

When using \"scipy\" method x-errors are not used, y-errors are optional.\
When using \"mle\" method y-errors are required, x-errors are optional. Note that using xerrors is considerably more computationally expensive.

### Using Custom Graphics

To generate your own plot you can use the returned values of `curve_fit`. Using a defined model `f(x, *params)`:

```python
fit = curve_fit(f, x, y)
#calculate standard deviations for possible later use
standard_deviations = np.sqrt(np.diagonal(fit.cov))

#extract interval from fit object
lower_conf = fit.lower
upper_conf = fit.upper
#get points at which confidence interval has been calculated
model_axis = fit.axis 

from matplotlib import pyplot as plt
#scatter data
plt.scatter(x, y)
# plots the fitted parameters
plt.plot(x, f(model_axis, *fit.params), color = "black")
#plots the confidence interval
plt.plot(model_axis, lower_conf, color = "red")
plt.plot(model_axis, upper_conf, color = "red")
#Display
plt.show()
```

## fitting_toolkit.plot_fit

The fitting toolkit ships with built-in functions for displaying data with their fitted functions and their respective confidence intervals.
```python
plot_fit(xdata, ydata, fit, xerror = None, yerror = None, markersize = 4, capsize = 4, fit_color = "black", fit_label = "Least Squares Fit", confidence_label = "1$\\sigma$-Confidence", fig = None, ax = None, **kwargs)
```

| Parameters | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| xdata    | numpy.ndarray | The x-values of the data points.
| ydata    | numpy.ndarray | The y-values of the data points.
| fit      | fitting_toolkit.Fit | Wrapper object containing the fitted model, fit 
| **Optional Arguments** |
| xerror   | numpy.ndarray, optional | The uncertainties in the x-values of the data points. Default is None.
| yerror   | numpy.ndarray, optional | The uncertainties in the y-values of the data points. Default is None.

| **Display Options** |
|fit_color | color, optional | color of the fitted function.
|markersize| int, optional | The size of the markers for the data points. Default is 4.
|capsize   | int, optional | The size of the caps on the error bars. Default is 4.
|fit_label | str, optional | Label applied to the least square fit.
|confidence_label | str, optional | Label applied to upper confidence threshold.
|fig       | matplotlib.pyplot.Figure, optional | Figure Object to use for plotting. If not provided it is either inferred from `ax` if given, or a new object is generated.
|ax        | matplotlib.axes.Axes, optional | Axes object to be used for plotting. If not provided it is either inferred from `fig`, or a new object is generated. 
|**kwargs  || Additional arguments passed to `pyplot.subplots()`


If the upper and lower bounds were generated with a custom resolution, the same resolution must be provided in the `model_resolution` parameter.

You may also pass keyword arguments to `matplotlib.pyplot.subplots()` via `**kwargs`. 
For comprehensive documentation please consult [`subplots()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html), [`figure()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib.pyplot.figure) and [`add_plot()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.add_subplot.html#matplotlib.figure.Figure.add_subplot).
Please note that it's assumed that `subplots()` returns a figure object and a single axes object.

Common parameters include:
| Parameters | | |
|----------|------------|-----------------|
| **Name** | **Type**   | **Description** |
| figsize | (float, float) |     Width, height in inches.
| facecolor | color | The background color.
| layout | str | The layout mechanism for positioning of plot elements to avoid overlapping Axes decorations (labels, ticks, etc).
| aspect | str or float | {'auto', 'equal'} or float, aspect ratio of axes.
| alpha | float | Set the alpha value used for blending - not supported on all backends.

The matplotlib objects used are returned:

| Returns | | |
|----------|--------------------------|-----------------|
| **Name** | **Type**                 | **Description** |
| fig      | matplotlib.figure.Figure | Figure object used for graph.
| ax       | matplotlib.axes.Axes     | Axes object used for graph.


## fitting_toolkit.confidence_interval

Given already fitted parameters and a covariance matrix, a confidence interval can be calculated using 
```python
confidence_interval(model, xdata: np.array, params: np.array, cov: np.array, resamples: int, nsigma: float = 1)
```

| Parameters | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| model    | function | Model fitted to data
| xdata    | np.array | The independent variable at which the confidence interval is to be calculated.
| params   | np.array | Fitted parameters passed onto `model`.
| cov      | np.array | Covariance matrix of `params`
| resamples| int, optional | Number of resamples to be calculated.
| nsigma | float, optional | Specifies the number of standard deviations corresponding to the desired confidence interval, when assuming a normal distribution.

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
Errors relative to the fitted function can be calculated as follows:
```python
sigma_pos = upper_conf - f(x, *params)
sigma_neg = f(x, *params) - lower_conf
```
So that the fitted value `x[i]` are
```python
print(f"f({x[i]:.2e}) = {f(x[i], *params):.2e} (+{sigma_pos[i]:.2e}/-{sigma_neg[i]:.2e})")
```