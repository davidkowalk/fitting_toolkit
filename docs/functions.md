
# Code Documentation

By separating the fitting functionality from the display options, a user can utilize the parts independently of each other.

## Using the Fitting Functionality

To fit a dataset, call:
```python
curve_fit(model, xdata: np.array, ydata: np.array, yerror = None, resamples = 5000, model_resolution: int = None, model_axis = None, nsigma:float = 1, **kwargs)
```

| Parameters | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| model    | function | Function to be fitted. Must take `xdata` as a first argument and then an arbitrary number of fit parameters.|
| xdata    | np.array | The independent variable where the data is measured. Each element should be float convertible if it is an array-like object.
| ydata    | np.array | The dependent data, a length M array - nominally f(xdata, ...)
| yerror   | np.array | (optional) Determines the uncertainty in ydata. Pass absolute values.
| resamples| int      | (optional) Number of samples to be generated in parameter space for bootstrapping.
|model_resolution | int, optional | If specified the confidence interval will be calculated at linearly spaced points along x-axis. Otherwise xdata is used.
| model_axis | numpy.ndarray, optional | If specified this axis is used instead of axis generated via model_resolution.
| **kwargs | any      | (optional) Parameters to be passed on to `scipy.optimize.curve_fit`

| Returns | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| params   | np.array | List of optimal parameters. Can be separated by `p1, p1, ..., pn = params`
| cov      | np.array | Covariance Matrix of parameters as provided by `scipy.optimize.curve_fit`. Standard deviations can be calculated by `sigma = np.sqrt(np.diagonal(cov))`
| lower_conf | np.array | Absolute y-values of lower edge of 1-sigma confidence interval.|
| upper_conf | np.array | Absolute y-values of upper edge of 1-sigma confidence interval.

### Using Custom Graphics

To generate your own plot you can use the returned values of `curve_fit`. Using a defined model `f(x, *params)`:

```python
params, cov, lower_conf, upper_conf = curve_fit(f, x, y)
#calculate standard deviations for possible later use
standard_deviations = np.sqrt(np.diagonal(cov))

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

## Displaying the Fitted Function

The fitting toolkit ships with built-in functions for displaying data with their fitted functions and their respective confidence intervals.
```python
plot_fit(xdata, ydata, model, params, lower, upper, xerror = None, yerror = None, model_resolution: int = None, markersize = 4, capsize = 4, fit_color = "black", fit_label = "Least Squares Fit", confidence_label = "1$\\sigma$-Confidence", fig = None, ax = None, **kwargs)
```

| Parameters | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
|xdata     | numpy.ndarray | The x-values of the data points.
|ydata     | numpy.ndarray | The y-values of the data points.
|model     | function | The model function that takes `xdata` and model parameters as inputs.
|params    | numpy.ndarray | The parameters for the model fit.
|lower     | numpy.ndarray | The lower bounds of the confidence intervals for the model predictions.
|upper     | numpy.ndarray | The upper bounds of the confidence intervals for the model predictions.
| **Optional Arguments** |
|xerror    | numpy.ndarray, optional | The uncertainties in the x-values of the data points. Default is None.
|yerror    | numpy.ndarray, optional | The uncertainties in the y-values of the data points. Default is None.
|model_resolution | int, optional | If specified the confidence interval will be calculated at linearly spaced points along x-axis. Otherwise xdata is used.
| model_axis | numpy.ndarray, optional | If specified this axis is used instead of axis generated via model_resolution.
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


## Calculate Confidence Interval for an Existing Fit

Given already fitted parameters and a covariance matrix, a confidence interval can be calculated using `confidence_interval(model, xdata: np.array, params: np.array, cov: np.array, resamples: int, nsigma: float = 1)`

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
Errors relative to the fitted function can be calculated as follows:
```python
sigma_pos = upper_conf - f(x, *params)
sigma_neg = f(x, *params) - lower_conf
```
So that the fitted value `x[i]` are
```python
print(f"f({x[i]:.2e}) = {f(x[i], *params):.2e} (+{sigma_pos[i]:.2e}/-{sigma_neg[i]:.2e})")
```

## Generate Probability for Sigma Interval

To get probability to fall into n-sigma interval call `get_sigma_probability(n: float = 1)`

| Parameters | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| n        | float    | Number of sigmas in interval

| Returns | | |
|----------|----------|-----------------|
| **Name** | **Type** | **Description** |
| p        | float    | Probability of falling into sigma interval. $P(\mu - n*\sigma < X < \mu + n*\sigma) $


## Generating Thresholds

Given a bootstrapped distribution, generate a custom confidence interval.

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