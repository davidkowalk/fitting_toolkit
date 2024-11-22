
# Code Documentation

By separating the fitting functionality from the display options a user can utilize the parts independently of each other.

## Using the Fitting Functionality

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

### Using Custom Graphics

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

## Calculate Confidence Interval for an Existing Fit

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

## Generating Thresholds

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