# Fitting a model

The fitting toolkit provides two modes of fitting models to data.
1. Curve Fitting: Fits a curve to datapoints in an xy-Diagram
2. Distribution Fitting (Peak Fitting): Fits a probability distribution to a set of events

This document discusses both of these modes in detail and provides a set of instructions to use the inner workings of the toolkit independently.

## Curve Fitting

The fitting toolkit provides an inbuilt function to easily and quickly fit a model to data and calculate the appropriate confidence intervals.
```py
fitting_toolkit.curve_fit(model, xdata: np.array, ydata: np.array, yerror = None, method = "scipy", resamples = 5000, model_resolution: int = None, model_axis = None, nsigma:float = 1, **kwargs)
```

The curve fit function takes a lot of different inputs regarding both the curve fit itself and the estimation of the confidence interval. This description focuses on the former.
```py
fitting_toolkit.curve_fit(model, xdata: np.array, ydata: np.array, yerror = None, method = "scipy", **kwargs)
```

For a detailed description please reference the technical docs. The basic objective of curve fitting is to minimize the discrepancy between observed data and a model (or curve). First we define the curve depending on a set of parameters $\theta$. This has to be a callable, typically a python native function:
```py
def model(x, **theta)
```

The builtin fitting functions can handle an arbitrary number of arguments:
```py
def model(x, a, b, c, ...)
```
The parameter `x` has to be able to take a numpy array. Otherwise model must be veoctorized using `np.vectorize` which is exeedingly slow.

Choosing the criteria to judge what constitutes a "good fit" is, to a degree, subjective. The fitting toolkit provides two modes of fitting: least squares fitting (LSF) and maximum likelyhood estimation (MLE). 

| LSF | MLE 
|:---:|:---:
|Minimizes the sum of squared differences (residuals) between observed data and model predictions. | Maximizes the likelihood function, which measures how likely the observed data are given the model parameters.
| Geometrically interpretable in terms of minimizing the distance between points (squared deviations). | distance between points (squared deviations).	Probabilistic interpretation is broader and fits into statistical modeling frameworks.
| Unstable for non polynomial models | More stable for a wide variety of curves.
| Sensitive to outliers because squared errors emphasize large deviations. | Robust to outliers if the likelihood function is chosen appropriately
| Less robust when errors are non-Gaussian. | Flexible to handle varying error distributions.
| Often simpler and computationally faster, especially in linear models. | Can be computationally intensive, especially for complex models or large datasets

When using Gaussian errors in the y-axis LSF and MLE optimization function can be shown to have global extrema at the same point in parameter space. With the fitting toolkit you can select between LSF and MLE with the `method` argument. When `method = "scipy"` the least squares algorithm provided by SciPy is used. When `method = "mle"` the native MLE implementation is called instead.

### Least Square (SciPy)

Call the scipy implementation with

```py
fit = fitting_toolkit.curve_fit(model, xdata, ydata, yerror)
```

The method does not need to be specified because `scipy` is the default.  `yerror` is optional. You may specify additional key word arguments which are automatically passed to `scipy.curve_fit()`. Please reference the [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) for this. A `fitting_toolkit.Fit` object is returned.

### Maximum Likelyhood

Fit using MLE with

```py
fit = fitting_toolkit.curve_fit(model, xdata, ydata, yerror = dy, xerror = dx, method = "mle")
```

Note that MLE can handle both x- and y-errors. ([Reference](../resources/max_likelyhood_est.md#1-computing-pd-of-data-point)) When fitting with MLE the yerrors are required, xerrors are optional. Since xerrors are covered by `**kwargs` they must be specified using the `xerror` key word. In the current implementation Gaussian errors are assumend.

You may also specify an initial guess of the parameters $\theta _0$ with `theta_0 = [a0, b0, ...]`. Additional key word arguments are passed to `scipy.minimize`. ([Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize)) Among other things this allows you to specify a minimizer algorithm, bounds, constraints, tolerance, etc.

A `fitting_toolkit.Fit` object is returned. 

## Distribution Fitting / Peak Fitting

Many statistical processes require us to fit a probability distribution to a set of measurements of a random variable. (For example the angle in a scattering experiment.) This may be achieved by fitting a curve to a normalized histogram, however this solution is sensitive to bin width and sample size. A better solution is to fit the distribution directly to the data by finding the set of parameters that directly maximizes the likelyhood (defined as the additive inverse of the natural logarithm of the probability density function) of the observed datapoints.

> What is the probability distribution in the set of distributions defined by my parameter space with the highest likelyhood of producing the observed results?

Let `events` be a set of measurements of a random variable.

```py
fit = fitting_toolkit.fit_peaks(events, peak_estimates = None, peak_limits = None, sigma_init=None, theta_0 = None, anneal = False, model = None, local_options = {}, anneal_options = {})
```

The functionality of peak fitting is fundamentally determined by the `anneal` parameter, which decides whether [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) should be used to find a set of initial parameters close to the global maximum in parameter space.

### Fit with Local Minimization

Most common optimization problems are convex, which means that they have 1 optimal state better than all other optimal states and a state closer to the optimum is always better than a state further from the optimum.

One of these is the single gaussian curve. In these cases the local minimizer is sufficient for finding the optimal set of parameters. Run:

```py
fit = fitting_toolkit.fit_peaks(events, peak_estimates, sigma_init, local_options = {})
```
Optionally you can supply a full set of initial parameters `theta_0`. In this case `sigma_init` may be omitted. Bounds can be passed via `local_options`.

### Fit with Simulated Annealing


[Simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) is an iterative optimization method for finding global optima in a constrained subspace. It is a computationally more expensive than local minimization and should be used for problems with non-convex optimization functions in parameter space. Usual examples are Gaussian fits with multiple peaks.

To use the simulated annealing call:
```py
fit = fitting_toolkit.fit_peaks(events, peak_estimates, peak_limits, sigma_init, anneal = True, annealing_options = {})
```

## Submodule fitting_toolkit.fit

The submodule `fitting_toolkit.fit` contains all functions for fitting not covered by `scipy`.
This section covers notable use cases. For a more detailed description reference the [technical documentation](../../technical%20docs/submodules/fit.md).

### fitting_toolkit.fit.curve_fit_mle

To call the mle implementation directly without generating a confidence interval call
```py
params, cov = fitting_toolkit.fit.curve_fit_mle(model, xdata: np.array, ydata: np.array, yerror, theta_0 = None, xerror = None, **kwargs)
```

Fits model curve to (xdata, ydata) using maximum likelyhood estimate.
Standard deviation in y is required, standard deviation in x is optional. If the error in x is negligable it should be omitted for performance reasons. Returns `params, cov`

**Example:**
```py
from fitting_toolkit.fit import curve_fit_mle as curve_fit
x_data = np.array([...])
y_data = np.array([...])

#Define a 1% Error
y_error = 0.01*y_data
x_error = 0.01*x_data #(Optional)

def model(x, a, b):
    return a * x + b

params, cov = curve_fit(model, x_data, y_data, xerror = x_error, yerror = y_error)
```

### fitting_toolkit.fit.fit_distribution_mle

Fits a probability distribution to a set of events using maximum likelyhood estimation with `scipy.optimize.minimize`.

```py
params, cov = fitting_toolkit.fit.fit_distribution_mle(model, events:np.array, theta_0:np.ndarray = None, data_range = None, **kwargs)
```

**Examle:**
```py
from fitting_toolkit.fit import fit_distribution_mle as fit_peak
import numpy as np
events = np.array([...]) # Measurements of random variable

#define normal distribution
def model(x, mu, sigma):
    return np.exp(-((x - mu)/sigma)**2)/np.sqrt(2**np.pi*sigma**2)

params, cov = fit_peak(model, events)
```

To only fit events in a certain range use
```py
data_range = (min, max)
```

### fitting_toolkit.fit.fit_distribution_anneal

Fits a probability distribution to a set of events using maximum likelyhood estimation with `scipy.optimize.dual_annealing`. Does not compute the covariance matrix!

```py
params = fit_distribution_anneal(model, events, bounds, data_range = None, **kwargs)
```

**Example**:
```py
from fitting_toolkit.fit import fit_distribution_anneal as fit_peak
import numpy as np
events = np.array([...]) # Measurements of random variable
bounds = ((min_mu, max_u), (min_sigma, max_sigma))

#define normal distribution
def model(x, mu, sigma):
    return np.exp(-((x - mu)/sigma)**2)/np.sqrt(2**np.pi*sigma**2)

params = fit_peak(model, events, bounds)
```

To only fit events in a certain range use
```py
data_range = (min, max)
```