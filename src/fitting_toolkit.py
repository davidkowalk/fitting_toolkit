from scipy.optimize import curve_fit as sc_curve_fit
import numpy as np
from matplotlib import pyplot as plt

def generate_thresholds(data, lower_frac=1/6, upper_frac=5/6):
    """
    Generates two thresholds such that:
    - A fraction (lower_frac) of the data is below the lower threshold.
    - A fraction (1 - upper_frac) of the data is above the upper threshold.
    
    Args:
        data (numpy.ndarray): The dataset.
        lower_frac (float): Fraction of data below the lower threshold (default 1/6).
        upper_frac (float): Fraction of data above the lower threshold (default 5/6).
    
    Returns:
        tuple: (lower_threshold, upper_threshold)
    """
    lower_threshold = np.percentile(data, lower_frac * 100)  # 16.67th percentile
    upper_threshold = np.percentile(data, upper_frac * 100)  # 83.33rd percentile
    return lower_threshold, upper_threshold

def confidence_interval(model, xdata: np.array, params: np.array, cov: np.array, resamples: int) -> tuple[np.array, np.array]:
    """
    Computes the confidence intervals for the predictions of a model based on a set of input data.

    The function performs bootstrapping by generating multiple resamples of the model parameters
    and computes the lower and upper thresholds for the confidence intervals at each data point.

    Args:
        model (function): The model function that takes input data and model parameters.
        xdata (numpy.ndarray): The input data for which the confidence intervals are calculated.
        params (numpy.ndarray): The initial model parameters.
        cov (numpy.ndarray): The covariance matrix of the model parameters, used to generate resamples.
        resamples (int): The number of resampling iterations to generate for bootstrapping.

    Returns:
        tuple: A tuple containing:
            - lower_conf (numpy.ndarray): The lower bounds of the confidence intervals for each data point.
            - upper_conf (numpy.ndarray): The upper bounds of the confidence intervals for each data point.
    """
    random = np.random.multivariate_normal(params, cov, resamples)

    params_resamples = random.transpose()

    lower_conf = list()
    upper_conf = list()

    for x in xdata:
        distr = model(x, *params_resamples)
        interval = generate_thresholds(distr)
        lower_conf.append(interval[0])
        upper_conf.append(interval[1])
    
    return np.array(lower_conf), np.array(upper_conf)

def curve_fit(model, xdata: np.array, ydata: np.array, yerror = None, resamples = 5000, confidence_resolution: int = None, **kwargs) -> tuple[np.array, np.array, np.array, np.array]:
    """
    Fits a model to data and calculates confidence intervals for the fitted parameters and predictions.

    This function uses SciPy's `curve_fit` to estimate the parameters and their covariance matrix.
    It then computes confidence intervals for the model predictions at the given input data points
    using a resampling approach.

    Args:
        model (function): The model function to fit. It should take `xdata` and the model parameters as inputs.
        xdata (numpy.ndarray): The input data to fit the model to.
        ydata (numpy.ndarray): The observed data corresponding to `xdata`.
        yerror (numpy.ndarray, optional): The uncertainties in the observed data `ydata`. Default is None.
        resamples (int, optional): The number of resampling iterations for bootstrapping confidence intervals. Default is 5000.
        confidence_resolution (int, optional): If specified the confidence interval will be calculated at linearily spaced points along x-axis. Otherwise xdata is used.
        **kwargs: Additional arguments passed to SciPy's `curve_fit` function.

    Returns:
        tuple: A tuple containing:
            - params (numpy.ndarray): The optimized parameters for the model.
            - cov (numpy.ndarray): The covariance matrix of the optimized parameters.
            - lower_conf (numpy.ndarray): The lower bounds of the confidence intervals for each data point.
            - upper_conf (numpy.ndarray): The upper bounds of the confidence intervals for each data point.
    """
    params, cov = sc_curve_fit(f = model, xdata = xdata, ydata = ydata, sigma = yerror, **kwargs)
    if confidence_resolution is None:
        resampled_points = xdata
    elif confidence_resolution > 0:
        resampled_points = np.linspace(min(xdata), max(xdata), confidence_resolution) 
    else:
        raise ValueError("Unable to specify confidence points")
    
    lower_conf, upper_conf = confidence_interval(model, resampled_points, params, cov, resamples)

    return params, cov, lower_conf, upper_conf

def plot_fit(xdata, ydata, model, params, lower, upper, xerror = None, yerror = None, confidence_resolution: int = None, markersize = 4, capsize = 4, **kwargs) -> tuple[plt.figure, plt.axes]:
    """
    Plots the model fit to the data along with its confidence intervals.

    This function creates a plot of the data points with optional error bars, the model's fit, 
    and the confidence intervals for the predictions. The confidence intervals are represented 
    as dashed lines around the model fit.

    Args:
        xdata (numpy.ndarray): The x-values of the data points.
        ydata (numpy.ndarray): The y-values of the data points.
        model (function): The model function that takes `xdata` and model parameters as inputs.
        params (numpy.ndarray): The parameters for the model fit.
        lower (numpy.ndarray): The lower bounds of the confidence intervals for the model predictions.
        upper (numpy.ndarray): The upper bounds of the confidence intervals for the model predictions.
        xerror (numpy.ndarray, optional): The uncertainties in the x-values of the data points. Default is None.
        yerror (numpy.ndarray, optional): The uncertainties in the y-values of the data points. Default is None.
        markersize (int, optional): The size of the markers for the data points. Default is 4.
        capsize (int, optional): The size of the caps on the error bars. Default is 4.

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure): The figure object for the plot.
            - ax (matplotlib.axes.Axes): The axes object for the plot.

    Notes:
        - The model fit is shown as a solid line.
        - The confidence intervals are shown as dashed lines labeled as "1σ-Confidence."
        - The top and right spines of the plot are hidden for better visualization.
        - A grid is added to the plot for improved readability.
    """

    if confidence_resolution is None:
        resampled_points = xdata
    elif confidence_resolution > 0:
        resampled_points = np.linspace(min(xdata), max(xdata), confidence_resolution) 
    else:
        raise ValueError("Unable to specify confidence points")
    
    fig, ax = plt.subplots()

    #ax.scatter(xdata, ydata, color = "black", s = 2)
    ax.errorbar(xdata, ydata, yerr = yerror, xerr = xerror, fmt=".", linestyle = "", color = "black", capsize=capsize, markersize = markersize)
    ax.plot(xdata, model(xdata, *params), color = "black", linewidth = 1, linestyle = "-")
    ax.plot(resampled_points, upper, color = "black", linewidth = 0.75, linestyle = "--", label = "1$\sigma$-Confidence")
    ax.plot(resampled_points, lower, color = "black", linewidth = 0.75, linestyle = "--")

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid("both")
    ax.set_axisbelow(True)

    return fig, ax

