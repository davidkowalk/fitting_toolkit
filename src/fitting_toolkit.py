from scipy.optimize import curve_fit as sc_curve_fit
from scipy.special import erf
import numpy as np
from matplotlib import pyplot as plt

class Fit():
    """
    Class for wrapping all relevant information for a fitted function
    Fit(model, params, cov, x: np.ndarray, y: np.ndarray, upper: np.ndarray, lower: np.ndarray, dx: np.ndarray = None, dy: np.ndarray = None, resampled_points: np.ndarray = None)
    """

    def __init__(self, model, params: np.ndarray, cov: np.ndarray, x: np.ndarray, upper: np.ndarray, lower: np.ndarray):
        """
        model (function): The model function that takes `xdata` and model parameters as inputs.
        params (numpy.ndarray): The parameters for the model fit.
        lower (numpy.ndarray): The lower bounds of the confidence intervals for the model predictions.
        upper (numpy.ndarray): The upper bounds of the confidence intervals for the model predictions.
        """
        self.model = model
        self.axis = x

        self.upper = upper
        self.lower = lower 

        self.params = params
        self.cov = cov


#===================
# General Utilities
#===================

def array(*x):
    """
    Takes list as arguments and creates numpy array.

    Args:
        *x: Elements of array

    Returns:
        numpy.array(x)

    Example:
        array(2, 1, 4, 3) is eqivalent to numpy.array([2, 1, 4, 3])
    """
    return np.array(x)

def generate_thresholds(data, lower_frac=0.15865, upper_frac=0.84135):
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
    lower_threshold = np.percentile(data, lower_frac * 100)
    upper_threshold = np.percentile(data, upper_frac * 100)
    return lower_threshold, upper_threshold

def get_sigma_probability(n: float = 1):
    """
    Returns probability for event to fall into n-sigma interval assuming a gaussian distribution:
    P(mu - n*sigma < X < mu + n*sigma)  

    Args:
        n (float): Number of sigmas in interval

    Returns:
        p (float): Probability of falling into sigma interval.
    """

    return 1/2 * (erf(n / 1.4142135623730951) - erf(-n / 1.4142135623730951))

# =========================
#  Package Functionalities
# =========================

def confidence_interval(model, xdata: np.array, params: np.array, cov: np.array, resamples: int, nsigma: float = 1) -> tuple[np.array, np.array]:
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
        nsigma (float): Number of standard deviation in interval.

    Returns:
        tuple: A tuple containing:
            - lower_conf (numpy.ndarray): The lower bounds of the confidence intervals for each data point.
            - upper_conf (numpy.ndarray): The upper bounds of the confidence intervals for each data point.
    """
    random = np.random.multivariate_normal(params, cov, resamples)

    params_resamples = random.transpose()

    P = get_sigma_probability(nsigma)
    upper_threshold = 0.5 + P/2
    lower_threshold = 0.5 - P/2

    lower_conf = list()
    upper_conf = list()

    for x in xdata:
        distr = model(x, *params_resamples)
        interval = generate_thresholds(distr, lower_frac = lower_threshold, upper_frac = upper_threshold)
        lower_conf.append(interval[0])
        upper_conf.append(interval[1])
    
    return np.array(lower_conf), np.array(upper_conf)

def curve_fit(model, xdata: np.array, ydata: np.array, yerror = None, resamples = 5000, model_resolution: int = None, model_axis = None, nsigma:float = 1, **kwargs) -> tuple[np.array, np.array, np.array, np.array]:
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
        model_resolution (int, optional): If specified the confidence interval and model will be calculated at linearly spaced points along x-axis. Otherwise xdata is used.
        model_axis (np.ndarray, optional): If specified this axis is used instead of axis generated via model_resolution
        nsigma (float): Number of standard deviation passed to confidence_interval()
        **kwargs: Additional arguments passed to SciPy's `curve_fit` function.

    Returns:
        fit (fitting_toolkit.Fit): Wrapper object containing the fitted model, fit results and confidence interval. 
    """

    if not(np.shape(xdata) == np.shape(ydata)):
        raise ValueError(f"x-data and y-data have different lengths and thus cannot be broadcast together.\nx: {np.shape(xdata)}, y: {np.shape(ydata)}")

    params, cov = sc_curve_fit(f = model, xdata = xdata, ydata = ydata, sigma = yerror, **kwargs)
    if not model_axis is None:
         resampled_points = model_axis
    elif model_resolution is None:
        resampled_points = xdata
    elif model_resolution > 0:
        resampled_points = np.linspace(min(xdata), max(xdata), model_resolution) 
    else:
        raise ValueError("Unable to specify confidence points")
    
    lower_conf, upper_conf = confidence_interval(model, resampled_points, params, cov, resamples, nsigma)

    return Fit(model, params, cov, resampled_points, upper_conf, lower_conf)

def plot_fit(xdata, ydata, fit, xerror = None, yerror = None, markersize = 4, capsize = 4, fit_color = "black", fit_label = "Least Squares Fit", confidence_label = "1$\\sigma$-Confidence", fig = None, ax = None, **kwargs) -> tuple[plt.figure, plt.axes]:
    """
    Plots the model fit to the data along with its confidence intervals.

    This function creates a plot of the data points with optional error bars, the model's fit, 
    and the confidence intervals for the predictions. The confidence intervals are represented 
    as dashed lines around the model fit.

    Args:
        xdata (numpy.ndarray): The x-values of the data points.
        ydata (numpy.ndarray): The y-values of the data points.
        fit (fitting_toolkit.Fit): Wrapper object containing the fitted model, fit results and confidence interval. 
        xerror (numpy.ndarray, optional): The uncertainties in the x-values of the data points. Default is None.
        yerror (numpy.ndarray, optional): The uncertainties in the y-values of the data points. Default is None.
        fit_color (color, optional): color of the fitted function.
        markersize (int, optional): The size of the markers for the data points. Default is 4.
        capsize (int, optional): The size of the caps on the error bars. Default is 4.
        fit_label (str, optional): Label applied to the least square fit.
        confidence_label(str, optional): Label applied to upper confidence threshold.
        fig (matplotlib.pyplot.Figure, optional): Figure Object to use for plotting. If not provided it is either inferred from ax if given or a new object is generated.
        ax (matplotlib.axes.Axes, optional): Axes object to be used for plotting. If not provided it is either inferred from fig, or a new object is generated. 
        **kwargs: Additional arguments passed to `pyplot.subplots()`

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

    if not(np.shape(xdata) == np.shape(ydata)):
        raise ValueError(f"x-data and y-data have different lengths and thus cannot be broadcast together.\nx: {np.shape(xdata)}, y: {np.shape(ydata)}")


    if not(np.shape(fit.axis) == np.shape(fit.lower)):
        raise ValueError(f"x-axis does not match length of lower confidence interval\nx: {np.shape(fit.axis)}, y: {np.shape(fit.lower)}")
    if not(np.shape(fit.axis) == np.shape(fit.upper)):
        raise ValueError(f"x-axis does not match length of upper confidence interval\nx: {np.shape(fit.axis)}, y: {np.shape(fit.upper)}")
    
    if fig is None and ax is None:
        fig, ax = plt.subplots(**kwargs)
        
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid("both")
        ax.set_axisbelow(True)

    elif ax is None:
        ax = fig.axes[0] #Choose first axes object in Figure

    elif fig is None:
        fig = ax.get_figure()

    ax.errorbar(xdata, ydata, yerr = yerror, xerr = xerror, fmt=".", linestyle = "", color = fit_color, capsize=capsize, markersize = markersize)
    ax.plot(fit.axis, fit.model(fit.axis, *fit.params), color = fit_color, linewidth = 1, linestyle = "-", label = fit_label)
    ax.plot(fit.axis, fit.upper, color = fit_color, linewidth = 0.75, linestyle = "--", label = confidence_label)
    ax.plot(fit.axis, fit.lower, color = fit_color, linewidth = 0.75, linestyle = "--")

    return fig, ax

