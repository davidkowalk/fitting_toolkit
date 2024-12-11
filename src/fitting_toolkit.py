from scipy.optimize import curve_fit as curve_fit_scipy
import numpy as np
from matplotlib import pyplot as plt

from . fit import curve_fit_mle, fit_distribution_mle, fit_distribution_anneal
from . utils import generate_thresholds, get_sigma_probability, generate_gaussian_mix

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

    def __str__(self):
        str_repr = f"""Fit(
    model = {self.model.__code__.co_name}({", ".join(self.model.__code__.co_varnames)})
    params = ({", ".join(self.params.astype(str))})
    cov = {'\n\t' + str(self.cov).replace('\n', '\n\t')}
    axis = array{str(np.shape(self.axis))}
    lower = array{str(np.shape(self.lower))}
    upper = array{str(np.shape(self.upper))}
)"""
        return str_repr
    
    def __repr__(self):
        return self.__str__()
    
    def reduced_chi_sqrd(self, x: np.ndarray, y:np.ndarray):
        """
        Calculates the reduced Chi-Squared statistic for fit.

        Args:
            x (np.ndarray): list of x-data
            y (np.ndarray): list of y-data

        Returns:
            reduced_chi_sqrd (float)
        """
        residuals = y - self.model(x, *self.params)
        nu = len(x) - self.model.__code__.co_argcount + 1

        return np.dot(np.transpose(residuals), residuals)/nu


# =========================
#  Package Functionalities
# =========================

def confidence_interval(model, xdata: np.array, params: np.array, cov: np.array, resamples: int, nsigma: float = 1) -> tuple[np.array, np.array]:
    """
    Computes the confidence intervals for the predictions of a model based on a set of input data.

    The function performs parametric bootstrapping by generating multiple resamples of the model parameters
    and computes the lower and upper thresholds for the confidence intervals at each axis point.

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

def curve_fit(model, xdata: np.array, ydata: np.array, yerror = None, method = "scipy",
              resamples = 5000, model_resolution: int = None, model_axis = None, nsigma:float = 1, **kwargs) -> tuple[np.array, np.array, np.array, np.array]:
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
        method (str, optional): Select method used for fitting the model. Must either be \"scipy\" for scipy's builtin least squares fit or \"mle\" for maximum likelyhood estimation.
        resamples (int, optional): The number of resampling iterations for bootstrapping confidence intervals. Default is 5000.
        model_resolution (int, optional): If specified the confidence interval and model will be calculated at linearly spaced points along x-axis. Otherwise xdata is used.
        model_axis (np.ndarray, optional): If specified this axis is used instead of axis generated via model_resolution
        nsigma (float): Number of standard deviation passed to confidence_interval()
        **kwargs: Additional arguments passed to SciPy's `curve_fit` function.

    Returns:
        fit (fitting_toolkit.Fit): Wrapper object containing the fitted model, fit results and confidence interval. 

    When using \"scipy\" method x-errors are not used, y-errors are optional.
    When using \"mle\" method y-errors are required, x-errors are optional. Note that using xerrors is considerably more computationally expensive.
    """

    if not(np.shape(xdata) == np.shape(ydata)):
        raise ValueError(f"x-data and y-data have different lengths and thus cannot be broadcast together.\nx: {np.shape(xdata)}, y: {np.shape(ydata)}")

    if method == "scipy":
        params, cov = curve_fit_scipy(f = model, xdata = xdata, ydata = ydata, sigma = yerror, **kwargs)
    elif method == "mle":
        if yerror is None:
            raise ValueError("When using maximum likelyhood estimation y-errors are required.")
        if 0 in yerror:
            raise ValueError("Error cannot be 0.")
        
        params, cov = curve_fit_mle(model = model, xdata=xdata, ydata=ydata, yerror=yerror, **kwargs)
    else:
        raise ValueError("Invalid method. Method must either be \"scipy\" for non linear leasts squares or \"mle\" for maximum likelyhood estimate.")

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


def fit_peaks(events, peak_estimates, peak_limits, sigma_init, theta_0 = None, anneal = False, model = None, **kwargs):
    """
    Fits a mixture of Gaussian peaks to a given set of events using maximum likelihood estimation (MLE) 
    or simulated annealing for initial parameter optimization.

    Args:
        events (array-like): The data points or events to which the Gaussian mixture model will be fitted. Typically a 1D array.
        peak_estimates (np.ndarray): Initial estimates for the locations (means) of the Gaussian peaks.
        peak_limits (float or np.ndarray): Limits for how far each Gaussian mean can deviate from the initial peak estimates.
        sigma_init (float or np.ndarray): 
            Initial estimates for the standard deviations (sigmas) of the Gaussian peaks. If a single value is provided, it will be applied to all peaks.
            If anneal is set to True this value will be used as upper limit, if anneal is not used it will be used as initial estimate.
        theta_0** (array-like, optional):  
          Initial parameters for the Gaussian mixture model, following the format:  
          [mu_1, sigma_1, weight_1, ..., weight_(n-1), mu_n, sigma_n]`.  
          If `None`, the parameters will be initialized based on peak_estimates and sigma_init.
          If anneal = True it will be ignored
        anneal (bool, optional): If True, uses simulated annealing to optimize the initial parameters. Default is False. This option is significantly more computationally expensive than local search.
            Recommended only for non convex probability space, i.e. multiple peaks.
        model (callable, optional): A custom model to use instead of a Gaussian mixture. If None, a Gaussian mixture model is generated. Provide taylored model for large number of peaks.
        **kwargs: Additional arguments passed to the fitting functions (`fit_distribution_anneal` or `fit_distribution_mle`).

    Returns:
        Fit (`Fit` object):  
          A Fit object containing the following:
          - model: The fitted model (Gaussian mixture or custom model).
          - params: Optimized parameters for the model.
          - cov: Covariance matrix for the parameter estimates.
          - Other attributes are set to `None` in the current implementation.
    """
    peak_number = len(peak_estimates)
    if model is None:
        gauss_mix = generate_gaussian_mix(peak_number)
    else:
        gauss_mix = model

    if anneal:

        min_sigma = np.min(np.abs(events - np.roll(events, 1)))

        #generate bounds
        mu_bounds = np.transpose((peak_estimates-peak_limits, peak_estimates+peak_limits))
        sigma_bounds = (min_sigma, sigma_init)
        a_bounds = [0, 1]

        bounds = list()

        for i in range(peak_number):
            bounds.extend([mu_bounds[i], sigma_bounds, a_bounds])
        
        bounds = bounds[:-1]

        #fit using annealment
        theta_0 = fit_distribution_anneal(gauss_mix, events, bounds)
    
    if theta_0 is None:
        #arange parameters appropriately
        #scheme: mu_1, s_1, a_1, mu_2, s_2, a_2..., a_n-1, mu_n, s_n
        theta_0 = list()
        if not hasattr(sigma_init, "__iter__"):
            sigma_init = [sigma_init]*peak_number

        for est in zip(peak_estimates, sigma_init):
            theta_0.extend(est)
            theta_0.append(1/peak_number)
    
        del(theta_0[-1])

    params, cov = fit_distribution_mle(gauss_mix, events, theta_0)
    return Fit(model, params, cov, None, None, None) #Return without confidence interval



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

