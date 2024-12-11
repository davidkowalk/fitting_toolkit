"""
This submodule contains the algorithms and wrappers used for fitting a curve
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, dual_annealing

# maximum likelyhood
# inverse Hessian Matrix is Covariance in log likelyhood
# https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470824566.app1
 
def neg_log_likelihood_per_point_yerr(model, theta: np.ndarray, x: np.ndarray, y: np.ndarray, yerror:np.ndarray) -> np.ndarray:
    return 0.5 * (np.log(2 * np.pi * yerror**2) + ((y - model(x, *theta)) / yerror)**2)

def neg_log_likelihood_per_point_xyerr(model, theta: np.ndarray, x: np.ndarray, y: np.ndarray, xerror:np.ndarray, yerror:np.ndarray) -> np.ndarray:

    def single_neg_log_likelihood_per_point_xyerr(xi, yi, sig_xi, sig_yi):

        def integrand(u):
            term_y = -((yi - model(u, *theta)) / sig_yi)**2
            term_x = -((xi - u) / sig_xi)**2
            return np.exp(0.5 *(term_y + term_x))

        norm = (2 * np.pi * sig_xi * sig_yi)

        integral, _ = quad(integrand, -np.inf, np.inf)
        return -np.log(integral / norm)

    vectorized_likelihood = np.vectorize(single_neg_log_likelihood_per_point_xyerr)
    return vectorized_likelihood(x, y, xerror, yerror)

def neg_log_likelyhood(theta, model, x, y, yerror, xerror = None):
    """
    Computes the negative of the natural logarithm of the probability density,
    that a model with parameters theta produces the data (x, y).

    Standard deviation in y is required, standard deviation in x is optional.
    If the error in x is negligable it should be omitted for performance reasons.

    Args:
        theta (np.ndarray): parameters at which the probability density is to be calculated
        model (function): function to be fitted to data
        x (np.ndarray): x-position of data
        y (np.ndarray): y-position of data
        yerror (np.ndarray / float): standard deviation of y-values
        xerror (np.ndarray / float, optional): standard deviation of x-values

    Returns:
        p (float): Propability density that the model with parameters theta produces the values (x, y)
    """

    if xerror is None:
        return np.sum(neg_log_likelihood_per_point_yerr(model, theta, x, y, yerror))

    return np.sum(neg_log_likelihood_per_point_xyerr(model, theta, x, y, xerror, yerror))

def curve_fit_mle(model, xdata: np.array, ydata: np.array, yerror, theta_0 = None, xerror = None, **kwargs):
    """
    Fits model curve to (xdata, ydata) using maximum likelyhood estimate.

    Standard deviation in y is required, standard deviation in x is optional.
    If the error in x is negligable it should be omitted for performance reasons.

    Args:
        model (function): function to be fitted to data
        x (np.ndarray): x-position of data
        y (np.ndarray): y-position of data
        yerror (np.ndarray / float): standard deviation of y-values
        xerror (np.ndarray / float, optional): standard deviation of x-values
        theta_0 (np.ndarray, optional): Initial guess of parameters. If not provided all paramters are initially set to zero
        **kwargs (optional): additional key word arguments passed onto scipy.optimize.minimize

    Returns:
        params (np.ndarray): Optimal Parameters
        covariance (np.ndarray): Covariance matrix of fitted parameters
    """

    if theta_0 == None:
        theta_0 = np.zeros(model.__code__.co_argcount -1)

    result = minimize(neg_log_likelyhood, theta_0, args=(model, xdata, ydata, yerror, xerror), **kwargs)
    params = result.x
    cov = result.hess_inv

    return params, cov

def neg_log_event_likelyhood(model, event, theta):
    x = -np.log(model(event, *theta))
    return x

def fit_distribution_mle(model, events:np.array, theta_0:np.ndarray = None, data_range = None, **kwargs):
    """
    Finds optimal parameters for probability distribution via maximum likelyhood estimation.
    Let events x be measurements of a random variable which are independent and identically distributed with probability density p(x, *theta).
    This function finds the parameters with the highest cumulative negative logarithm of the probability density.

    Args:
        model (function): Distribution to be fitted
        events (np.ndarray): Elements observed
        theta_0 (np.ndarray): Initial guess of parameters
        data_range (tuple): Interval in which events are fit.
        **kwargs: Additional arguments passed to scipy.optimize.minimizer

    Returns
        params (np.ndarray): Best fit parameters
        cov (np.ndarray or scipy.sparse.linalg.LinearOperator): Covariance matrix of parameters

    This method may quickly become unreliable for combined resolutions. Peaks should be fit separately.
    """
    
    def total_log_likelyhood(theta, model, events):
        return np.sum(neg_log_event_likelyhood(model, events, theta))

    if data_range is not None:
        events = np.copy(events)
        events = events[np.logical_and(events > data_range[0], events < data_range[1])]
    
        
    if theta_0 is None:
        theta_0 = np.zeros(model.__code__.co_argcount -1)
    
    result = minimize(total_log_likelyhood, theta_0, args=(model, events), **kwargs)
    params = result.x
    cov = result.hess_inv

    return params, cov

def fit_distribution_anneal(model, events, bounds, data_range = None, **kwargs):
    """
    Finds optimal parameters for probability distribution via maximum likelyhood estimation using simulated annealing.
    Let events x be measurements of a random variable which are independent and identically distributed with probability density p(x, *theta).
    This function finds the parameters with the highest cumulative negative logarithm of the probability density.

    Args:
        model (function): Distribution to be fitted
        events (np.ndarray): Elements observed
        bounds: Bounds for variables. There are two ways to specify the bounds:
            1. Instance of scipy.Bounds class.
            2. Sequence of (min, max) pairs for each element in x.
        data_range (tuple): Interval in which events are fit.
        **kwargs: Additional arguments passed to scipy.optimize.minimizer

    Returns
        params (np.ndarray): Best fit parameters
        cov (np.ndarray or scipy.sparse.linalg.LinearOperator): Covariance matrix of parameters

    This method may quickly become unreliable for combined resolutions. Peaks should be fit separately.
    """
    def total_log_likelyhood(theta, model, events):
        return np.sum(neg_log_event_likelyhood(model, events, theta))

    if data_range is not None:
        events = np.copy(events)
        events = events[np.logical_and(events > data_range[0], events < data_range[1])]
    
    result = dual_annealing(total_log_likelyhood, bounds, args=(model, events), **kwargs)
    params = result.x

    return params
