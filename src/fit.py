"""
This submodule contains the algorithms and wrappers used for fitting a curve
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

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

def curve_fit_mle(model, xdata: np.array, ydata: np.array, yerror, theta_0 = None, xerror = None):
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

    Returns:
        params (np.ndarray): Optimal Parameters
        covariance (np.ndarray): Covariance matrix of fitted parameters
    """

    if theta_0 == None:
        theta_0 = np.zeros(model.__code__.co_argcount -1)

    result = minimize(neg_log_likelyhood, theta_0, args=(model, xdata, ydata, yerror, xerror))
    params = result.x
    cov = result.hess_inv

    return params, cov