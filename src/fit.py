"""
This submodule contains the algorithms and wrappers used for fitting a curve
"""

import numpy as np
from scipy.integrate import quad

# maximum likelyhood
# inverse Hessian Matrix is Covariance in log likelyhood
# https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470824566.app1
 
def neg_log_likelihood_per_point_yerr(model, theta: np.ndarray, x: np.ndarray, y: np.ndarray, sigma_y:np.ndarray) -> np.ndarray:
    return 0.5 * (np.log(2 * np.pi * sigma_y**2) + ((y - model(x, theta)) / sigma_y)**2)

def neg_log_likelihood_per_point_xyerr(model, theta: np.ndarray, x: np.ndarray, y: np.ndarray, sigma_x:np.ndarray, sigma_y:np.ndarray) -> np.ndarray:
    def integrand(u):
        term_y = -((y - model(u, theta)) / sigma_y)**2
        term_x = -((x - u) / sigma_x)**2
        return np.exp(0.5 *(term_y + term_x))
    
    norm = (2 * np.pi * sigma_x * sigma_y)
    
    integral, _ = quad(integrand, -np.inf, np.inf)
    return -np.log(integral / norm)


