"""
Generate and aply constraint functions.
"""

import numpy as np

def get_constraint(mean, deviation, type="gauss"):
    """
    Returns a function which constrains the input variables
    Types: gauss, uniform_prior or flat, exponential or exp
    For unconstrained fits you may pass np.inf
    """

    mean = np.asarray(mean)
    sigma = np.asarray(deviation)

    if type == "gauss":

        def constraint(*x):
            x = np.asarray(x)
            chi2 = np.sum(((x - mean) / sigma) ** 2)
            return np.exp(-0.5 * chi2)

        return constraint

    elif type in ("uniform_prior", "flat"):

        constrained = ~np.isinf(sigma)

        def constraint(*x):
            x = np.asarray(x)

            if np.all((np.abs(x-mean)[constrained] <= sigma[constrained])):
                norm = np.ones_like(x)
                norm[constrained] = 1.0 / sigma[constrained]
                return np.prod(norm)
            else:
                return 0.0
        return constraint
    
    elif type in ("exponential", "exp"):
        
        def constaint(*x):
            return np.prod(np.exp(-np.abs(x-mean)/sigma))

        return constaint

    else:
        raise ValueError("type must be in (gauss, uniform_prior, exp)")

def get_log_constraint(mean, deviation, type="gauss"):
    mean = np.asarray(mean)
    sigma = np.asarray(deviation)

    if type == "gauss":

        def constraint(*x):
            x = np.asarray(x)
            return -0.5 * np.sum(((x - mean) / sigma) ** 2)

        return constraint

    elif type in ("uniform_prior", "flat"):

        constrained = ~np.isinf(sigma)

        def constraint(*x):
            x = np.asarray(x)

            if np.all((np.abs(x-mean)[constrained] <= sigma[constrained])):
                norm = np.ones_like(x)
                norm[constrained] = 1.0 / sigma[constrained]
                return np.sum(np.log(norm))
            else:
                return 0.0
        return constraint
    
    elif type in ("exponential", "exp"):
        
        def constaint(*x):
            return np.sum(np.log(-np.abs(x-mean)/sigma))

        return constaint

    else:
        raise ValueError("type must be in (gauss, uniform_prior, exp)")