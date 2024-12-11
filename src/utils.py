from scipy.special import erf
import numpy as np

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

def normal(x, mu, sigma, w):
    return (w / (np.sqrt(2 * np.pi) * sigma))*np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def generate_gaussian_mix(n):

    """
    Dynamically generates a function for the superposition of `n` Gaussian functions.
    
    Args:
        n (int): Number of Gaussian functions to include in the superposition.
    
    Returns:
        function: A callable function `f(x, params)` where `params` is a flat array of weights, means, 
                  and standard deviations for each Gaussian component, of size 3*n.
    """

    def gaussian_mix(x, *params):
        
        if len(params) != 3 * n - 1:
            raise ValueError(f"Expected {3 * n - 1} parameters, but got {len(params)}.\n Parameters: {params}")
        
        if len(np.shape(x)) == 0:
            x = [x]

        params = np.asarray(params)
        mu = params[0::3]  # Means
        sigma = params[1::3]  # Standard deviations
        a = params[2::3]  # Weights

        return  np.sum(normal(np.transpose([x]), mu[:-1], sigma[:-1], a), axis = 1) + normal(np.transpose([x]), mu[-1], sigma[-1], 1-np.sum(a))[:,0]

    return gaussian_mix