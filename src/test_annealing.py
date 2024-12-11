#=======================================================================================================================
# This file serves only to test the fitting of peaks. It is mostly AI Generated and should be deleted before release!!!
#=======================================================================================================================

import numpy as np
from scipy.optimize import Bounds

from fit import fit_distribution_anneal
from fitting_toolkit import fit_peaks

from matplotlib import pyplot as plt

def normal(x, mu, sigma):
    """
    Gaussian (normal) probability density function.
    
    Args:
        x (np.ndarray): Data points.
        mu (float): Mean of the distribution.
        sigma (float): Standard deviation of the distribution.
    
    Returns:
        np.ndarray: Probability density values.
    """
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


# Generate IID dataset from two normal distributions
def generate_data(n_samples, mu1, sigma1, mu2, sigma2, weight1=0.5):
    """
    Generate a dataset sampled from a mixture of two normal distributions.
    
    Args:
        n_samples (int): Total number of samples.
        mu1 (float): Mean of the first normal distribution.
        sigma1 (float): Standard deviation of the first normal distribution.
        mu2 (float): Mean of the second normal distribution.
        sigma2 (float): Standard deviation of the second normal distribution.
        weight1 (float): Proportion of samples from the first distribution (default is 0.5).
    
    Returns:
        np.ndarray: Generated dataset.
    """
    n1 = int(n_samples * weight1)
    n2 = n_samples - n1
    
    data1 = np.random.normal(mu1, sigma1, n1)
    data2 = np.random.normal(mu2, sigma2, n2)
    
    return np.concatenate([data1, data2])


def model(x, a, mu1, s1, mu2, s2):
    return a*normal(x, mu1, s1) + (1-a)*normal(x, mu2, s2)

def test_gauss_mix():
    from utils import generate_gaussian_mix
    from scipy.integrate import quad

    gauss = generate_gaussian_mix(2)
    params = (1, 0.5, 0.9, 3, 0.5)
    

    A = quad(gauss, -np.inf, np.inf, args = params)
    print(A[0])

    #x = np.linspace(0, 5)
    #y = gauss(x, *params)

def test_anneal():
    np.random.seed(42)  # For reproducibility
    data = generate_data(n_samples=1000, mu1=0, sigma1=1, mu2=3, sigma2=.5, weight1=0.7)
    
    # Fit the model
    bounds = ([0,1] , [-0.5, 0.5], [0, 1.5], [2, 3], [0, 2])  # Bounds for [mu, sigma]
    params = fit_distribution_anneal(model, data, bounds)

    h, x,_ = plt.hist(data, 50, density=True, align="left")
    plt.step(x, model(x, *params))
    plt.show()

def test_peak_fit():
    np.random.seed(42)  # For reproducibility
    data = generate_data(n_samples=1000, mu1=0.2, sigma1=1, mu2=3.1, sigma2=.5, weight1=0.7)

    params, cov, model = fit_peaks(data, np.array([0, 3]), peak_limits=0.5, max_sigma=2)

    print(params)


if __name__ == "__main__":
    test_peak_fit()