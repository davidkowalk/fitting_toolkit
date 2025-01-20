import numpy as np

from fitting_toolkit.fit import fit_distribution_anneal
from fitting_toolkit import fit_peaks, plot_fit

from fitting_toolkit.utils import args_to_dict

from matplotlib import pyplot as plt

def normal(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def get_data(n_samples, mu1, sigma1, mu2, sigma2, weight1=0.5):
    """
    Generates random Data from two normal distributions
    """
    n1 = int(n_samples * weight1)
    n2 = n_samples - n1
    
    data1 = np.random.normal(mu1, sigma1, n1)
    data2 = np.random.normal(mu2, sigma2, n2)
    
    return np.concatenate([data1, data2])


def model(x, a, mu1, s1, mu2, s2):
    return a*normal(x, mu1, s1) + (1-a)*normal(x, mu2, s2)

def test_anneal():
    np.random.seed(42)  # For reproducibility
    data = get_data(n_samples=1000, mu1=0, sigma1=1, mu2=3, sigma2=.5, weight1=0.7)
    
    # Fit the model
    bounds = ([0,1] , [-0.5, 0.5], [0, 1.5], [2, 3], [0, 2])  # Bounds for [mu, sigma]
    params = fit_distribution_anneal(model, data, bounds)

    h, bins = np.histogram(data, 50)
    dh = np.sqrt(h)/2
    x = (bins[:-1]+bins[1:])/2

    norm = np.sum(h*(bins[1]-bins[0]))
    
    plt.errorbar(x, h/norm, dh/len(data), color = "black", linestyle = "")
    #plt.scatter(x, h, color = "black", s=3)
    plt.step(x, model(x, *params), where="mid")
    plt.show()

def test_peak_fit():
    data = get_data(n_samples=1000, mu1=0.2, sigma1=1, mu2=3.1, sigma2=.5, weight1=0.7)
    fit = fit_peaks(data, np.array([0, 3]), peak_limits=0.5, sigma_init=2, anneal=True)

    h, bins = np.histogram(data, 50)
    dh = np.sqrt(h)
    x = (bins[:-1]+bins[1:])/2

    fit.axis = x

    norm = np.sum(h*(bins[1]-bins[0]))

    fig, ax = plot_fit(x, h/norm, fit, line_kwargs=args_to_dict(drawstyle = "steps"))
    fig.show()

    fig, ax = plt.subplots()

    #errorbars are misrepresented here because it looks nicer
    plt.errorbar(x, h/norm, dh/len(data), color = "black", linestyle = "", capsize=4)
    plt.step(x, fit.model(x, *fit.params), where="mid")
    plt.show()

    print(fit)
    print(fit.reduced_chi_sqrd(x[dh != 0], h[dh != 0]/norm, dh[dh != 0]/norm))


if __name__ == "__main__":
    test_peak_fit()
    #test_anneal()