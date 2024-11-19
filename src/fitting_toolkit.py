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

def curve_fit(model, xdata: np.array, ydata: np.array, yerror = None, resamples = 5000, **kwargs) -> tuple[np.array, np.array, np.array, np.array]:
    params, cov = sc_curve_fit(f = model, xdata = xdata, ydata = ydata, sigma = yerror, absolute_sigma=True, **kwargs)
    lower_conf, upper_conf = confidence_interval(model, xdata, params, cov, resamples)

    return params, cov, lower_conf, upper_conf

def plot_fit(xdata, ydata, model, params, lower, upper, xerror = None, yerror = None, markersize = 4, capsize = 4) -> tuple[plt.figure, plt.axes]:
    fig, ax = plt.subplots()

    #ax.scatter(xdata, ydata, color = "black", s = 2)
    ax.errorbar(xdata, ydata, yerr = yerror, xerr = xerror, fmt=".", linestyle = "", color = "black", capsize=capsize, markersize = markersize)
    ax.plot(xdata, model(xdata, *params), color = "black", linewidth = 1, linestyle = "-")
    ax.plot(xdata, upper, color = "black", linewidth = 0.75, linestyle = "--", label = "1$\sigma$-Confidence")
    ax.plot(xdata, lower, color = "black", linewidth = 0.75, linestyle = "--")

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid("both")
    ax.set_axisbelow(True)

    return fig, ax

