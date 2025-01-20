"""
This is an example model for an experiment from the second undergraduate lab course. (232)
Here we demonstrate how to plot two separate fits in the same graph
"""

import numpy as np
from fitting_toolkit import curve_fit, plot_fit
from matplotlib import pyplot as plt

def lin(x, a, b):
    return a*x + b

# R = infty
x_inf = 1-np.array([20, 30, 40, 50, 60, 70, 80])/100
U_inf = np.array([3.2, 2.8, 2.4, 2.0, 1.6, 1.2, 0.8])
dU = 0.1

fit_inf = curve_fit(lin, x_inf, U_inf, dU, nsigma = 2, absolute_sigma = True)

# R = 50 Ohm
x_20 = x_inf
U_20 = np.array([2.7, 2.4, 2.0, 1.7, 1.5, 1.1, 0.7])
fit_20 = curve_fit(lin, x_20, U_20, dU, nsigma = 2, absolute_sigma = True)
#params_21, cov_21, lower_21, upper_21 = curve_fit(lin, x_20, U_20, dU, confidence_resolution = 50, nsigma = 2, absolute_sigma = True)
#params_22, cov_22, lower_22, upper_22 = curve_fit(lin, x_20, U_20, dU, confidence_resolution = 50, nsigma = 3, absolute_sigma = True)


#show both fits
fig, ax = plt.subplots()
ax.grid("both")
fig, ax = plot_fit(1-np.array([20, 30, 40, 50, 60, 70, 80])/100, U_inf, fit_inf, yerror=dU, fit_label="$R = \\infty$", confidence_label="2-$\\sigma$", fig = fig)
fig, ax = plot_fit(x_20, U_20, fit_20, yerror=dU, fit_color = "crimson", fit_label="$R = 50\\Omega$", confidence_label="2-$\\sigma$", fig = fig, ax = ax)
#fig, ax = plot_fit(x_20, U_20, lin, params_21, lower_21, upper_21, yerror=dU, confidence_resolution = 50, fit_color = "crimson", fit_label="$R = 50\\Omega$", confidence_label=None, fig = fig, ax = ax)
#fig, ax = plot_fit(x_20, U_20, lin, params_22, lower_22, upper_22, yerror=dU, confidence_resolution = 50, fit_color = "crimson", fit_label="$R = 50\\Omega$", confidence_label=None, fig = fig, ax = ax)
ax.legend()
ax.set_xlabel("x / l")
ax.set_ylabel("U / V")
plt.show()