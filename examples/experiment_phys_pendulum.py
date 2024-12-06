"""
This is an example model for a quick experiment on a physical pendulum.
len describes the measured pendulum length in cm, t the period length in seconds.
"""
import numpy as np
from fitting_toolkit import curve_fit, plot_fit, array
from matplotlib import pyplot as plt

def model(l, a, b):
    return a*np.sqrt(l) + b 

len = array(60, 50, 40, 30)
t = array(13.14, 12.35, 10.70, 9.3)/10

dlen = 0.5
dt = 0.5/10

confidence_resolution = 100

fit = curve_fit(model, len, t, yerror=dt, model_resolution = confidence_resolution, absolute_sigma = True)
fig, ax = plot_fit(len, t, fit, xerror=dlen, yerror=dt)
ax.set_xlabel("Pendulum Length / cm")
ax.set_ylabel("Oscillation Time T / s")
plt.show()