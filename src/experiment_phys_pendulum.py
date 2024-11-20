"""
This is an example model for a quick experiment on a physical pendulum.
len describes the measured pendulum length in cm, t the period length in seconds.
"""
import numpy as np
from fitting_toolkit import *

def model(l, a, b):
    return a*np.sqrt(l) + b 

len = np.array((60, 50, 40, 30))
t = np.array((13.14, 12.35, 10.70, 9.3))/10

dlen = 0.5
dt = 0.5/10

confidence_resolution = 100

params, cov, lower_conf, upper_conf = curve_fit(model, len, t, yerror=dt, confidence_resolution = confidence_resolution)
fig, ax = plot_fit(len, t, model, params, lower_conf, upper_conf, xerror=dlen, yerror=dt, confidence_resolution = confidence_resolution)
ax.set_xlabel("Pendulum Length / cm")
ax.set_ylabel("Oscillation Time T / s")
plt.show()