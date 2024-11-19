import numpy as np
from fitting_toolkit import *

def model(l, a, b):
    return a*np.sqrt(l) + b 

len = np.array((60, 50, 40, 30))
t = np.array((13.14, 12.35, 10.70, 9.3))/10

dlen = 0.5
dt = 0.5/10

params, cov, lower_conf, upper_conf = curve_fit(model, len, t, yerror=dt)
fig, ax = plot_fit(len, t, model, params, lower_conf, upper_conf, xerror=dlen, yerror=dt)
ax.set_xlabel("Pendulum Length / cm")
ax.set_ylabel("Oscillation Time T / s")
plt.show()