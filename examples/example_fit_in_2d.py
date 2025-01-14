import numpy as np


def model(xy, A, x0, y0, sx, sy):
    #xy is a list of tuples (x, y)
    x, y = xy
    return A * np.exp( -0.5 * (((x - x0)/sx)**2 + ((y - y0) / sy)**2))

x = np.linspace(0,4, 200)
y = np.linspace(0,5, 200)

xy_data = np.meshgrid(x, y)

np.random.seed(10172)
z = model(xy_data, 2.2, 2, 2.5, 0.8, 0.4)
z += 0.05*np.random.normal(size=z.shape)

theta_0 = np.array([3, 2, 2, 1, 1])

#fit data
from fitting_toolkit.fitting_toolkit import custom_fit as fit
popt, pcov = fit(model, xy_data, z, sigma=np.ones_like(z), theta_0=theta_0)
print(popt)
print(np.sqrt(np.diag(pcov)))

# Display Data and fit
from matplotlib import pyplot as plt
plt.pcolormesh(x, y, z)
#plt.contour(x, y, z, 6, colors = "gray")
plt.contour(x, y, model(xy_data, *popt), 5, colors = "black")
plt.show()
