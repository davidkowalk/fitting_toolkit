import unittest
import numpy as np
from src.fitting_toolkit import plot_fit, curve_fit

"""
Runs Tests for ./src/fitting_toolkit.py
Windows: python -m unittest discover -s tests
Linux: python3 -m unittest discover -s tests
"""

class UnitTests(unittest.TestCase):
    
    def test_curve_fit_linear(self):
        #This test may not pass simply due to statistics

        #print("test_curve_fit_linear")
        #print("Please be aware that this test may fail due to statistics.")
        #print("If this test fails please run again before reporting.")

        def model(x, m, c):
            return m * x + c

        #Define Parameters
        np.random.seed(420) #set seed for reproducability
        # note that the seed 31416926 makes this test fail reliably
        n = 10
        x = np.linspace(0, 2, n)
        dy = 1
        m = np.random.normal(0, 2)
        c = np.random.normal(2, 3)

        #simulate Data
        y = m*x + c + np.random.normal(loc = 0, scale = dy, size = n)
        fit = curve_fit(model, x, y, yerror=None, nsigma=1, absolute_sigma = True)
        params, cov, lower, upper = fit.params, fit.cov, fit.lower, fit.upper
        #y_fit = model(x, *params)

        diff = (np.abs(m  - params[0]), np.abs(c - params[1]))
        sigmas = np.sqrt(np.diagonal(cov))

        self.assertLessEqual(diff[0], sigmas[0]*2)
        self.assertLessEqual(diff[1], sigmas[1]*2)

        fig, ax = plot_fit(x, y, fit, yerror=dy)
        fig.savefig("./tests/plot.png")