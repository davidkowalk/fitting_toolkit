import unittest
from unittest.mock import patch
import numpy as np
from src.fitting_toolkit import get_sigma_probability, generate_thresholds, curve_fit
from src.fitting_toolkit import plot_fit

"""
Runs Tests for ./src/fitting_toolkit.py
Windows: python -m unittest discover -s tests
Linux: python3 -m unittest discover -s tests
"""

class UnitTests(unittest.TestCase):
    
    def test_get_sigma_probability(self):
        #print("get_sigma_probability")
        n = np.array((0.5, 1, 1.5, 2, 2.5, 3))
        p = get_sigma_probability(n)
        p_expected = [0.382924922548026, 0.682689492137086, 0.866385597462284, 0.954499736103642, 0.987580669348448, 0.997300203936740]

        for i, P in enumerate(p):
            self.assertAlmostEqual(P, p_expected[i], places = 10)

    def test_threshold(self):
        #print("test_threshold")
        x = np.linspace(0, 10, 100)
        lower, upper = generate_thresholds(x, lower_frac=0.1, upper_frac=0.9)
        self.assertEqual(len(x[x<lower]), 10)
        self.assertEqual(len(x[x<upper]), 90)
    
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
        params, cov, lower, upper = curve_fit(model, x, y, yerror=None, nsigma=1, absolute_sigma = True)
        #y_fit = model(x, *params)

        diff = (np.abs(m  - params[0]), np.abs(c - params[1]))
        sigmas = np.sqrt(np.diagonal(cov))

        self.assertLessEqual(diff[0], sigmas[0]*2)
        self.assertLessEqual(diff[1], sigmas[1]*2)

        fig, ax = plot_fit(x, y, model, params, lower, upper, yerror=dy)
        fig.savefig("./tests/plot.png")

    def test_infinite_covariance_warning(self):

        def mock_curve_fit(*args, **kwargs):
            """Mock function to simulate infinite covariance matrix in curve_fit."""
            params = np.array([1.0, 1.0])
            cov = np.array([[np.inf, 0], [0, np.inf]])
            return params, cov

        def model(x, a, b):
            return a * x + b
        
        xdata = np.array([1, 2, 3, 4, 5])
        ydata = model(xdata, 2, 1)
        yerror = np.array([0.1] * len(xdata))

        with patch('src.fitting_toolkit.sc_curve_fit', wraps=mock_curve_fit):
            with self.assertWarns(RuntimeWarning) as cm:
                params, cov, lower_conf, upper_conf = curve_fit(
                    model, xdata, ydata, yerror=yerror
                )
            
            self.assertIn("Covariance matrix includes infinite values", str(cm.warning))
            self.assertIsNone(lower_conf)
            self.assertIsNone(upper_conf)