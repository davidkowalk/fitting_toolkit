import unittest
import numpy as np
from src.fitting_toolkit import get_sigma_probability, generate_thresholds, confidence_interval, curve_fit

"""
Runs Tests for ./src/fitting_toolkit.py
Windows: python -m unittest discover -s tests
Linux: python3 -m unittest discover -s tests
"""

class UnitTests(unittest.TestCase):
    
    def test_sigma(self):
        print("Testing Sigma Probability")
        n = np.array((0.5, 1, 1.5, 2, 2.5, 3))
        p = get_sigma_probability(n)
        p_expected = [0.382924922548026, 0.682689492137086, 0.866385597462284, 0.954499736103642, 0.987580669348448, 0.997300203936740]

        for i, P in enumerate(p):
            self.assertAlmostEqual(P, p_expected[i], places = 10)

    def test_threshold(self):
        x = np.linspace(0, 10, 100)
        lower, upper = generate_thresholds(x, lower_frac=0.1, upper_frac=0.9)
        self.assertEqual(len(x[x<lower]), 10)
        self.assertEqual(len(x[x<upper]), 90)

    def test_confidence_interval_linear(self):
        pass
    
    def test_curve_fit_linear(self):
        pass
        #define data
        #define covariance
        #fit line to data
        #test if 2/3 of datapoint are in confidence interval