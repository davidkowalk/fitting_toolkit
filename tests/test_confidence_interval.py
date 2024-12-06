import unittest
import numpy as np
from src.fitting_toolkit import get_sigma_probability, confidence_interval

def mock_model(x, *params):
    # Linear model: y = m * x + b
    return params[0] * x + params[1]


class TestConfidenceInterval(unittest.TestCase):
    def setUp(self):
        # Setup for the tests
        self.model = mock_model
        self.xdata = np.array([1, 2, 3, 4, 5])
        self.params = np.array([2, 1])  # Linear: y = 2x + 1
        self.cov = np.array([[0.1, 0], [0, 0.1]])  # Small variance
        self.resamples = 1000
        self.nsigma = 1

    def test_nominal_case(self):
        # Test with normal inputs
        lower, upper = confidence_interval(self.model, self.xdata, self.params, self.cov, self.resamples, self.nsigma)
        self.assertEqual(len(lower), len(self.xdata))
        self.assertEqual(len(upper), len(self.xdata))
        self.assertTrue(np.all(lower < upper))  # Ensure bounds are valid

    def test_small_resamples(self):
        # Test with very small number of resamples
        lower, upper = confidence_interval(self.model, self.xdata, self.params, self.cov, 1, self.nsigma)
        self.assertEqual(len(lower), len(self.xdata))
        self.assertEqual(len(upper), len(self.xdata))

    def test_high_nsigma(self):
        # Test with high nsigma
        lower, upper = confidence_interval(self.model, self.xdata, self.params, self.cov, self.resamples, 3)
        self.assertEqual(len(lower), len(self.xdata))
        self.assertEqual(len(upper), len(self.xdata))
        self.assertTrue(np.all(lower < upper))

    def test_single_xdata_point(self):
        # Test with a single xdata point
        xdata = np.array([5])
        lower, upper = confidence_interval(self.model, xdata, self.params, self.cov, self.resamples, self.nsigma)
        self.assertEqual(len(lower), len(xdata))
        self.assertEqual(len(upper), len(xdata))

    def test_empty_xdata(self):
        # Test with empty xdata
        xdata = np.array([])
        lower, upper = confidence_interval(self.model, xdata, self.params, self.cov, self.resamples, self.nsigma)
        self.assertEqual(len(lower), 0)
        self.assertEqual(len(upper), 0)

    def test_performance(self):
        # Test with large resamples and xdata
        large_xdata = np.linspace(0, 100, 1000)
        resamples = 10000
        lower, upper = confidence_interval(self.model, large_xdata, self.params, self.cov, resamples, self.nsigma)
        self.assertEqual(len(lower), len(large_xdata))
        self.assertEqual(len(upper), len(large_xdata))

if __name__ == '__main__':
    unittest.main()
