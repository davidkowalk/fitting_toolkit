import unittest
import numpy as np
from src.utils import get_sigma_probability
from src import confidence_interval

def mock_model(x, m, b):
    # Linear model: y = m * x + b
    return m * x + b


class TestConfidenceInterval(unittest.TestCase):
    def setUp(self):
        # Setup for the tests
        self.model = mock_model
        self.xdata = np.array([1, 2, 3, 4, 5])
        self.params = np.array([2, 1]) 
        self.cov = np.array([[0.1, 0], [0, 0.1]])
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

    def test_statistical_accuracy_with_resampling(self):
        """
        Test whether the expected number of resampled parameters
        produce points that fall within the calculated confidence interval.
        """

        # Setup
        np.random.seed(581) 
        true_params = np.array([2, 1])
        cov = np.array([[0.1, 0.05], [0.05, 0.1]])

        xdata = np.linspace(0, 10, 50)
        resamples = 500
        nsigma = 1

        #Compute confidence intervals
        lower, upper = confidence_interval(self.model, xdata, true_params, cov, resamples, nsigma)

        # Resample parameters
        resampled_params = np.random.multivariate_normal(true_params, cov, resamples)

        # Check how many resampled parameters produce points inside interval
        in_interval = 0
        total_points = len(xdata) * resamples

        for params in resampled_params:
            predictions = self.model(xdata, *params)
            in_interval += np.sum((predictions >= lower) & (predictions <= upper))

        #Compute the coverage
        empirical_coverage = in_interval / total_points
        expected_coverage = get_sigma_probability(nsigma)

        #Compare Coverage
        tolerance = 0.02
        self.assertAlmostEqual(empirical_coverage, expected_coverage, delta=tolerance, msg=f"\nEmpirical coverage {empirical_coverage:.3f} does not match expected {expected_coverage:.3f}")

if __name__ == '__main__':
    unittest.main()
