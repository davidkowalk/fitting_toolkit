import unittest
import numpy as np
from src.fitting_toolkit.fitting_toolkit import fit_peaks, Fit
from src.fitting_toolkit.utils import normal

class TestFitPeaks(unittest.TestCase):
    def setUp(self):
        """Set up common variables for the tests."""
        # Generate synthetic events (random data points)
        np.random.seed(42)
        self.peaks = np.array([0, 5])
        self.events = np.concatenate([
            np.random.normal(loc=self.peaks[0], scale=1, size=100),  # Peak 1
            np.random.normal(loc=self.peaks[1], scale=1.5, size=100),  # Peak 2
        ])
        self.peak_estimates = self.peaks # Initial guesses for peak locations
        self.peak_limits = 1  # Allow peaks to deviate by up to ±2 units
        self.sigma_init = 2  # Initial guess for standard deviation

    def test_single_peak(self):
        """Test fitting a single Gaussian peak."""
        events = np.random.normal(loc=2, scale=0.5, size=100)
        peak_estimates = np.array([2])  # Initial guess
        peak_limits = 1
        sigma_init = 0.5

        # Call fit_peaks
        result = fit_peaks(events, peak_estimates, peak_limits, sigma_init)

        # Check if the result is a Fit object
        self.assertIsInstance(result, Fit)

        # Check that the fitted mean is close to the true value
        fitted_means = result.params[0::3]  # Extract means (mu)
        self.assertAlmostEqual(fitted_means[0], 2, delta=0.1)

    def test_multiple_peaks(self):
        """Test fitting multiple Gaussian peaks."""
        result = fit_peaks(
            self.events, self.peak_estimates, self.peak_limits, self.sigma_init, anneal = True
        )

        # Check if the result is a Fit object
        self.assertIsInstance(result, Fit)

        # Check the fitted means are close to the true values
        fitted_means = result.params[0::3]  # Extract means (mu)
        np.testing.assert_allclose(fitted_means, [0, 5], atol=0.5)

    def test_annealing(self):
        """Test fitting with simulated annealing."""
        result = fit_peaks(self.events, self.peak_estimates, self.peak_limits, self.sigma_init, anneal=True)

        # Check if the result is a Fit object
        self.assertIsInstance(result, Fit)

        # Check the fitted means are close to the true values
        fitted_means = result.params[0::3]  # Extract means (mu)
        np.testing.assert_allclose(fitted_means, [0, 5], atol=0.5)

    def test_custom_model(self):
        """Test fitting with a custom model."""
        def custom_model(x, m1, s1, a, m2, s2):
            return normal(x, m1, s1, a) + normal(x, m2, s2, 1-a)

        result = fit_peaks(self.events, self.peak_estimates, self.peak_limits, self.sigma_init, anneal = True, model=custom_model)
        self.assertIsInstance(result, Fit)

        # Check the fitted means are close to the true values
        peak1, peak2 = result.params[0], result.params[3]
        self.assertAlmostEqual(peak1, 0, delta=0.5)
        self.assertAlmostEqual(peak2, 5, delta=0.5)


if __name__ == "__main__":
    unittest.main()
