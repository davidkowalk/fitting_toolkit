import unittest
from unittest.mock import patch
import numpy as np
from src.fitting_toolkit.fitting_toolkit import plot_fit, curve_fit

"""
Runs Tests for ./src/fitting_toolkit.py
Windows: python -m unittest discover -s tests
Linux: python3 -m unittest discover -s tests
"""

class TestCurveFit(unittest.TestCase):
    
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

        with patch('src.fitting_toolkit.fitting_toolkit.curve_fit_scipy', wraps=mock_curve_fit):
            with self.assertWarns(RuntimeWarning) as cm:
                fit = curve_fit(
                    model, xdata, ydata, yerror=yerror
                )
            
            self.assertIn("Covariance matrix includes infinite values", str(cm.warning))
            self.assertIsNone(fit.lower)
            self.assertIsNone(fit.upper)


from src.fitting_toolkit import multivariate_fit 

#This section was generated using GPT4 and needs to be checked by a human before release!

class TestMultivariateFit(unittest.TestCase):
    def setUp(self):
        # Example linear model for testing
        self.model = lambda input, a, b: a * input[:, 0] + b

        # Input data: 2D array with shape (n, m)
        self.input = np.array([[1], [2], [3], [4]])

        # Output data: Observed values
        self.output = np.array([2.2, 4.1, 6.3, 8.0])

        # Standard deviation of errors (weights)
        self.sigma = np.array([0.1, 0.1, 0.1, 0.1])

        # Initial guess for parameters
        self.theta_0 = np.array([1.0, 0.0])

        np.random.seed(41950)

    def test_optimization_result(self):
        """Test that the optimized parameters are close to the expected values."""
        expected_parameters = [2.0, 0.2]  # Approximate expected values

        popt, pcov = multivariate_fit(self.model, self.input, self.output, self.sigma, self.theta_0)
        np.testing.assert_allclose(popt, expected_parameters, atol=0.1, err_msg="Optimized parameters do not match expected values.")
        
        self.assertIsInstance(pcov, np.ndarray, "Covariance matrix is not a numpy array.")
        self.assertEqual(pcov.shape, (2, 2), "Covariance matrix shape is incorrect.")

    def test_with_constant_output(self):
        """Test the function with a constant output to check edge cases."""
        constant_output = np.array([5.0, 5.0, 5.0, 5.0])
        self.output = constant_output

        popt, pcov = multivariate_fit(self.model, self.input, self.output, self.sigma, self.theta_0)

        # For a constant output, the slope should ideally be 0
        self.assertAlmostEqual(popt[0], 0, delta=1e-2, msg="Slope parameter is incorrect for constant output.")

    def test_with_incorrect_input_shapes(self):
        """Test the function with mismatched input shapes."""
        incorrect_input = np.array([[1, 2], [3, 4], [5, 6]])  # Shape mismatch with output

        with self.assertRaises(ValueError, msg="Shape mismatch between input and output should raise an error."):
            multivariate_fit(self.model, incorrect_input, self.output, self.sigma, self.theta_0)

    def test_with_scalar_sigma(self):
        """Test the function with sigma provided as a scalar."""
        scalar_sigma = 0.1

        popt, pcov = multivariate_fit(self.model, self.input, self.output, scalar_sigma, self.theta_0)

        self.assertEqual(len(popt), 2, "Optimized parameters length is incorrect.")

    #========================================================================================================
    # Human Written Tests:

    def test_slope(self):

        x = np.linspace(-3, 3, 200)
        y = np.linspace(1, 4, 150)

        xy = np.meshgrid(x, y)

        def model(xy, a, b, z0):
            return a*xy[0] + b*xy[1] + z0
        
        params = (0.3141, 1.312, 1.61)
        z = model(xy, *params) + 0.1*np.random.random(np.shape(xy))

        theta_0 = (0, 0, 1.5)
        popt, pcov = multivariate_fit(model, xy, z, np.ones_like(z), theta_0)
        np.testing.assert_allclose(popt, params, atol=0.1, err_msg="Optimized parameters do not match expected values.")


    def test_gauss(self):

        params = [2.2, 2, 2.5, 0.8, 0.4]

        def model(xy, A, x0, y0, sx, sy):
            return A * np.exp( -0.5 * (((xy[0] - x0)/sx)**2 + ((xy[1] - y0) / sy)**2))

        x = np.linspace(0,4, 157)
        y = np.linspace(0,5, 150)

        xy_data = np.meshgrid(x, y)

        np.random.seed(10172)
        z = model(xy_data, *params)
        z += 0.05*np.random.normal(size=z.shape)
        dz = 0.1

        theta_0 = np.array([2, 2, 2, 1, 1])
        popt, pcov = multivariate_fit(model, xy_data, z, sigma=dz, theta_0=theta_0)
        np.testing.assert_allclose(popt, params, rtol=0.1, err_msg="Optimized parameters do not match expected values.")
        
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