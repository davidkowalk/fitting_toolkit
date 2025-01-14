import unittest
import numpy as np
from src.fitting_toolkit import multivariate_fit 

#This file was generated using GPT4 and needs to be checked by a human before release!

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
        