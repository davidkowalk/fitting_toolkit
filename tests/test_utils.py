import unittest
import numpy as np
from scipy.integrate import quad
from src.fitting_toolkit.utils import get_sigma_probability, generate_thresholds, generate_gaussian_mix

class TestUtils(unittest.TestCase):

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


    #===============
    # Test Gaussian
    #===============
    def test_generate_gaussian_mix_single(self):
        gaussian_mix = generate_gaussian_mix(1)

        self.assertTrue(callable(gaussian_mix))
        x = np.linspace(-5, 5, 100)
        params = (0, 1)

        result = gaussian_mix(x, *params)
        self.assertEqual(result.shape, x.shape)
        self.assertAlmostEqual(quad(gaussian_mix, -np.inf, np.inf, args = params)[0], 1)

    def test_generate_gaussian_mix_multiple(self):
        gaussian_mix = generate_gaussian_mix(2)

        self.assertTrue(callable(gaussian_mix))

        x = np.linspace(-5, 5, 100)
        params = (0, 1, 0.6, 2, 1)

        result = gaussian_mix(x, *params)

        self.assertEqual(result.shape, x.shape)
        self.assertAlmostEqual(quad(gaussian_mix, -np.inf, np.inf, args = params)[0], 1)

    def tes_invalid_parameter_count(self):
        gaussian_mix = generate_gaussian_mix(2)

        # Define incorrect number of parameters (should be 3*n-1 = 5)
        params = (0, 1, 1, 2)  # Incorrect length (only 4 parameters)

        with self.assertRaises(ValueError):
            gaussian_mix(np.linspace(-5, 5, 100), *params)

    def test_edge_case_empty_input(self):
        gaussian_mix = generate_gaussian_mix(0)
        params = []
        with self.assertRaises(ValueError):
            gaussian_mix(np.linspace(-5, 5, 100), *params)

    def test_gaussian_output_shape(self):
        gaussian_mix = generate_gaussian_mix(3)
        params = (0, 1, 0.5, 1, 1, 0.3, 2, 0.5)
        
        x = np.linspace(-5, 5, 100)
        result = gaussian_mix(x, *params)
        self.assertEqual(result.shape[0], x.shape[0])
