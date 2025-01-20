import unittest
from unittest.mock import patch, Mock
import numpy as np
from scipy.integrate import quad
from src.fitting_toolkit.utils import get_sigma_probability, generate_thresholds, generate_gaussian_mix, array, args_to_dict
from src.fitting_toolkit.utils import version, versions, stats

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

    def test_invalid_parameter_count(self):
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
    
    def test_array(self):
        list = [3, 1, 4, 1, 5, 9]

        list_arr = array(*list)
        np.testing.assert_equal(list, list_arr)
        self.assertEqual(type(list_arr), np.ndarray)

    def test_kwargs_to_dict(self):
        kw1 = "Nuclear Power"
        kw2 = "Japan"
        kw3 = "Calorimeter"
        kw4 = "Dinosour"

        kw_dict = args_to_dict(kw1=kw1, kw2=kw2, kw3=kw3, kw4=kw4)
        self.assertEqual(type(kw_dict), dict)
        self.assertEqual(kw1, kw_dict["kw1"])
        self.assertEqual(kw2, kw_dict["kw2"])
        self.assertEqual(kw3, kw_dict["kw3"])
        self.assertEqual(kw4, kw_dict["kw4"])


    #Test version utils
    @patch('requests.get')
    def test_versions_print(self, mock_get):
        # Mock response for requests.get
        mock_get.return_value.json.return_value = {
            "releases": {
                "1.0.0": [{"upload_time": "2023-01-01T12:00:00"}],
                "2.0.0": [{"upload_time": "2024-01-01T12:00:00"}],
            }
        }

        with patch("builtins.print") as mock_print:  # Mock print function
            versions(print_versions=True, return_list=False)

        # Assert print statements
        mock_print.assert_any_call("Version\tDate")
        mock_print.assert_any_call("1.0.0\t2023-01-01")
        mock_print.assert_any_call("2.0.0\t2024-01-01")

    @patch('requests.get')
    def test_versions_return(self, mock_get):
        # Mock response for requests.get
        mock_get.return_value.json.return_value = {
            "releases": {
                "1.0.0": [{"upload_time": "2023-01-01T12:00:00"}],
                "2.0.0": [{"upload_time": "2024-01-01T12:00:00"}],
            }
        }

        result = versions(print_versions=False, return_list=True)
        # Assert the returned value
        self.assertEqual(
            list(result),
            [
                ("1.0.0", [{"upload_time": "2023-01-01T12:00:00"}]),
                ("2.0.0", [{"upload_time": "2024-01-01T12:00:00"}]),
            ],
        )

    @patch('importlib.metadata.version')
    def test_version(self, mock_version):
        # Mock response for importlib.metadata.version
        mock_version.return_value = "3.1.2"

        result = version()
        # Assert the version is returned correctly
        self.assertEqual(result, "3.1.2")

    @patch('requests.get')
    def test_stats(self, mock_get):
        # Mock response for requests.get
        mock_get.return_value.json.return_value = {
            "data": {"last_month": 12345, "last_week": 2345}
        }

        result = stats()
        # Assert the returned statistics
        self.assertEqual(result, {"last_month": 12345, "last_week": 2345})