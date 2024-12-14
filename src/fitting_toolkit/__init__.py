"""
fitting_toolkit: A Python package for fitting models and related utilities.

Modules:
- fitting_toolkit: Core module for toolkit functionalities.
- utils: Helper utilities for data manipulation and preprocessing.
- fit: Contains fitting functions.
"""

from . import fit
from . import utils
from .fitting_toolkit import Fit, confidence_interval, curve_fit, fit_peaks, plot_fit

# Define __all__ to specify what gets imported with "from fitting_toolkit import *"
from .fitting_toolkit import __all__ as toolkit_all
__all__ = toolkit_all