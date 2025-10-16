"""
workhealthlab: A Python Package for Sociological Data Analysis
=============================================================

workhealthlab is a comprehensive toolkit for sociological research, providing
tools for data analysis, visualization, and modeling.

Modules:
    - analyses: Statistical modeling and regression analysis
    - cleansing: Data harmonization and cleaning
    - visuals: Data visualization functions
    - utils: Utility functions and styling
    - data: Data management
"""

__version__ = "0.1.0"

# Import submodules
from . import analyses
from . import cleansing
from . import visuals
from . import utils
from . import data

__all__ = [
    'analyses',
    'cleansing',
    'visuals',
    'utils',
    'data',
]
