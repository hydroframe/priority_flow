"""
PriorityFlow - Python toolkit for topographic processing for hydrologic models.

This is a Python port of the original R package developed by Laura Condon and Reed Maxwell.
"""

__version__ = "1.0.0"
__author__ = "Laura Condon & Reed Maxwell (Python port)"

# Import core functions
from .core import (
    init_queue,
    d4_traverse_b,
)

# Import slope functions
from .slopes import (
    slope_calc_standard,
    slope_calc_upwind,
    river_slope,
    river_smoothing,
)

# Import utility functions
from .utils import (
    get_border,
    fix_drainage,
    fix_flat,
    calc_flow,
)

# Import I/O functions
from .io import (
    write_raster,
    read_raster,
)

# Make key functions available at package level
__all__ = [
    "init_queue",
    "d4_traverse_b", 
    "slope_calc_standard",
    "slope_calc_upwind",
    "river_slope",
    "river_smoothing",
    "get_border",
    "fix_drainage",
    "fix_flat",
    "calc_flow",
    "write_raster",
    "read_raster",
] 