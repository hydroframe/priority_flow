"""
PriorityFlow - Python toolkit for topographic processing for hydrologic models.

This is a Python port of the original R package developed by Laura Condon and Reed Maxwell.
"""

__version__ = "1.0.0"
__author__ = "Laura Condon & Reed Maxwell (Python port)"

from .init_queue import init_queue
from .d4_traverse import d4_traverse_b
from .slope_calc_standard import slope_calc_standard
from .slope_calc_upwind import slope_calc_upwind
from .data_loader import (
    load_dem,
    load_watershed_mask,
    load_river_mask,
    load_all_test_data,
    get_data_info,
)
from .fix_drainage import fix_drainage
from .downstream_extract import path_extract
from .define_subbasins import calc_subbasins
from .cal_stream_order import calc_stream_order
from .write_raster import write_raster
from .stream_distance import stream_dist
from .stream_traverse import stream_traverse
from .river_smoothing import river_smooth
from .run_pf import run_pf
from .river_slope import riv_slope
from .peak_distance import peak_dist
from .linear_distance import lin_dist
from .flat_fix import fix_flat
from .get_border import get_border
from .find_orphan import find_orphan
from .drainage_area import drainage_area, calculate_drainage_area_alt
from .define_watershed import delin_watershed
from .border_direction_fix import fix_border_dir
from .calc_flow import calc_flow

__all__ = [
    "init_queue",
    "d4_traverse_b",
    "slope_calc_standard",
    "slope_calc_upwind",
    "load_dem",
    "load_watershed_mask",
    "load_river_mask",
    "load_all_test_data",
    "get_data_info",
    "fix_drainage",
    "path_extract",
    "calc_subbasins",
    "calc_stream_order",
    "write_raster",
    "stream_dist",
    "stream_traverse",
    "river_smooth",
    "run_pf",
    "riv_slope",
    "peak_dist",
    "lin_dist",
    "fix_flat",
    "get_border",
    "find_orphan",
    "drainage_area",
    "calculate_drainage_area_alt",
    "delin_watershed",
    "fix_border_dir",
    "calc_flow",
]
