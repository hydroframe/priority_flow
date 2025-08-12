"""
Border direction fixing functions for PriorityFlow.

This module contains functions for correcting the flow directions of border cells
based on slope analysis to ensure proper drainage patterns.
"""

import numpy as np
from typing import Tuple


def fix_border_dir(
    direction: np.ndarray,
    dem: np.ndarray,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4)
) -> np.ndarray:
    """
    Fix the directions of border cells.
    
    Originally all border cells are initialized to point out of the domain. 
    This function checks the slopes and areas to determine if they need to swap
    direction to point into the domain instead.
    
    Parameters
    ----------
    direction : np.ndarray
        Flow direction matrix [nx, ny] - will be modified in place
    dem : np.ndarray
        Digital Elevation Model matrix [nx, ny]
    d4 : tuple, optional
        Direction numbering system for down, left, top, right.
        Defaults to (1, 2, 3, 4).
    
    Returns
    -------
    np.ndarray
        The modified direction array
    
    Notes
    -----
    This is a Python port of the R function FixBorderDir from PriorityFlow.
    The function analyzes slopes at domain boundaries to determine whether
    border cells should drain inward or outward.
    
    WARNING: This function modifies the input direction array in place
    for backwards compatibility with the R version.
    
    Direction mapping:
    - d4[0] = 1: Down
    - d4[1] = 2: Left  
    - d4[2] = 3: Up
    - d4[3] = 4: Right
    """
    
    # Get dimensions
    ny = direction.shape[1]  # Number of columns
    nx = direction.shape[0]  # Number of rows
    
    # Check top border (y = ny-1 in 0-indexed Python)
    # If slope between ny-1 and ny-2 is positive, flow should point in (down)
    sy_top = dem[:, ny-1] - dem[:, ny-2]
    flip_top = np.where(sy_top > 0)[0]
    direction[flip_top, ny-1] = d4[0]  # Point down (into domain)
    
    # Check bottom border (y = 0 in 0-indexed Python)
    # If slope between 1 and 0 is negative, flow should point in (up)
    sy_bot = dem[:, 1] - dem[:, 0]
    flip_bot = np.where(sy_bot < 0)[0]
    direction[flip_bot, 0] = d4[2]  # Point up (into domain)
    
    # Check right border (x = nx-1 in 0-indexed Python)
    # If slope between nx-1 and nx-2 is positive, flow should point in (left)
    sx_right = dem[nx-1, :] - dem[nx-2, :]
    flip_right = np.where(sx_right > 0)[0]
    direction[nx-1, flip_right] = d4[1]  # Point left (into domain)
    
    # Check left border (x = 0 in 0-indexed Python)
    # If slope between 1 and 0 is negative, flow should point in (right)
    sx_left = dem[1, :] - dem[0, :]
    flip_left = np.where(sx_left < 0)[0]
    direction[0, flip_left] = d4[3]  # Point right (into domain)
    
    return direction


 