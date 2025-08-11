"""
Get Border functions for PriorityFlow.

This module provides functions to identify border cells by reading in a mask
and returning a mask of the border cells for irregular boundaries.
"""

import numpy as np
from typing import Optional


def get_border(mask_mat: np.ndarray) -> np.ndarray:
    """
    Identify border cells.
    
    Function that reads in a mask and returns a mask of the border cells for the
    irregular boundary. Border cells are identified as cells that have fewer than
    4 valid neighbors within the domain.
    
    Parameters
    ----------
    mask_mat : np.ndarray
        Matrix mask with values of 0 for cells outside the domain and 1 for cells inside the domain
    
    Returns
    -------
    np.ndarray
        Matrix mask where 1 indicates border cells and 0 indicates non-border cells
    
    Notes
    -----
    This function implements a border detection algorithm that:
    1. Identifies cells at the domain boundary
    2. Counts valid neighbors for each cell
    3. Classifies cells with fewer than 4 neighbors as border cells
    4. Handles irregular domain boundaries
    
    The algorithm uses D4 connectivity (4-directional) to identify border cells:
    - Cells with 4 valid neighbors are interior cells (not borders)
    - Cells with fewer than 4 valid neighbors are border cells
    - Only cells within the domain mask are considered
    
    Border cells are typically:
    - Domain edge cells
    - Cells adjacent to masked areas
    - Cells in irregular boundary regions
    - Cells with incomplete neighbor connectivity
    """
    ny, nx = mask_mat.shape
    
    # Initialize border matrix with 1s (assuming all cells are borders initially)
    border = np.ones((ny, nx))
    
    # Calculate neighbor connectivity for internal cells
    # For each cell, count how many of its 4 neighbors are within the domain
    border[1:(ny-1), 1:(nx-1)] = (
        mask_mat[0:(ny-2), 1:(nx-1)] +      # North neighbor
        mask_mat[2:ny, 1:(nx-1)] +          # South neighbor
        mask_mat[1:(ny-1), 0:(nx-2)] +      # West neighbor
        mask_mat[1:(ny-1), 2:nx]            # East neighbor
    )
    
    # Apply the original mask to only consider cells within the domain
    border = border * mask_mat
    
    # Classify cells based on neighbor count:
    # - Cells with 4 neighbors are interior cells (set to 0)
    # - Cells with fewer than 4 neighbors are border cells (set to 1)
    border[(border < 4) & (border != 0)] = 1
    border[border == 4] = 0
    
    return border 