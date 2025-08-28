"""
River Slope functions for PriorityFlow.

This module provides functions to apply minimum slope and secondary scaling
to river cells, ensuring proper flow characteristics in river networks.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def riv_slope(
    direction: np.ndarray,
    slopex: np.ndarray,
    slopey: np.ndarray,
    minslope: float,
    river_mask: np.ndarray,
    remove_sec: bool = False
) -> Dict[str, np.ndarray]:
    """
    Apply minimum slope and secondary scaling to river cells.
    
    A function that will apply a minimum slope threshold for the primary flow
    direction along river cells. This function can also limit secondary slopes
    out of river cells to zero.
    
    Parameters
    ----------
    direction : np.ndarray
        Nx by Ny matrix of flow directions following the convention (1=down, 2=left, 3=up, 4=right)
    slopex : np.ndarray
        Nx by Ny matrix of slopes in the x direction (should be face centered slopes as calculated with SlopeCalcStan)
    slopey : np.ndarray
        Nx by Ny matrix of slopes in the y direction (should be face centered slopes as calculated with SlopeCalcStan)
    minslope : float
        Threshold for slope adjustment. Any primary direction slope for a river cell will be adjusted such that abs(slope) >= minslope
    river_mask : np.ndarray
        Nx by Ny matrix indicating the mask of river cells that the min slope will be applied to
    remove_sec : bool, optional
        Flag, if set to True any secondary outflows on river cells will be set to zero. Defaults to False
    
    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing:
        - 'slopex': The adjusted Nx by Ny matrix of slopex values
        - 'slopey': The adjusted Nx by Ny matrix of slopey values
        - 'adj_mask': Nx by Ny matrix indicating cells that were adjusted (1=adjusted, 0=not adjusted)
        - 'SlopeOutlet': Nx by Ny matrix of the outlet slope for every grid cell
        - 'SlopeOutletNew': Nx by Ny matrix of the outlet slope for every grid cell after processing
    
    Notes
    -----
    This function implements a river slope adjustment algorithm that:
    1. Identifies river cells using the provided river mask
    2. Applies minimum slope thresholds to primary flow directions
    3. Optionally removes secondary outflow slopes
    4. Tracks which cells were adjusted and how
    
    The algorithm handles:
    - D4 flow direction processing
    - Primary slope threshold enforcement
    - Secondary slope removal (optional)
    - Adjustment tracking and reporting
    - Face-centered slope calculations
    """
    nx, ny = direction.shape
    
    # Columns: (1) deltax, (2) deltay, (3) direction number assuming you are looking downstream
    kd = np.array([
        [0, -1, 1],    # Down
        [-1, 0, 2],    # Left
        [0, 0, 3],     # Up
        [0, 0, 4]      # Right
    ])
    
    # Setup outputs
    outdirslope = np.zeros((nx, ny))
    outdirslope_new = np.zeros((nx, ny))
    adj_mask = np.zeros((nx, ny))
    slopex_new = slopex.copy()
    slopey_new = slopey.copy()
    
    # Loop over the domain adjusting slopes along river cells as needed
    # Only looping over internal cells
    for j in range(1, ny-1):
        for i in range(1, nx-1):
            if river_mask[i, j] == 1:
                # List all of the outflow slopes - order: down, left, up, right
                sec_out = np.array([
                    max(slopey[i, j-1], 0),      # Down
                    max(slopex[i-1, j], 0),      # Left
                    -min(slopey[i, j], 0),       # Up
                    -min(slopex[i, j], 0)        # Right
                ])
                
                # Adjust the outlet slope
                if not np.isnan(direction[i, j]):
                    # Zero out the primary flow direction outflow so this is just a list of secondary outflows
                    sec_out[int(direction[i, j]) - 1] = 0
                    
                    # Set the primary direction slope to be >= minslope
                    if direction[i, j] == 1 and abs(slopey[i, j-1]) < minslope:
                        slopey_new[i, j-1] = np.sign(slopey[i, j-1]) * minslope
                        outdirslope[i, j] = slopey[i, j-1]
                        outdirslope_new[i, j] = slopey_new[i, j-1]
                        adj_mask[i, j] = 0.5
                    elif direction[i, j] == 2 and abs(slopex[i-1, j]) < minslope:
                        slopex_new[i-1, j] = np.sign(slopex[i-1, j]) * minslope
                        outdirslope[i, j] = slopex[i-1, j]
                        outdirslope_new[i, j] = slopex_new[i-1, j]
                        adj_mask[i, j] = 0.5
                    elif direction[i, j] == 3 and abs(slopey[i, j]) < minslope:
                        slopey_new[i, j] = np.sign(slopey[i, j]) * minslope
                        outdirslope_new[i, j] = slopey_new[i, j]
                        adj_mask[i, j] = 0.5
                    elif direction[i, j] == 4 and abs(slopex[i, j]) < minslope:
                        slopex_new[i, j] = np.sign(slopex[i, j]) * minslope
                        outdirslope_new[i, j] = slopex_new[i, j]
                        adj_mask[i, j] = 0.5
                    
                    # If Remove.sec is TRUE set any secondary outflow slopes to 0
                    # The code below substitutes the kd values directly. 
                    # The R code is more explicit.
                    if remove_sec:
                        if np.max(sec_out) > 0:
                            # Down direction (secondary outflow)
                            if sec_out[0] > 0:
                                slopey_new[i, j-1] = 0
                                adj_mask[i, j] = adj_mask[i, j] + 1
                            
                            # Left direction (secondary outflow)
                            if sec_out[1] > 0:
                                slopex_new[i-1, j] = 0
                                adj_mask[i, j] = adj_mask[i, j] + 1
                            
                            # Up direction (secondary outflow)
                            if sec_out[2] > 0:
                                slopey_new[i, j] = 0
                                adj_mask[i, j] = adj_mask[i, j] + 1
                            
                            # Right direction (secondary outflow)
                            if sec_out[3] > 0:
                                slopex_new[i, j] = 0
                                adj_mask[i, j] = adj_mask[i, j] + 1
    
    output_list = {
        "slopex": slopex_new,
        "slopey": slopey_new,
        "adj_mask": adj_mask,
        "SlopeOutlet": outdirslope,
        "SlopeOutletNew": outdirslope_new
    }
    
    return output_list 