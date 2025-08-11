"""
Find Orphan functions for PriorityFlow.

This module provides functions to find orphan branches - unprocessed river cells
that have D8 neighbors on the river network or on the boundary. This is useful
for identifying missed cells during river network processing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def find_orphan(
    dem: np.ndarray,
    mask: np.ndarray,
    marked: np.ndarray
) -> Dict[str, Union[int, np.ndarray]]:
    """
    Find orphan branches in river network processing.
    
    Function to look for unprocessed river cells that have D8 neighbors on the
    river network or on the boundary. This is useful for identifying missed cells
    during river network processing and ensuring complete coverage.
    
    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model matrix
    mask : np.ndarray
        River network mask (1 for river cells, 0 for non-river cells)
    marked : np.ndarray
        Matrix of grid cells that have been processed (1 for processed, 0 for unprocessed)
    
    Returns
    -------
    Dict[str, Union[int, np.ndarray]]
        A dictionary containing:
        - 'norphan': Number of orphaned branches found
        - 'queue': Queue of marked neighbors to be processed (numpy array with columns: x, y, elevation)
    
    Notes
    -----
    This function implements an orphan detection algorithm that:
    1. Identifies unprocessed river cells (masked but not marked)
    2. Checks D8 connectivity to find marked neighbors
    3. Counts marked neighbors for each orphan cell
    4. Creates a processing queue from marked neighbors of orphan cells
    
    The algorithm uses D8 connectivity (8-directional) to ensure complete
    neighbor checking and proper orphan identification.
    
    Orphan cells are typically caused by:
    - Processing order issues in river network algorithms
    - Boundary conditions affecting flow direction
    - Numerical precision issues in elevation comparisons
    - Complex river network topology
    """
    nx, ny = dem.shape
    
    # Matrix of offsets in x and y for D8 neighbors
    # Order: down, down-left, left, up-left, up, up-right, right, down-right
    kd = np.array([
        [0, -1],   # Down
        [-1, -1],  # Down-left
        [-1, 0],   # Left
        [-1, 1],   # Up-left
        [0, 1],    # Up
        [1, 1],    # Up-right
        [1, 0],    # Right
        [1, -1]    # Down-right
    ])
    
    queue = None
    
    # Look for unprocessed river cells that have a D8 neighbor on the river network
    mlist = np.where((mask == 1) & (marked == 0))[0]
    missed = np.zeros((nx, ny))
    missed.flat[mlist] = 1
    
    # Convert flat indices to 2D indices
    missed_indices = np.unravel_index(mlist, (nx, ny))
    missed_2d = np.zeros((nx, ny))
    missed_2d[missed_indices] = 1
    
    # Check and see if missed cells have D8 neighbors that were marked in the last traverse
    # Count the number of marked cells by summing the marked mask shifted for each of the D8 directions
    ncount = np.zeros((nx, ny))
    
    # Apply D8 neighbor counting with boundary checking
    ncount[1:(nx-1), 1:(ny-1)] = (
        marked[2:nx, 2:ny] +      # Down-right
        marked[0:(nx-2), 0:(ny-2)] +  # Up-left
        marked[2:nx, 0:(ny-2)] +      # Down-left
        marked[0:(nx-2), 2:ny] +      # Up-right
        marked[1:(nx-1), 2:ny] +      # Up
        marked[1:(nx-1), 0:(ny-2)] +  # Down
        marked[2:nx, 1:(ny-1)] +      # Right
        marked[0:(nx-2), 1:(ny-1)]    # Left
    )
    
    ncount = ncount * missed_2d
    norphan = np.sum(ncount > 0)  # Number of orphaned branches
    
    if norphan > 0:
        # Add the marked neighbors of any orphan cell to a new queue
        addloc = np.where(ncount > 0)  # Get the xy locations of the orphans
        
        queue_list = []
        
        for n in range(norphan):
            xn = addloc[0][n]
            yn = addloc[1][n]
            
            # For each orphan, look in all 8 directions and add any marked cell to the queue
            for k in range(8):
                xtemp = xn + kd[k, 0]
                ytemp = yn + kd[k, 1]
                
                # Check bounds before accessing array
                if (0 <= xtemp < nx and 0 <= ytemp < ny and marked[xtemp, ytemp] == 1):
                    queue_list.append([xtemp, ytemp, dem[xtemp, ytemp]])
        
        # Convert list to numpy array if we have entries
        if queue_list:
            queue = np.array(queue_list)
        else:
            queue = np.empty((0, 3))
    else:
        print("No Orphans Found")
        queue = np.empty((0, 3))
    
    output_list = {
        "norphan": int(norphan),
        "queue": queue
    }
    
    return output_list 