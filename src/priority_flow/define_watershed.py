"""
Define Watershed functions for PriorityFlow.

This module provides functions to define the watershed (upstream area) from a point
or set of outlet points based on the flow direction file.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def delin_watershed(
    outlets: np.ndarray,
    direction: np.ndarray,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    printflag: bool = False
) -> Dict[str, np.ndarray]:
    """
    Function to define the watershed for a point or set of outlet points based on the flow direction file.
    
    Parameters
    ----------
    outlets : np.ndarray
        x,y coordinates of the outlet points or points to mask upstream areas for.
        This should be a 2D array with a separate row for each point.
    direction : np.ndarray
        Flow direction matrix
    d4 : Tuple[int, int, int, int], optional
        Directional numbering system: down, left, top, right. Defaults to (1, 2, 3, 4)
    printflag : bool, optional
        Optional flag to print out the number of cells in the queue during iterations.
        Defaults to False
    
    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing:
        - 'watershed': Binary mask of the watershed area (1 for watershed cells, 0 otherwise)
        - 'xrange': Range of x coordinates covered by the watershed as numpy array [min_x, max_x]
        - 'yrange': Range of y coordinates covered by the watershed as numpy array [min_y, max_y]
    
    Notes
    -----
    This function implements an upstream traversal algorithm that:
    1. Starts from specified outlet points
    2. Traverses upstream following flow directions
    3. Marks all cells that contribute flow to the outlet points
    4. Returns the complete watershed boundary and extent
    
    The algorithm uses a queue-based approach to efficiently process large watersheds.
    """
    nx, ny = direction.shape
    
    # Initialize a matrix to store the mask
    marked = np.zeros((nx, ny))
    
    # D4 neighbors - ordered down, left, top, right
    kd = np.array([
        [0, -1],   # Down
        [-1, 0],   # Left
        [0, 1],    # Top
        [1, 0]     # Right
    ])
    
    # Make masks of which cells drain down, up, left, right
    down = np.zeros((nx, ny))
    up = np.zeros((nx, ny))
    left = np.zeros((nx, ny))
    right = np.zeros((nx, ny))
    
    down[direction == d4[0]] = 1
    left[direction == d4[1]] = 1
    up[direction == d4[2]] = 1
    right[direction == d4[3]] = 1
    
    # Initialize the queue with the outlet points
    # Ensure outlets is a 2D numpy array
    if outlets.ndim == 1:
        # Single point: reshape to 2D array with one row
        queue = outlets.reshape(1, -1)
    elif outlets.ndim == 2:
        # Multiple points: use as is
        queue = outlets.copy()
    else:
        raise ValueError("outlets must be a 1D or 2D numpy array")
    
    nqueue = len(queue)
    count0 = 0
    ii = 1
    
    while nqueue > 0:
        if printflag:
            print(f"lap {ii} ncell {nqueue}")
        
        queue2 = None
        
        # Loop through the queue
        for i in range(nqueue):
            xtemp = int(queue[i, 0])
            ytemp = int(queue[i, 1])
            
            # Mark the current cell as part of the watershed
            marked[xtemp, ytemp] = 1
            
            # Look for cells that drain to this cell
            for d in range(4):
                xus = xtemp - kd[d, 0]
                yus = ytemp - kd[d, 1]
                
                # Check if the upstream cell is within domain bounds
                if (xus * yus > 0 and xus < nx and yus < ny):
                    # Check if the direction is valid and the cell hasn't been marked
                    if (not np.isnan(direction[xus, yus]) and marked[xus, yus] == 0):
                        if direction[xus, yus] == d4[d]:
                            # Mark the upstream cell as part of the watershed
                            marked[xus, yus] = 1
                            
                            # Add the upstream cell to the queue for further processing
                            if queue2 is None:
                                queue2 = np.array([[xus, yus]])
                            else:
                                queue2 = np.vstack([queue2, [xus, yus]])
        
        # Continue processing if there are more cells in the queue
        if queue2 is not None and len(queue2) >= 2:
            queue = queue2
            nqueue = len(queue)
            ii += 1
        else:
            nqueue = 0
    
    # Get the coordinates of all marked cells
    masklist = np.where(marked == 1)
    
    # Calculate the range of coordinates covered by the watershed
    if len(masklist[0]) > 0:
        xrange = np.array([np.min(masklist[0]), np.max(masklist[0])])
        yrange = np.array([np.min(masklist[1]), np.max(masklist[1])])
    else:
        # If no watershed was found, return empty arrays
        xrange = np.array([])
        yrange = np.array([])
    
    output_list = {
        "watershed": marked,
        "xrange": xrange,
        "yrange": yrange
    }
    
    return output_list