"""
Init Queue functions for PriorityFlow.

This module provides functions to initialize a queue and initialize marked and step
matrices for DEM processing. It sets up the initial processing queue for priority
flood algorithms and identifies outlet cells.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def init_queue(
    dem: np.ndarray,
    initmask: Optional[np.ndarray] = None,
    domainmask: Optional[np.ndarray] = None,
    border: Optional[np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4)
) -> Dict[str, np.ndarray]:
    """
    Initialize queue for topographic processing.
    
    Sets up a queue and initializes marked and step matrices for DEM processing.
    This function identifies outlet cells and prepares the initial processing queue
    for priority flood algorithms.
    
    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model matrix
    initmask : np.ndarray, optional
        Mask of the same dimensions as dem denoting a subset of cells to be considered
        for the queue (e.g., if you want to setup a run starting with only river cells).
        If no init mask is included, every border cell will be added to the queue.
    domainmask : np.ndarray, optional
        Mask of the domain extent to be considered. If no domain mask is provided,
        boundaries will be calculated from the rectangular extent.
    border : np.ndarray, optional
        Alternatively you can input your own border rather than having it be calculated
        from the domain mask. For example, if you want to have the river network and
        the borders combined, you can input this as a border.
    d4 : Tuple[int, int, int, int], optional
        Directional numbering system: down, left, top, right. Defaults to (1, 2, 3, 4)
    
    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing:
        - 'mask': Matrix indicating the cells that were input as potential output points
        - 'queue': List of the outlet cells with three columns: x, y, elevation
        - 'marked': Matrix indicating the outlet cells that were identified (1=outlet, 0=not outlet)
        - 'basins': Matrix indicating the basin number for each outlet point (each outlet is assigned a unique basin number)
        - 'direction': Matrix indicating the flow direction for each outlet point (follows the d4 numbering scheme)
    
    Notes
    -----
    This function implements a queue initialization algorithm that:
    1. Sets up processing matrices (marked, step, basin, direction)
    2. Identifies border cells or uses provided border
    3. Creates initial processing queue from outlet cells
    4. Assigns flow directions pointing out of the domain
    5. Prepares basin numbering for outlet cells
    
    The algorithm handles:
    - Automatic border detection from domain mask
    - Custom border input for specialized processing
    - Outlet cell identification and queue setup
    - Flow direction assignment for domain boundaries
    - Basin numbering for multiple outlets
    """
    # Initialize queue and matrices
    nx, ny = dem.shape
    queue = None
    marked = np.zeros((nx, ny))
    step = np.zeros((nx, ny))
    
    # Setup flow directions
    # D4 neighbors
    kd = np.array([
        [0, -1, d4[0]],    # Down
        [-1, 0, d4[1]],    # Left
        [0, 1, d4[2]],     # Top
        [1, 0, d4[3]]      # Right
    ])
    
    if initmask is None:
        print("No init mask provided, all border cells will be added to queue")
        initmask = np.ones((nx, ny))
    
    if domainmask is None:
        print("No domain mask provided, using entire domain")
        domainmask = np.ones((nx, ny))
    
    # Setup the border
    # TODO: we can call get_border.py here to avoid re-implementing the same logic
    if border is None:
        print("No border provided, setting border using domain mask")
        border = np.ones((nx, ny))
        border[1:(nx-1), 1:(ny-1)] = (
            domainmask[0:(nx-2), 1:(ny-1)] +
            domainmask[2:nx, 1:(ny-1)] +
            domainmask[1:(nx-1), 0:(ny-2)] +
            domainmask[1:(nx-1), 2:ny]
        )
        border = border * domainmask
        border[(border < 4) & (border != 0)] = 1
        border[border == 4] = 0
    
    # Initialize basin matrix and identify outlet cells
    basin = np.zeros((nx, ny))
    maskbound = initmask * border
    blist = np.where(maskbound > 0)[0]  # Array indices
    binlist = np.unravel_index(blist, (nx, ny))  # xy indices
    
    # Create queue with x, y coordinates and elevations
    if len(blist) > 0:
        queue = np.column_stack((binlist[0], binlist[1], dem.flat[blist]))
    else:
        queue = np.empty((0, 3))
    
    # Mark outlet cells and assign basin numbers
    marked.flat[blist] = 1
    basin.flat[blist] = np.arange(1, len(blist) + 1)
    
    # Assign flow direction to point out of the domain
    direction = np.full((nx, ny), np.nan)
    
    for i in range(queue.shape[0]):
        xtemp = int(queue[i, 0])
        ytemp = int(queue[i, 1])
        temp = np.zeros(4)
        
        for d in range(4):
            xtest = xtemp + kd[d, 0]
            ytest = ytemp + kd[d, 1]
            
            # If temp2 falls outside the domain, give it a value of 0
            if (xtest * ytest == 0 or xtest >= nx or ytest >= ny):
                temp[d] = 0
            else:
                # Give it a value of the mask
                temp[d] = domainmask[xtest, ytest]
        
        # Find direction with minimum mask value (prefer outside domain)
        dtemp = np.argmin(temp)
        direction[xtemp, ytemp] = kd[dtemp, 2]
    
    output_list = {
        "mask": initmask,
        "queue": queue,
        "marked": marked,
        "basins": basin,
        "direction": direction
    }
    
    return output_list 