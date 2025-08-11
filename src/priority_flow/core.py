"""
Core PriorityFlow algorithms for DEM processing.

This module contains the main algorithms for topographic processing:
- init_queue: Initialize processing queue and matrices
- d4_traverse_b: Main priority flood algorithm for DEM processing
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
import warnings


def init_queue(
    dem: np.ndarray,
    init_mask: Optional[np.ndarray] = None,
    domain_mask: Optional[np.ndarray] = None,
    border: Optional[np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4)
) -> Dict[str, np.ndarray]:
    """
    Initialize queue and matrices for topographic processing.
    
    Sets up a priority queue and initializes marked and step matrices for DEM processing.
    This function identifies the outlet cells that will serve as drainage targets.
    
    Parameters
    ----------
    dem : np.ndarray
        Digital Elevation Model matrix [nx, ny]
    init_mask : np.ndarray, optional
        Mask denoting subset of cells to consider for the queue (e.g., river cells).
        If None, every border cell will be added to the queue.
    domain_mask : np.ndarray, optional
        Mask of the domain extent to be considered.
        If None, boundaries will be calculated from the rectangular extent.
    border : np.ndarray, optional
        Custom border definition. If provided, this will be used instead of
        calculating from domain_mask.
    d4 : tuple, optional
        Direction numbering system for down, left, top, right.
        Defaults to (1, 2, 3, 4).
    
    Returns
    -------
    dict
        Dictionary containing:
        - marked: Matrix indicating outlet cells (1=outlet, 0=not outlet)
        - queue: Array of outlet cells with columns [x, y, elevation]
        - init_mask: Matrix indicating cells input as potential output points
        - basins: Matrix indicating basin number for each outlet point
        - direction: Matrix indicating flow direction for each outlet point
    
    Notes
    -----
    This is a Python port of the R function InitQueue from PriorityFlow.
    The algorithm ensures D4 connectivity and identifies proper drainage targets.
    """
    
    # Get dimensions
    nx, ny = dem.shape
    
    # Initialize matrices
    queue = None
    marked = np.zeros((nx, ny), dtype=int)
    step = np.zeros((nx, ny), dtype=int)
    
    # Setup flow directions - D4 neighbors
    # kd[:, 0] = x offsets, kd[:, 1] = y offsets, kd[:, 2] = direction values
    kd = np.zeros((4, 3), dtype=int)
    kd[:, 0] = [0, -1, 0, 1]      # x offsets: down, left, top, right
    kd[:, 1] = [-1, 0, 1, 0]      # y offsets: down, left, top, right  
    kd[:, 2] = [d4[0], d4[1], d4[2], d4[3]]  # direction values
    
    # Handle missing init_mask
    if init_mask is None:
        print("No init mask provided, all border cells will be added to queue")
        init_mask = np.ones((nx, ny), dtype=int)
    
    # Handle missing domain_mask
    if domain_mask is None:
        print("No domain mask provided, using entire domain")
        domain_mask = np.ones((nx, ny), dtype=int)
    
    # Setup the border
    if border is None:
        print("No border provided, setting border using domain mask")
        border = np.ones((nx, ny), dtype=int)
        
        # Calculate border cells (cells with < 4 neighbors)
        # Note: Python is 0-indexed, so we adjust the indexing
        border[1:nx-1, 1:ny-1] = (
            domain_mask[0:nx-2, 1:ny-1] +      # left neighbor
            domain_mask[2:nx, 1:ny-1] +         # right neighbor
            domain_mask[1:nx-1, 0:ny-2] +      # bottom neighbor
            domain_mask[1:nx-1, 2:ny]           # top neighbor
        )
        
        # Mark cells with < 4 neighbors as border cells
        border = border * domain_mask
        border[(border < 4) & (border != 0)] = 1
        border[border == 4] = 0
    
    # Create basin matrix and identify outlet cells
    basins = np.zeros((nx, ny), dtype=int)
    mask_bound = init_mask * border
    blist = np.where(mask_bound > 0)[0]  # 1D indices
    binlist = np.where(mask_bound > 0)   # 2D indices (x, y)
    
    # Create queue from outlet cells
    if len(blist) > 0:
        # Convert to column format: [x, y, elevation]
        queue = np.column_stack([
            binlist[0],      # x coordinates
            binlist[1],      # y coordinates  
            dem[binlist[0], binlist[1]]  # elevations
        ])
        
        # Mark outlet cells
        marked[binlist[0], binlist[1]] = 1
        
        # Assign basin numbers
        basins[binlist[0], binlist[1]] = np.arange(1, len(blist) + 1)
    
    # Assign flow directions pointing out of domain
    direction = np.full((nx, ny), np.nan)
    
    if queue is not None and len(queue) > 0:
        for i in range(len(queue)):
            xtemp = queue[i, 0]
            ytemp = queue[i, 1]
            temp = np.zeros(4)
            
            # Check each D4 direction
            for d in range(4):
                xtest = xtemp + kd[d, 0]
                ytest = ytemp + kd[d, 1]
                
                # Check if neighbor is outside domain
                if (xtest < 0 or xtest >= nx or ytest < 0 or ytest >= ny):
                    temp[d] = 0
                else:
                    temp[d] = domain_mask[xtest, ytest]
            
            # Find direction pointing outside domain (temp[d] == 0)
            outside_dirs = np.where(temp == 0)[0]
            if len(outside_dirs) > 0:
                # Use first available outside direction
                direction[xtemp, ytemp] = kd[outside_dirs[0], 2]
    
    return {
        'marked': marked,
        'queue': queue,
        'init_mask': init_mask,
        'basins': basins,
        'direction': direction
    }


def d4_traverse_b(
    dem: np.ndarray,
    queue: np.ndarray,
    marked: np.ndarray,
    mask: Optional[np.ndarray] = None,
    step: Optional[np.ndarray] = None,
    direction: Optional[np.ndarray] = None,
    basins: Optional[np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    print_step: bool = False,
    n_chunk: int = 100,
    epsilon: float = 0.0,
    print_flag: bool = False
) -> Dict[str, np.ndarray]:
    """
    Priority flow processing of D4 stream networks.
    
    This is the main algorithm that processes the DEM using a priority flood approach.
    It ensures that every cell has a valid drainage pathway to an outlet.
    
    Parameters
    ----------
    dem : np.ndarray
        Elevation matrix [nx, ny]
    queue : np.ndarray
        Priority queue with columns [x, y, elevation]
    marked : np.ndarray
        Matrix indicating which cells have been processed
    mask : np.ndarray, optional
        Mask with 1s for cells to be processed and 0s for everything else.
        Defaults to processing everything.
    step : np.ndarray, optional
        Matrix of step numbers for processed cells. Defaults to all zeros.
    direction : np.ndarray, optional
        Matrix of flow directions for processed cells. Defaults to all NaN.
    basins : np.ndarray, optional
        Matrix of basin numbers. If provided, every cell will be assigned
        the same basin as the cell that adds it.
    d4 : tuple, optional
        Direction numbering system. Defaults to (1, 2, 3, 4).
    print_step : bool, optional
        If True, print step number and queue size. Defaults to False.
    n_chunk : int, optional
        Parameter for queue splitting. The top 'n_chunk' values will be put
        into the primary queue initially. Defaults to 100.
    epsilon : float, optional
        Amount to add to filled areas to avoid creating flats. Defaults to 0.0.
    print_flag : bool, optional
        Optional flag to print function progress. Defaults to False.
    
    Returns
    -------
    dict
        Dictionary containing:
        - dem: Processed DEM with elevations adjusted for drainage
        - direction: Flow direction matrix
        - step: Step number matrix
        - basins: Basin assignment matrix
        - marked: Updated marked matrix
    
    Notes
    -----
    This is a Python port of the R function D4TraverseB from PriorityFlow.
    The algorithm implements a modified priority flood approach for depression filling.
    """
    
    if print_flag:
        print("Starting D4 traverse processing...")
    
    # Get dimensions
    nx, ny = dem.shape
    dem_new = dem.copy()
    
    # Setup matrices for anything that wasn't input
    if mask is None:
        mask = np.ones((nx, ny), dtype=int)
    if step is None:
        step = np.zeros((nx, ny), dtype=int)
    if direction is None:
        direction = np.full((nx, ny), np.nan)
    if basins is None:
        basins = np.zeros((nx, ny), dtype=int)
    
    # D4 neighbors - walking upstream so directions point opposite
    kd = np.zeros((4, 3), dtype=int)
    kd[:, 0] = [0, -1, 0, 1]      # x offsets: down, left, top, right
    kd[:, 1] = [-1, 0, 1, 0]      # y offsets: down, left, top, right
    kd[:, 2] = [d4[2], d4[3], d4[0], d4[1]]  # Opposite directions for upstream
    
    # Split queue if needed
    n_queue = len(queue) if queue is not None else 0
    
    if n_queue > n_chunk:
        # Sort by elevation and split
        sorted_indices = np.argsort(queue[:, 2])
        queue1 = queue[sorted_indices[:n_chunk]]
        queue2 = queue[sorted_indices[n_chunk:]]
        th = queue2[0, 2] if len(queue2) > 0 else queue1[-1, 2] * 1.1
        n_queue2 = len(queue2)
        n_queue = len(queue1)
        
        if print_step:
            print(f"Initial queue: {len(queue)}, splitting. Q1={n_queue}, Q2={n_queue2}")
    else:
        if print_step:
            print(f"Initial queue: {n_queue}, not splitting")
        queue1 = queue.copy() if queue is not None else None
        queue2 = None
        n_queue2 = 0
        th = queue1[-1, 2] * 1.1 if queue1 is not None else 0
    
    # Main processing loop
    step_count = 0
    
    while n_queue > 0 and queue1 is not None:
        # Pick the lowest DEM cell on the queue
        pick = np.argmin(queue1[:, 2])
        xc = int(queue1[pick, 0])
        yc = int(queue1[pick, 1])
        dem_c = queue1[pick, 2]
        
        if print_flag and step_count % 1000 == 0:
            print(f"Step {step_count}: Processing cell ({xc}, {yc}) at elevation {dem_c}")
        
        # Look for D4 neighbor cells that are on the mask and add to queue
        count = 0
        
        # Check if the original cell is on the border
        bdr_chk = direction[xc, yc]
        
        for k in range(4):
            xk = xc + kd[k, 0]
            yk = yc + kd[k, 1]
            
            # Check bounds
            if 0 <= xk < nx and 0 <= yk < ny:
                # Check if neighbor is on mask and not marked
                if mask[xk, yk] == 1 and marked[xk, yk] == 0:
                    # Calculate new elevation for neighbor
                    new_elev = max(dem_c + epsilon, dem[xk, yk])
                    
                    # Update DEM
                    dem_new[xk, yk] = new_elev
                    
                    # Mark cell and assign direction
                    marked[xk, yk] = 1
                    direction[xk, yk] = kd[k, 2]
                    
                    # Assign basin if provided
                    if basins is not None:
                        basins[xk, yk] = basins[xc, yc]
                    
                    # Add to queue
                    new_entry = np.array([[xk, yk, new_elev]])
                    if queue1 is None:
                        queue1 = new_entry
                    else:
                        queue1 = np.vstack([queue1, new_entry])
                    
                    count += 1
        
        # Remove processed cell from queue
        queue1 = np.delete(queue1, pick, axis=0)
        n_queue = len(queue1)
        
        # Check if we need to refill from secondary queue
        if n_queue == 0 and queue2 is not None and len(queue2) > 0:
            # Move some cells from secondary to primary queue
            n_to_move = min(n_chunk, len(queue2))
            queue1 = queue2[:n_to_move]
            queue2 = queue2[n_to_move:]
            n_queue = len(queue1)
            
            if print_step:
                print(f"Refilled primary queue from secondary: {n_queue} cells")
        
        step_count += 1
    
    if print_flag:
        print(f"Processing complete after {step_count} steps")
    
    return {
        'dem': dem_new,
        'direction': direction,
        'step': step,
        'basins': basins,
        'marked': marked
    } 