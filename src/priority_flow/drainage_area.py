"""
Drainage Area functions for PriorityFlow.

This module provides functions to calculate the drainage area (number of cells
draining to any cell) given a flow direction file using a topological sorting approach.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def drainage_area(
    direction: np.ndarray,
    mask: Optional[np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    printflag: bool = False
) -> np.ndarray:
    """
    Calculate drainage area for each cell.
    
    Calculates the number of cells draining to any cell given a flow direction file.
    Uses a topological sorting approach with queue management to process cells
    in the correct order (headwaters first, then downstream cells).
    
    Parameters
    ----------
    direction : np.ndarray
        Flow direction matrix
    mask : np.ndarray, optional
        Processing mask. Defaults to processing everything
    d4 : Tuple[int, int, int, int], optional
        Directional numbering system: down, left, top, right. Defaults to (1, 2, 3, 4)
    printflag : bool, optional
        Whether to print progress information. Defaults to False
    
    Returns
    -------
    np.ndarray
        Matrix containing the drainage area (number of upstream cells) for each cell
    
    Notes
    -----
    This function implements a topological sorting algorithm for drainage area calculation:
    1. Identifies headwater cells (cells with no upstream neighbors)
    2. Processes cells in topological order (upstream to downstream)
    3. Accumulates drainage area by adding upstream areas to downstream cells
    4. Uses queue management to ensure correct processing order
    
    The algorithm handles:
    - Border detection and processing
    - D4 neighbor connectivity
    - Upstream neighbor counting
    - Topological sorting for correct area accumulation
    - Masked area handling
    """
    nx, ny = direction.shape
    
    if mask is None:
        mask = np.ones((nx, ny))  # Default to processing everything
    
    # Setup the border
    border = np.ones((nx, ny))
    border[1:(nx-1), 1:(ny-1)] = (
        mask[0:(nx-2), 1:(ny-1)] +
        mask[2:nx, 1:(ny-1)] +
        mask[1:(nx-1), 0:(ny-2)] +
        mask[1:(nx-1), 2:ny]
    )
    border = border * mask
    border[(border < 4) & (border != 0)] = 1
    border[border == 4] = 0
    
    # Initialize drainage area matrix
    drainarea = np.ones((nx, ny))
    
    # D4 neighbors
    # Ordered: down, left, top, right
    kd = np.array([
        [0, -1],   # Down
        [-1, 0],   # Left
        [0, 1],    # Top
        [1, 0]     # Right
    ])
    
    # Make masks of which cells drain down, up, left, right
    down = np.zeros((nx, ny))
    left = np.zeros((nx, ny))
    up = np.zeros((nx, ny))
    right = np.zeros((nx, ny))
    
    down[direction == d4[0]] = 1
    left[direction == d4[1]] = 1
    up[direction == d4[2]] = 1
    right[direction == d4[3]] = 1
    
    # Calculate the number of cells draining to any cell
    draincount = np.zeros((nx, ny))
    draincount[:, 0:(ny-1)] = draincount[:, 0:(ny-1)] + down[:, 1:ny]
    draincount[:, 1:ny] = draincount[:, 1:ny] + up[:, 0:(ny-1)]
    draincount[0:(nx-1), :] = draincount[0:(nx-1), :] + left[1:nx, :]
    draincount[1:nx, :] = draincount[1:nx, :] + right[0:(nx-1), :]
    
    # Give values outside the mask and on the border a negative count so they aren't processed
    draincount[mask == 0] = -99
    
    # Initialize a queue with all the headwater cells (i.e. cells with zero upstream neighbors)
    draintemp = draincount.copy()
    queue_indices = np.where(draintemp == 0)
    queue = np.column_stack((queue_indices[0], queue_indices[1]))
    qlist = np.where(draintemp == 0)[0]
    
    # Create blist for tracking cells with upstream neighbors
    blist_indices = np.where(draintemp > 0)
    blist = np.column_stack((blist_indices[0], blist_indices[0], blist_indices[1]))
    
    nqueue = queue.shape[0]
    
    ii = 1
    
    while nqueue > 0:
        if printflag:
            print(f"lap {ii} ncell {nqueue}")
        
        # Loop through the queue
        for i in range(nqueue):
            # Look downstream, add 1 to the area and subtract 1 from the drainage count
            xtemp = queue[i, 0]
            ytemp = queue[i, 1]
            
            # If it has a flow direction
            if not np.isnan(direction[xtemp, ytemp]):
                dirtemp = np.where(np.array(d4) == direction[xtemp, ytemp])[0][0]
                xds = xtemp + kd[dirtemp, 0]
                yds = ytemp + kd[dirtemp, 1]
                
                # Add one to the area of the downstream cell as long as that cell is in the domain
                if (xds < nx and xds >= 0 and yds < ny and yds >= 0):
                    drainarea[xds, yds] = drainarea[xds, yds] + drainarea[xtemp, ytemp]
                    
                    # Subtract one from the number of upstream cells from the downstream cell
                    draintemp[xds, yds] = draintemp[xds, yds] - 1
            
            # Set the drain temp to -99 for current cell to indicate it's been done
            draintemp[xtemp, ytemp] = -99
        
        # Make a new queue with the cells with zero upstream drains left
        if blist.size > 0:
            ilist = np.where(draintemp[blist[:, 0]] == 0)[0]
            if ilist.size > 0:
                queue = blist[ilist, 1:3]
                blist = np.delete(blist, ilist, axis=0)
            else:
                queue = np.empty((0, 2))
        else:
            queue = np.empty((0, 2))
            if printflag:
                print("blist empty")
        
        nqueue = queue.shape[0]
        
        # Handle single cell cases
        if nqueue == 1:
            queue = queue.reshape(1, 2)
        if blist.size == 3:
            blist = blist.reshape(1, 3)
        
        ii += 1
    
    # Apply mask to final result
    drainarea = drainarea * mask
    
    return drainarea


def calculate_drainage_area_alt(
    direction: np.ndarray,
    mask: Optional[np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    printflag: bool = False
) -> np.ndarray:
    """
    Alternative implementation of drainage area calculation.
    
    This is a more Pythonic implementation that may be more efficient
    for large datasets. It uses vectorized operations where possible.
    
    Parameters
    ----------
    direction : np.ndarray
        Flow direction matrix
    mask : np.ndarray, optional
        Processing mask. Defaults to processing everything
    d4 : Tuple[int, int, int, int], optional
        Directional numbering system: down, left, top, right. Defaults to (1, 2, 3, 4)
    printflag : bool, optional
        Whether to print progress information. Defaults to False
    
    Returns
    -------
    np.ndarray
        Matrix containing the drainage area for each cell
    """
    nx, ny = direction.shape
    
    if mask is None:
        mask = np.ones((nx, ny))
    
    # Initialize drainage area
    drainarea = np.ones((nx, ny))
    
    # D4 neighbor offsets
    kd = np.array([
        [0, -1],   # Down
        [-1, 0],   # Left
        [0, 1],    # Top
        [1, 0]     # Right
    ])
    
    # Create direction masks
    dir_masks = {}
    for i, d in enumerate(d4):
        dir_masks[d] = (direction == d).astype(int)
    
    # Calculate upstream neighbor count
    draincount = np.zeros((nx, ny))
    
    # Vectorized neighbor counting
    for i, d in enumerate(d4):
        mask_d = dir_masks[d]
        dx, dy = kd[i]
        
        # Shift mask and add to count
        if dx == 0:  # Vertical shift
            if dy > 0:  # Upward shift
                draincount[:, dy:] += mask_d[:, :-dy]
            else:  # Downward shift
                draincount[:, :dy] += mask_d[:, -dy:]
        else:  # Horizontal shift
            if dx > 0:  # Rightward shift
                draincount[dx:, :] += mask_d[:-dx, :]
            else:  # Leftward shift
                draincount[:dx, :] += mask_d[-dx:, :]
    
    # Mark masked areas
    draincount[mask == 0] = -99
    
    # Topological sorting
    processed = np.zeros((nx, ny), dtype=bool)
    drainarea_final = np.ones((nx, ny))
    
    while True:
        # Find headwater cells (cells with no upstream neighbors)
        headwaters = (draincount == 0) & (mask == 1) & ~processed
        
        if not np.any(headwaters):
            break
        
        if printflag:
            n_headwaters = np.sum(headwaters)
            print(f"Processing {n_headwaters} headwater cells")
        
        # Process headwater cells
        for x, y in zip(*np.where(headwaters)):
            if not processed[x, y]:
                # Add current cell's area to downstream cell
                if not np.isnan(direction[x, y]):
                    dir_idx = np.where(np.array(d4) == direction[x, y])[0][0]
                    dx, dy = kd[dir_idx]
                    xds, yds = x + dx, y + dy
                    
                    if (0 <= xds < nx and 0 <= yds < ny and mask[xds, yds] == 1):
                        drainarea_final[xds, yds] += drainarea_final[x, y]
                        draincount[xds, yds] -= 1
                
                processed[x, y] = True
                draincount[x, y] = -99
    
    # Apply mask
    drainarea_final = drainarea_final * mask
    
    return drainarea_final 