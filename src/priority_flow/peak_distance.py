"""
Peak Distance functions for PriorityFlow.

This module provides functions to calculate the maximum and minimum distance
from headwater cells to topographic peaks using a topological sorting approach.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def peak_dist(
    direction: np.ndarray,
    mask: Optional[np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    printflag: bool = False
) -> Dict[str, np.ndarray]:
    """
    Calculate distance to topographic peaks.
    
    Calculates the maximum and minimum distance from a headwater cell to
    topographic peaks using a topological sorting approach. This function
    identifies headwater cells and propagates distance information downstream
    following flow directions.
    
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
    Dict[str, np.ndarray]
        A dictionary containing:
        - 'mindist': Matrix containing the minimum distance from headwaters to each cell
        - 'maxdist': Matrix containing the maximum distance from headwaters to each cell
    
    Notes
    -----
    This function implements a distance calculation algorithm that:
    1. Identifies headwater cells (cells with no upstream neighbors)
    2. Uses topological sorting to process cells in correct order
    3. Propagates distance information downstream following flow directions
    4. Tracks both minimum and maximum distances from headwaters
    
    The algorithm handles:
    - Border detection and processing
    - D4 neighbor connectivity
    - Upstream neighbor counting
    - Topological sorting for correct distance propagation
    - Masked area handling
    """
    nx, ny = direction.shape
    
    if mask is None:
        mask = np.ones((nx, ny))  # Default to processing everything
    
    # Setup the border
    # TODO: we can call get_border.py here to avoid re-implementing the same logic
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
    
    # Initialize distance matrices
    mindist = np.zeros((nx, ny))
    maxdist = np.zeros((nx, ny))
    
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
    
    # Initialize a queue with all the headwater cells (i.e., cells with zero upstream neighbors)
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
            # Look downstream, add 1 to the distance and subtract 1 from the drainage count
            xtemp = queue[i, 0]
            ytemp = queue[i, 1]
            
            # If it has a flow direction
            if not np.isnan(direction[xtemp, ytemp]):
                dirtemp = np.where(np.array(d4) == direction[xtemp, ytemp])[0][0]
                xds = xtemp + kd[dirtemp, 0]
                yds = ytemp + kd[dirtemp, 1]
                
                # Add one to the distance of the downstream cell as long as that cell is in the domain
                if (xds < nx and xds >= 0 and yds < ny and yds >= 0):
                    # Update minimum distance
                    if mindist[xds, yds] == 0:
                        mindist[xds, yds] = mindist[xtemp, ytemp] + 1
                    else:
                        mindist[xds, yds] = min(mindist[xds, yds], mindist[xtemp, ytemp] + 1)
                    
                    # Update maximum distance
                    if maxdist[xds, yds] == 0:
                        maxdist[xds, yds] = maxdist[xtemp, ytemp] + 1
                    else:
                        maxdist[xds, yds] = max(maxdist[xds, yds], maxdist[xtemp, ytemp] + 1)
                    
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
    
    # Apply mask to final results
    mindist = mindist * mask
    maxdist = maxdist * mask
    
    output_list = {
        "mindist": mindist,
        "maxdist": maxdist
    }
    
    return output_list 