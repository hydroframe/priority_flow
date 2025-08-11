"""
Stream Distance functions for PriorityFlow.

This module provides functions to find the distance to the nearest stream point
following drainage directions using a stream mask and flow direction file.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def stream_dist(
    direction: np.ndarray,
    streammask: np.ndarray,
    domainmask: Optional[np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4)
) -> Dict[str, np.ndarray]:
    """
    Find the distance to the nearest stream point following drainage directions.
    
    This function uses a stream mask and a flow direction file to determine the overland
    flow distance from any point in the domain to its nearest stream neighbor following
    the defined primary flow directions.
    
    Parameters
    ----------
    direction : np.ndarray
        Flow direction matrix
    streammask : np.ndarray
        Mask with a value of 1 for every stream cell and 0 for all non-stream cells.
        Refer to the CalcSubbasins function to derive this mask.
    domainmask : np.ndarray, optional
        Optional mask of the domain area with a value of 1 for cells inside the domain
        and 0 for cells outside the domain. If no mask is provided, it will default to
        using the entire rectangular domain.
    d4 : Tuple[int, int, int, int], optional
        D4 direction numbering scheme. Defaults to (1, 2, 3, 4) for (down, left, up, right).
    
    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing:
        - 'stream.dist': Matrix of distances in cells along the flow directions to any stream cell
        - 'stream.xind': Matrix of x indices of each cell's closest stream cell
        - 'stream.yind': Matrix of y indices of each cell's closest stream cell
    
    Notes
    -----
    This function implements a stream distance calculation algorithm that:
    1. Starts from stream cells and works upstream following flow directions
    2. Calculates overland flow distances to the nearest stream neighbor
    3. Tracks the coordinates of the closest stream cell for each location
    4. Handles custom D4 direction numbering schemes
    5. Respects domain masks and boundary conditions
    
    The algorithm handles:
    - Queue-based processing starting from stream cells
    - Flow direction following for distance calculation
    - Stream cell coordinate tracking
    - Domain boundary and mask respect
    - Custom D4 direction scheme support
    
    Note: This function will ignore cells that drain to the outside of the domain.
    A border option could be added later to handle these cases.
    """
    nx, ny = direction.shape
    
    # D4 neighbor offsets
    # Rows: down, left, up, right
    # Columns: (1) deltax, (2) deltay, (3) direction number
    ku = np.array([
        [0, 1, 1],     # Down
        [1, 0, 2],     # Left
        [0, -1, 3],    # Up
        [-1, 0, 4]     # Right
    ])
    
    # Default to processing everything if domain mask not provided
    if domainmask is None:
        domainmask = np.ones((nx, ny))
    
    # Renumber the directions to 1=down, 2=left, 3=up, 4=right if a different numbering scheme was used
    dir2 = direction.copy()
    if d4[0] != 1:
        dir2[direction == d4[0]] = 1
    if d4[1] != 2:
        dir2[direction == d4[1]] = 2
    if d4[2] != 3:
        dir2[direction == d4[2]] = 3
    if d4[3] != 4:
        dir2[direction == d4[3]] = 4
    
    # Start a queue with every cell on the stream mask
    # NOTE: This is going to ignore cells that drain to the outside of the domain
    # Could add a border option later potentially
    queue = np.where(streammask == 1)
    queue = np.column_stack((queue[0], queue[1]))  # Convert to 2D array format
    
    # distance = distance in cells along the flow directions to any stream cell
    # streamx and streamy are the x,y indices of each cell's closest stream cell respectively
    distance = np.full((nx, ny), np.nan)
    
    # Initialize distance to 0 along the stream mask
    # Initialize streamx and streamy to the stream cell indices along the stream mask
    distance[streammask == 1] = 0
    
    # Create coordinate matrices
    streamy = np.tile(np.arange(ny), (nx, 1))  # Matrix with y coordinates
    streamx = np.tile(np.arange(nx), (ny, 1)).T  # Matrix with x coordinates
    
    # Set non-stream cells to NaN
    streamx[streammask == 0] = np.nan
    streamy[streammask == 0] = np.nan
    
    active = True
    
    while active:
        if len(queue) == 0:
            break
            
        indx = queue[0, 0]
        indy = queue[0, 1]
        
        queuetemp = []
        
        # Loop over four directions, check for non-stream neighbors pointing to this cell
        for d in range(4):
            tempx = indx + ku[d, 0]
            tempy = indy + ku[d, 1]
            
            # If it's pointing to the cell, is within the mask of cells to be processed, and has domain mask == 1
            if (tempx >= 0 and tempy >= 0 and tempx < nx and tempy < ny):
                # Check if the neighbor points to the current cell (drainage direction check)
                if ((d + 1 - dir2[tempx, tempy]) == 0 and 
                    streammask[tempx, tempy] == 0 and 
                    domainmask[tempx, tempy] == 1):
                    
                    distance[tempx, tempy] = distance[indx, indy] + 1
                    streamx[tempx, tempy] = streamx[indx, indy]
                    streamy[tempx, tempy] = streamy[indx, indy]
                    queuetemp.append([tempx, tempy])
        
        # If cells were adjusted, then add to the top of the queue replacing the cell that was just done
        if len(queuetemp) > 0:
            queuetemp = np.array(queuetemp)
            queue = np.vstack([queuetemp, queue[1:]])
        else:
            # If no cells were adjusted, remove this cell from the queue
            if queue.shape[0] > 1:
                queue = queue[1:]
                # Fixing bug to keep queue formatted as a matrix if it drops down to one row
                if queue.shape[0] == 1:
                    queue = queue.reshape(1, 2)
            else:
                active = False
    
    output_list = {
        "stream.dist": distance,
        "stream.xind": streamx,
        "stream.yind": streamy
    }
    
    return output_list 