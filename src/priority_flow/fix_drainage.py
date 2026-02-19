"""
Fix Drainage functions for PriorityFlow.

This module provides functions to walk upstream from a point ensuring the DEM
is increasing by a minimum epsilon. This is useful for fixing drainage issues
and ensuring proper elevation gradients in river networks.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def fix_drainage(
    dem: np.ndarray,
    direction: np.ndarray,
    mask: np.ndarray,
    bank_epsilon: float,
    startpoint: Union[List[int], Tuple[int, int], np.ndarray],
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4)
) -> Dict[str, np.ndarray]:
    """
    Walk upstream from a point ensuring DEM is increasing by a minimum epsilon.
    
    This function walks upstream from a point following a flow direction file and
    checks that the elevation upstream is greater than or equal to the elevation
    at the point + epsilon. Once the function reaches a point where upstream cells
    pass the test, it stops.
    
    NOTE: This is only processing the immediate neighborhood and does not recurse
    over the entire domain. For example, if you did the entire overall priority
    flow processing with an epsilon of 0, then ran this function starting at a
    river point with an epsilon of 0.1, this function will traverse upstream from
    the river bottom checking that every cell connecting to the river cell is
    higher by at least this amount. Once it reaches a point where every connected
    cell passes this test, it will stop. Therefore, there could still be locations
    higher up on the hillslope with the original epsilon of zero. This is on
    purpose and this script is not intended to globally ensure a given epsilon.
    
    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model matrix
    direction : np.ndarray
        Flow direction matrix
    mask : np.ndarray
        Processing mask (1 for cells to be processed, 0 for cells to be ignored)
    bank_epsilon : float
        Minimum elevation difference required between upstream and downstream cells
    startpoint : Union[List[int], Tuple[int, int], np.ndarray]
        The x,y index of a grid cell to start walking upstream from
    d4 : Tuple[int, int, int, int], optional
        Directional numbering system: down, left, top, right. Defaults to (1, 2, 3, 4)
    
    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing:
        - 'dem.adj': Adjusted DEM matrix with corrected elevations (matches R FixDrainage)
        - 'processed': Matrix marking which cells were processed/adjusted
    
    Notes
    -----
    This function implements an upstream walking algorithm that:
    1. Starts from a specified grid cell
    2. Walks upstream following flow directions
    3. Checks elevation differences between connected cells
    4. Adjusts upstream cell elevations to meet minimum epsilon requirement
    5. Continues until no more adjustments are needed
    
    The algorithm handles:
    - D4 connectivity (4-directional upstream walking)
    - Custom D4 numbering schemes
    - Minimum elevation difference enforcement
    - Queue-based processing for efficient traversal
    - Boundary condition handling
    """
    nx, ny = direction.shape
    
    # D4 neighbors
    # Rows: down, left, top, right
    # Columns: (1) deltax, (2) deltay, direction number if you are walking upstream
    ku = np.array([
        [0, 1, 1],    # Down
        [1, 0, 2],    # Left
        [0, -1, 3],   # Top
        [-1, 0, 4]    # Right
    ])
    
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
    
    # Initialize
    marked = np.zeros((nx, ny))
    queue = np.array([[startpoint[0], startpoint[1]]])
    active = True
    dem2 = dem.copy()
    
    while active:
        indx = int(queue[0, 0])
        indy = int(queue[0, 1])
        
        queuetemp = None
        
        # Loop over four directions, check for non-stream neighbors pointing to this cell
        for d in range(4):
            tempx = indx + ku[d, 0]
            tempy = indy + ku[d, 1]
            
            # R: if(tempx*tempy>0 & tempx<nx & tempy<ny) - 1-based valid 1..nx-1, 1..ny-1
            # Python 0-based: valid 0..nx-1, 0..ny-1
            if tempx >= 0 and tempy >= 0 and tempx < nx and tempy < ny:
                if ((d + 1 - dir2[tempx, tempy]) == 0 and mask[tempx, tempy] == 1):
                    if (dem2[tempx, tempy] - dem2[indx, indy]) < bank_epsilon:
                        dem2[tempx, tempy] = dem2[indx, indy] + bank_epsilon
                        marked[tempx, tempy] = 1
                        
                        # Add to temporary queue
                        if queuetemp is None:
                            queuetemp = np.array([[tempx, tempy]])
                        else:
                            queuetemp = np.vstack([np.array([[tempx, tempy]]), queuetemp])
        
        # If cells were adjusted, add to the top of the queue replacing the cell that was just done
        if queuetemp is not None and queuetemp.size > 0:
            queue = np.vstack([queuetemp, queue[1:]])
        else:
            # If no cells were adjusted, remove this cell from the queue
            if queue.shape[0] > 1:
                queue = queue[1:]
                # Fix bug to keep queue formatted as a matrix if it drops down to one row
                if queue.size == 2:
                    queue = queue.reshape(1, 2)
            else:
                active = False
    
    output_list = {
        "dem_adj": dem2,
        "processed": marked
    }
    
    return output_list 