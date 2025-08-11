"""
Defining Subbasins functions for PriorityFlow.

This module provides functions to divide the domain into subbasins with individual
stream segments based on flow direction and drainage area thresholds.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def calc_subbasins(
    direction: np.ndarray,
    area: np.ndarray,
    mask: Optional[np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    riv_th: float = 50.0,
    printflag: bool = False,
    merge_th: float = 0.0
) -> Dict[str, Union[np.ndarray, np.ndarray]]:
    """
    Function to divide the domain into subbasins with individual stream segments.
    
    Parameters
    ----------
    direction : np.ndarray
        Flow direction matrix
    area : np.ndarray
        Drainage areas for every cell
    mask : np.ndarray, optional
        Processing mask. Defaults to processing everything
    d4 : Tuple[int, int, int, int], optional
        Directional numbering system: down, left, top, right. Defaults to (1, 2, 3, 4)
    riv_th : float, optional
        Threshold for the drainage area minimum used to designate cells as river cells.
        Defaults to 50
    printflag : bool, optional
        Flag to print function progress. Defaults to False
    merge_th : float, optional
        After all the subbasins have been defined, subbasins with areas < merge_th will be
        combined with their downstream neighbors. Defaults to 0 (no merging)
    
    Returns
    -------
    Dict[str, Union[np.ndarray, np.ndarray]]
        A dictionary containing:
        - 'segments': Stream segment matrix
        - 'subbasins': Subbasin drainage area matrix
        - 'RiverMask': River mask based on drainage area threshold
        - 'summary': Summary table with subbasin information
    
    Notes
    -----
    This function implements a three-step process:
    1. Walk down from every headwater marking stream segments
    2. Get the drainage basins for every segment
    3. Optionally merge small basins with downstream neighbors
    
    WARNING: Non-zero merge thresholds are not compatible with the RiverSmooth function.
    """
    nx, ny = direction.shape
    
    if mask is None:
        mask = np.ones((nx, ny))  # Default to processing everything
    
    # Setup the border
    border = np.ones((nx, ny))
    border[1:nx-1, 1:ny-1] = (mask[0:nx-2, 1:ny-1] + 
                               mask[2:nx, 1:ny-1] + 
                               mask[1:nx-1, 0:ny-2] + 
                               mask[1:nx-1, 2:ny])
    border = border * mask
    border[(border < 4) & (border != 0)] = 1
    border[border == 4] = 0
    
    # Initialize drainage area matrix
    subbasin = np.zeros((nx, ny))
    marked = np.zeros((nx, ny))
    
    # D4 neighbors - ordered down, left, top, right
    kd = np.array([
        [0, -1],   # Down
        [-1, 0],   # Left
        [0, 1],    # Top
        [1, 0]     # Right
    ])
    
    # Make a river mask based on the drainage area threshold
    rivers = area.copy()
    rivers[area < riv_th] = 0
    rivers[area >= riv_th] = 1
    
    if np.sum(rivers) == 0:
        print("Area Threshold too high. No river cells found. Please select a lower riv_th value")
        return {}
    
    # Make masks of which cells drain down, up, left, right
    down = np.zeros((nx, ny))
    up = np.zeros((nx, ny))
    left = np.zeros((nx, ny))
    right = np.zeros((nx, ny))
    
    down[direction == d4[0]] = 1
    left[direction == d4[1]] = 1
    up[direction == d4[2]] = 1
    right[direction == d4[3]] = 1
    
    # Calculate the number of river cells draining to any cell
    draincount = np.zeros((nx, ny))
    draincount[:, 0:ny-1] += down[:, 1:ny] * rivers[:, 1:ny]
    draincount[:, 1:ny] += up[:, 0:ny-1] * rivers[:, 0:ny-1]
    draincount[0:nx-1, :] += left[1:nx, :] * rivers[1:nx, :]
    draincount[1:nx, :] += right[0:nx-1, :] * rivers[0:nx-1, :]
    
    # Identify all the headwater cells
    headwater = np.zeros((nx, ny))
    headwater[(draincount == 0) & (rivers == 1)] = 1
    
    # Give values outside the mask and on the border a negative count so they aren't processed
    marked[mask == 0] = 1
    
    # Start with all the headwater cells (i.e. cells with zero upstream neighbors)
    headwater_indices = np.where(headwater == 1)
    blist = np.column_stack([headwater_indices[0], headwater_indices[1]])
    nheadwater = len(blist)
    
    ends = rivers.copy()
    ends[headwater_indices] = 2
    
    # Get just the river areas to use for this
    rivarea = area * rivers
    
    index = 0
    subbasin = np.zeros((nx, ny))
    marked = np.zeros((nx, ny))
    first = True
    
    ### 1. Walk down from every headwater marking stream segments
    for i in range(nheadwater):
        xtemp = int(blist[i, 0])
        ytemp = int(blist[i, 1])
        active = True
        index += 1
        subbasin[xtemp, ytemp] = index
        marked[xtemp, ytemp] = 1
        
        summarytemp = [index, xtemp, ytemp, 0, 0, 0, 0]
        
        while active:
            # Get the direction and find downstream cell
            dirtemp = np.where(np.array(d4) == direction[xtemp, ytemp])[0][0]
            xds = xtemp + kd[dirtemp, 0]
            yds = ytemp + kd[dirtemp, 1]
            
            # If the downstream neighbor hasn't already been processed and it's in the domain
            if (xds * yds > 0 and xds < nx and yds < ny):
                if marked[xds, yds] == 0 and mask[xds, yds] == 1:
                    # Check the area difference
                    accum = area[xds, yds] - area[xtemp, ytemp]
                    
                    # If there is a tributary coming in then start a new segment
                    if accum > riv_th:
                        summarytemp[3:5] = [xtemp, ytemp]
                        summarytemp[5] = index + 1
                        index += 1
                        ends[xtemp, ytemp] = 3
                        ends[xds, yds] = 2
                        
                        if first:
                            summary = np.array([summarytemp])
                            first = False
                        else:
                            summary = np.vstack([summary, summarytemp])
                        
                        summarytemp = [index, xds, yds, 0, 0, 0, 0]
                    
                    # Assign subbasin number to the downstream cell and mark it off
                    subbasin[xds, yds] = index
                    marked[xds, yds] = 1
                    xtemp = xds
                    ytemp = yds
                else:
                    # If the downstream neighbor has been processed then move on to the next headwater cell
                    active = False
                    ends[xtemp, ytemp] = 3
                    summarytemp[3:5] = [xtemp, ytemp]
                    summarytemp[5] = subbasin[xds, yds]
            else:
                # Outside the domain
                active = False
                ends[xtemp, ytemp] = 3
                summarytemp[3:5] = [xtemp, ytemp]
                summarytemp[5] = -1
        
        if first:
            summary = np.array([summarytemp])
            first = False
        else:
            summary = np.vstack([summary, summarytemp])
    
    # Set column names for summary
    # Columns: Basin_ID, start_x, start_y, end_x, end_y, Downstream_Basin_ID, Area
    
    ### 2. Get the drainage basins for every segment
    subbasinA = subbasin.copy()
    
    # Start a queue with all the cells in the river
    queue_indices = np.where(subbasin > 0)
    queue = np.column_stack([queue_indices[0], queue_indices[1]])
    blist = np.column_stack([queue_indices[0], queue_indices[1]])
    
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
            
            # Add one to the subbasin area for the summary
            sbtemp = int(subbasinA[xtemp, ytemp])
            summary[sbtemp - 1, 6] += 1  # Adjust for 0-indexing
            
            # Look for cells that drain to this cell
            for d in range(4):
                xus = xtemp - kd[d, 0]
                yus = ytemp - kd[d, 1]
                
                if (xus * yus > 0 and xus < nx and yus < ny):
                    if mask[xus, yus] == 1 and subbasinA[xus, yus] == 0:
                        if direction[xus, yus] == d4[d]:
                            # Assign the subbasin number to the upstream cell
                            subbasinA[xus, yus] = subbasinA[xtemp, ytemp]
                            # Add to the queue
                            if queue2 is None:
                                queue2 = np.array([[xus, yus]])
                            else:
                                queue2 = np.vstack([queue2, [xus, yus]])
        
        if queue2 is not None and len(queue2) >= 2:
            queue = queue2
            nqueue = len(queue)
            ii += 1
        else:
            nqueue = 0
    
    ### 3. If merge_th > 0, look for basins with areas less than the merge threshold,
    # that don't drain out of the domain and merge with their downstream neighbors
    
    delete = []  # List of subbasins to delete from summary list
    
    if merge_th > 0:
        print("WARNING: non-zero merge thresholds are not compatible with the RiverSmooth function")
        nsb = len(summary)
        
        for i in range(nsb):
            # Check if area is less than the threshold & it does not drain externally
            if summary[i, 6] < merge_th and summary[i, 5] > 0:
                delete.append(i)
                bas1 = int(summary[i, 0])
                bas2 = int(summary[i, 5])
                
                # Replace numbers in the subbasin matrix
                ilist = subbasin == bas1
                subbasin[ilist] = bas2
                
                # Replace numbers in the subbasin area matrix
                ilistA = subbasinA == bas1
                subbasinA[ilistA] = bas2
                
                # Adjust the summary matrix for the downstream basin
                # Increment the downstream basin's drainage area
                downstream_idx = np.where(summary[:, 0] == bas2)[0][0]
                summary[downstream_idx, 6] += summary[i, 6]
                
                # Change the location of the downstream basin's headwater cell
                summary[downstream_idx, 1:3] = summary[i, 1:3]
                
                # Change the downstream basin number for any upstream basins to downstream basin
                uplist = summary[:, 5] == bas1
                summary[uplist, 5] = bas2
        
        if delete:
            summary = np.delete(summary, delete, axis=0)
    
    output_list = {
        "segments": subbasin,
        "subbasins": subbasinA,
        "RiverMask": rivers,
        "summary": summary
    }
    
    return output_list 