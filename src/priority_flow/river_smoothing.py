"""
River Smoothing functions for PriorityFlow.

This module provides functions to smooth a DEM along a pre-defined stream network,
requiring pre-defined stream segments and subbasins from the CalcSubbasins function.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from .fix_drainage import fix_drainage


def river_smooth(
    dem: np.ndarray,
    direction: np.ndarray,
    mask: Optional[np.ndarray] = None,
    river_summary: np.ndarray = None,
    river_segments: np.ndarray = None,
    bank_epsilon: float = 0.01,
    river_epsilon: float = 0.0,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    printflag: bool = False
) -> Dict[str, Union[np.ndarray, np.ndarray]]:
    """
    Apply smoothing to a DEM along a pre-defined stream network.
    
    This function will smooth a DEM along a stream network. It requires pre-defined stream segments 
    and subbasins which can be obtained using the CalcSubbasins function.
    
    Parameters
    ----------
    dem : np.ndarray
        Digital Elevation Model matrix
    direction : np.ndarray
        Flow direction matrix
    mask : np.ndarray, optional
        Domain mask matrix. Defaults to processing everything if not provided.
    river_summary : np.ndarray
        A table summarizing the stream segments in a domain with 1 row per stream segment
        and 7 columns: (1) subbasin number, (2) x and (3) y index of the upstream end of the stream segment,
        (4) x and (5) y index of the downstream end of the river segment, (6) the subbasin number for the 
        downstream basin (-1 indicates a subbasin draining out of the domain), (7) drainage area of the subbasin
    river_segments : np.ndarray
        Nx by Ny matrix indicating the subbasin number for all grid cells on the river network
        (all cells not on the river network should be 0)
    bank_epsilon : float, optional
        The minimum elevation difference between cells walking up the banks from the river network. Defaults to 0.01.
    river_epsilon : float, optional
        The minimum elevation difference between cells along the river. Defaults to 0.0.
    d4 : Tuple[int, int, int, int], optional
        D4 direction numbering scheme. Defaults to (1, 2, 3, 4) for (down, left, up, right).
    printflag : bool, optional
        Flag to enable debug printing. Defaults to False.
    
    Returns
    -------
    Dict[str, Union[np.ndarray, np.ndarray]]
        A dictionary containing:
        - 'dem.adj': A matrix with the adjusted DEM values following the river smoothing operation
        - 'processed': A matrix indicating the cells that were processed by this routine (1=processed, 0=not processed)
        - 'summary': A summary of the reach properties with columns for segment ID, start/end coordinates, 
          length, elevations, and delta applied along the segment
    
    Notes
    -----
    This function implements a river network smoothing algorithm that:
    1. Processes river segments from downstream to upstream
    2. Ensures proper elevation gradients along river channels
    3. Applies minimum elevation differences between river cells
    4. Fixes drainage issues on hillslopes adjacent to river cells
    5. Maintains topological consistency of the river network
    
    The algorithm handles:
    - Terminal river segments (draining out of domain)
    - Internal river segments (draining to other segments)
    - Elevation smoothing along river channels
    - Hillslope drainage fixes
    - River network topology validation
    """
    nx, ny = direction.shape
    
    # Default to processing everything if mask not provided
    if mask is None:
        mask = np.ones((nx, ny))
    
    # D4 neighbors
    # Rows: down, left, top, right
    # Columns: (1) deltax, (2) deltay, direction number if walking downstream, and (4) upstream
    kd = np.array([
        [0, -1, 1, 3],    # Down
        [-1, 0, 2, 2],    # Left
        [0, 1, 3, 1],     # Top
        [1, 0, 4, 4]      # Right
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
    
    # Setup a river list
    nriver = river_summary.shape[0]
    marked_segments = np.zeros(nriver, dtype=int)  # marker for keeping track of which reaches are processed
    marked_matrix = np.zeros((nx, ny), dtype=int)
    
    # Setup a smoothing summary
    riversmooth_summary = np.zeros((nriver, 9))
    # Column names: SegmentID, Start.X, Start.Y, End.X, End.Y, Length, Top.Elevation, Bottom.Elevation, delta
    # TODO: This should be a dataframe!
    riversmooth_summary[:, :5] = river_summary[:, :5]
    
    # Make a mask of the hillslope cells
    hillmask = mask.copy()
    hillmask[river_segments > 0] = 0
    
    # First make a list of all the terminal river reaches
    # queue = which(river.summary[,6]==(-1))
    queue = np.where(river_summary[:, 5] <= 0)[0]  # Note: R is 1-indexed, Python is 0-indexed
    if len(queue) > 0:
        active = True
    else:
        print("No terminal river segments provided, not adjusting DEM")
        active = False
    
    # Start a new dem
    dem2 = dem.copy()
    
    # Get the length of every river segment
    max_river_index = int(np.max(river_summary[:, 0]))
    river_length = np.zeros(max_river_index + 1)  # +1 because river indices start at 1
    for i in range(nx):
        for j in range(ny):
            rtemp = int(river_segments[i, j])
            if rtemp > 0:
                river_length[rtemp] = river_length[rtemp] + 1
    
    # Loop over the river segments working upstream
    # Starting from every terminal segment (i.e., segments with a downstream river number of -1)
    while active:
        indr = queue[0]
        r = int(river_summary[indr, 0])  # river segment number
        rdown = int(river_summary[indr, 5])
        length = int(river_length[r])
        riversmooth_summary[indr, 5] = river_length[r]
        
        # Find the top and bottom elevations of the current river segment
        top = dem2[int(river_summary[indr, 1]), int(river_summary[indr, 2])]  # Start coordinates
        
        # If it's a terminal reach then the bottom elevation will be the bottom of the reach
        if rdown <= 0:
            bottom = dem2[int(river_summary[indr, 3]), int(river_summary[indr, 4])]  # End coordinates
            length = length - 1
        else:
            # If not then use the elevation downstream of the bottom point of the reach
            bdir = int(dir2[int(river_summary[indr, 3]), int(river_summary[indr, 4])])
            bottom = dem2[int(river_summary[indr, 3]) + kd[bdir-1, 0], int(river_summary[indr, 4]) + kd[bdir-1, 1]]
        
        topmin = bottom + river_epsilon * length
        if top < topmin:
            # Calculate the delta from the original dem
            top0 = dem[int(river_summary[indr, 1]), int(river_summary[indr, 2])]
            if rdown > 0:
                bdir = int(dir2[int(river_summary[indr, 3]), int(river_summary[indr, 4])])
                bottom0 = dem[int(river_summary[indr, 3]) + kd[bdir-1, 0], int(river_summary[indr, 4]) + kd[bdir-1, 1]]
            else:
                bottom0 = dem[int(river_summary[indr, 3]), int(river_summary[indr, 4])]
            
            # Use this delta from the original dem to adjust the top elevation
            delta = max((top0 - bottom0) / length, river_epsilon)
            top = bottom + delta * length
            dem2[int(river_summary[indr, 1]), int(river_summary[indr, 2])] = top
            
            if printflag:
                print(f"River top elevation < river bottom elevation for segment {r}")
                print(f"Original top {top0:.2f} and original bottom {bottom0:.2f}")
                print(f"Adjusting the top elevation from {top0:.2f} to {top:.2f}")
        
        if printflag:
            print(f"River segment: {r}")
            print(f"Start: {river_summary[indr, 1]} {river_summary[indr, 2]} {top:.1f}")
            print(f"End: {river_summary[indr, 3]} {river_summary[indr, 4]} {bottom:.1f}")
        
        # Walk from top to bottom smoothing out the river cells
        indx = int(river_summary[indr, 1])
        indy = int(river_summary[indr, 2])
        marked_matrix[indx, indy] = marked_matrix[indx, indy] + 1
        
        if length > 1:
            delta = (top - bottom) / length
        else:
            delta = 0
        
        if delta < 0:
            print(f"Warning: Calculated delta < 0, setting delta to 0 for segment {r}")
            delta = 0
        
        temp = top
        riversmooth_summary[indr, 6] = top
        riversmooth_summary[indr, 7] = bottom
        riversmooth_summary[indr, 8] = delta
        
        if length > 1:
            for i in range(2, length + 1):
                temp = temp - delta
                # Find the downstream point and adjust its elevation
                dirtemp = int(dir2[indx, indy])
                downindx = indx + kd[dirtemp-1, 0]
                downindy = indy + kd[dirtemp-1, 1]
                
                if river_segments[downindx, downindy] == r:
                    dem2[downindx, downindy] = temp
                    marked_matrix[downindx, downindy] = marked_matrix[downindx, downindy] + 1
                    
                    # Loop up the hillslope from the point and make sure everything drains
                    drainfix = fix_drainage(
                        dem=dem2, 
                        direction=dir2, 
                        mask=hillmask, 
                        bank_epsilon=bank_epsilon, 
                        startpoint=[downindx, downindy]
                    )
                    dem2 = drainfix['dem.adj']
                else:
                    print(f"Warning: Check Segment for branches {r}")
                
                # Move to the downstream point
                indx = downindx
                indy = downindy
        
        # Mark this segment as done
        marked_segments[indr] = marked_segments[indr] + 1
        
        # Find all of the river segments that drain to this segment
        uplist = np.where(river_summary[:, 5] == r)[0]
        
        # If there are upstream segments then add to the top of the Q overwriting the segment that
        # was just done if not just remove the current segment from the queue
        if len(uplist) > 0:
            queue = np.concatenate([uplist, queue[1:]])
        else:
            queue = queue[1:]
        
        if len(queue) == 0:
            active = False
    
    output_list = {
        "dem.adj": dem2,
        "processed": marked_matrix,
        "summary": riversmooth_summary
    }
    
    return output_list 