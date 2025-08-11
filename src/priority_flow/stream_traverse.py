"""
Stream Traverse functions for PriorityFlow.

This module provides functions to process stream networks by walking upstream
on D4 neighbors in a river mask, with D8 neighbor bridging for diagonal cells.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def stream_traverse(
    dem: np.ndarray,
    mask: np.ndarray,
    queue: np.ndarray,
    marked: np.ndarray,
    step: Optional[np.ndarray] = None,
    direction: Optional[np.ndarray] = None,
    basins: Optional[np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    printstep: bool = False,
    epsilon: float = 0.0
) -> Dict[str, np.ndarray]:
    """
    DEM processing of stream networks.
    
    Function to process stream networks walking upstream on D4 neighbors in a river mask.
    Where no D4 neighbors exist, it looks for D8 neighbors and creates D4 bridges
    to these diagonal cells.
    
    Parameters
    ----------
    dem : np.ndarray
        Digital Elevation Model matrix
    mask : np.ndarray
        Mask with zeros for non-river cells and 1 for river cells
    queue : np.ndarray
        A priority queue to start from with three columns: x, y, elevation
    marked : np.ndarray
        A matrix of which cells have been marked already
    step : np.ndarray, optional
        A matrix of the step number for cells that have been processed.
        Defaults to all zeros if not provided.
    direction : np.ndarray, optional
        A matrix of the flow directions for cells that have been processed.
        Defaults to all NaN if not provided.
    basins : np.ndarray, optional
        A matrix of basin numbers that can be created by the initialization script.
        If you input this, every cell will be assigned the same basin as the cell that adds it.
        Defaults to all zeros if not provided.
    d4 : Tuple[int, int, int, int], optional
        D4 direction numbering scheme. Defaults to (1, 2, 3, 4) for (down, left, up, right).
    printstep : bool, optional
        If True, it will print out the step number and the size of the queue. Defaults to False.
    epsilon : float, optional
        Amount to add to filled areas to avoid creating flats. Defaults to 0.0.
    
    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing:
        - 'dem': The processed DEM matrix
        - 'mask': The input river mask matrix
        - 'marked': Matrix of marked cells
        - 'step': Matrix of processing step numbers
        - 'direction': Matrix of flow directions
        - 'basins': Matrix of basin assignments
    
    Notes
    -----
    This function implements a stream network processing algorithm that:
    1. Processes stream networks by walking upstream on D4 neighbors
    2. Uses a priority queue based on elevation values
    3. Creates D4 bridges to D8 diagonal cells when needed
    4. Maintains flow direction and basin information
    5. Applies epsilon adjustments to prevent flat areas
    
    The algorithm handles:
    - Priority queue processing based on elevation
    - D4 neighbor exploration and processing
    - D8 neighbor bridging with D4 connections
    - Flow direction assignment and tracking
    - Basin number propagation
    - Step counting and progress tracking
    
    The function walks upstream from the queue starting points, ensuring that
    river cells are properly connected and elevations are maintained to prevent
    artificial flat areas in the stream network.
    """
    nx, ny = dem.shape
    dem_new = dem.copy()
    
    # Setup matrices for anything that wasn't input
    if step is None:
        step = np.zeros((nx, ny))  # Start all steps at zero
    
    if direction is None:
        direction = np.full((nx, ny), np.nan)  # Make a blank direction matrix
    
    if basins is None:
        basins = np.zeros((nx, ny))  # Make all the basins = 0
    
    # D4 neighbors
    # Ordered: down, left, top, right
    kd = np.array([
        [0, -1, d4[2]],    # Down
        [-1, 0, d4[3]],    # Left
        [0, 1, d4[0]],     # Top
        [1, 0, d4[1]]      # Right
    ])
    # We are walking upstream so the direction needs to point opposite
    
    # D8 neighbors
    kd8 = np.array([
        [-1, -1, d4[2], d4[3]],  # Top-left
        [-1, 1, d4[0], d4[3]],   # Bottom-left
        [1, 1, d4[0], d4[1]],    # Bottom-right
        [1, -1, d4[2], d4[1]]    # Top-right
    ])
    # Directions for the two D4 neighbor cells that are tested
    # The first neighbor cell is xC, yk (i.e., just moving up or down)
    # The second neighbor cell is xk, yC (i.e., just moving left or right)
    
    nqueue = queue.shape[0]
    nstep = 0
    
    while nqueue > 0:
        ##############
        # Pick the lowest DEM cell on the queue
        pick = np.argmin(queue[:, 2])
        x_c = int(queue[pick, 0])
        y_c = int(queue[pick, 1])
        dem_c = queue[pick, 2]
        
        if printstep:
            print(f'pick: {x_c} {y_c}')
        
        ##############
        # Look for D4 neighbor cells that are on the mask and add to queue
        count = 0
        for k in range(4):
            xk = x_c + kd[k, 0]
            yk = y_c + kd[k, 1]
            
            # Check that the neighbor is inside the domain and on the mask
            if (1 <= yk <= ny and 1 <= xk <= nx):
                if (mask[xk, yk] == 1 and marked[xk, yk] == 0):
                    dem_temp = max((dem_c + epsilon), dem[xk, yk])
                    dem_new[xk, yk] = dem_temp
                    queue = np.vstack([queue, [xk, yk, dem_temp]])
                    marked[xk, yk] = 1
                    step[xk, yk] = step[x_c, y_c] + 1
                    direction[xk, yk] = kd[k, 2]
                    basins[xk, yk] = basins[x_c, y_c]
                    count = count + 1
                    nqueue = nqueue + 1
        
        if printstep:
            print(f"{count} available D4 neighbors found")
        
        ##############
        # If you don't find any D4 neighbors, look for D8 neighbors
        # and choose the least cost D4 option to reach that D8 cell
        n4 = np.full((2, 4), np.nan)
        if count == 0:
            if printstep:
                print("No D4 neighbors, checking for diagonal")
            
            for k in range(4):
                xk = x_c + kd8[k, 0]
                yk = y_c + kd8[k, 1]
                
                count4 = 0
                n4 = np.full((2, 4), np.nan)
                
                # Check that the neighbor is inside the domain and on the mask
                if (1 <= yk <= ny and 1 <= xk <= nx):
                    # Check that it hasn't already been marked
                    if (marked[xk, yk] < 1 and mask[xk, yk] == 1):
                        if printstep:
                            print("found 1!")
                        
                        # Look for available D4 neighbors to add instead
                        if marked[x_c, yk] < 1:
                            n4[0, :] = [x_c, yk, dem[x_c, yk], kd8[k, 2]]
                            count4 = count4 + 1
                        
                        if marked[xk, y_c] < 1:
                            n4[1, :] = [xk, y_c, dem[xk, y_c], kd8[k, 3]]
                            count4 = count4 + 1
                        
                        # Choose the neighbor which is the lowest without going under
                        # and if not, fill the highest
                        if count4 > 0:
                            if np.nanmin(n4[:, 2]) >= dem_c:
                                npick = np.nanargmin(n4[:, 2])
                            else:
                                npick = np.nanargmax(n4[:, 2])
                            
                            dem_temp = max((dem_c + epsilon), dem[int(n4[npick, 0]), int(n4[npick, 1])])
                            dem_new[int(n4[npick, 0]), int(n4[npick, 1])] = dem_temp
                            queue = np.vstack([queue, [n4[npick, 0], n4[npick, 1], dem_temp]])
                            marked[int(n4[npick, 0]), int(n4[npick, 1])] = 1
                            direction[int(n4[npick, 0]), int(n4[npick, 1])] = n4[npick, 3]
                            step[int(n4[npick, 0]), int(n4[npick, 1])] = step[x_c, y_c] + 1
                            basins[int(n4[npick, 0]), int(n4[npick, 1])] = basins[x_c, y_c]
                            count = count + 1
                            
                            if printstep:
                                print(f"Added D8 bridge: {int(n4[npick, 0])} {int(n4[npick, 1])}")
                        else:
                            if printstep:
                                print(f"{x_c} {y_c} D8 mask cell found with no viable D4 neighbors")
        
        ##############
        # Remove from the queue and move on
        nqueue = queue.shape[0] // 3
        if nqueue > 1:
            queue = np.delete(queue, pick, axis=0)
            nqueue = queue.shape[0] // 3
        else:
            nqueue = 0
        
        # If there is only one row, it will treat it as a vector and things will break
        if nqueue == 1:
            queue = queue.reshape(1, 3)
        
        nstep = nstep + 1
        if printstep:
            print(f"Step: {nstep} NQueue: {nqueue}")
    
    output_list = {
        "dem": dem_new,
        "mask": mask,
        "marked": marked,
        "step": step,
        "direction": direction,
        "basins": basins
    }
    
    return output_list 