"""
Priority Flow Processing of D4 Stream Networks for PriorityFlow.

This module provides functions for processing stream network cells walking upstream
on D4 neighbors in a river mask, with priority queue management and D4 neighbor traversal.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union


def d4_traverse_b(
    dem: np.ndarray,
    queue: np.ndarray,
    marked: np.ndarray,
    mask: Optional[np.ndarray] = None,
    step: Optional[np.ndarray] = None,
    direction: Optional[np.ndarray] = None,
    basins: Optional[np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    printstep: bool = False,
    nchunk: int = 100,
    epsilon: float = 0.0,
    printflag: bool = False
) -> Dict[str, np.ndarray]:
    """
    Priority flow processing of D4 Stream Networks.
    
    This function will process all stream network cells walking upstream on D4 neighbors
    in a river mask. Where no D4 neighbors exist, it looks for D8 neighbors and creates
    D4 bridges to these diagonal cells.
    
    Parameters
    ----------
    dem : np.ndarray
        Elevation matrix
    queue : np.ndarray
        A priority queue to start from with three columns: x, y, elevation
    marked : np.ndarray
        A matrix of which cells have been marked already
    mask : np.ndarray, optional
        Mask with ones for cells to be processed and zeros for everything else.
        Defaults to a mask of all 1's
    step : np.ndarray, optional
        A matrix of the step number for cells that have been processed.
        Defaults to all zeros
    direction : np.ndarray, optional
        A matrix of the flow directions for cells that have been processed.
        Defaults to all zeros
    basins : np.ndarray, optional
        A matrix of basin numbers that can be created by the initialization script.
        If you input this, every cell will be assigned the same basin as the cell that adds it
    d4 : Tuple[int, int, int, int], optional
        Directional numbering system: the numbers to assign to down, left, top, right.
        Defaults to (1, 2, 3, 4)
    printstep : bool, optional
        If True, it will print out the step number and the size of the queue.
        Defaults to False
    nchunk : int, optional
        Parameter for queue splitting. The top 'nchunk' values will be put into the
        primary queue for processing initially. Defaults to 100
    epsilon : float, optional
        Amount to add to filled areas to avoid creating flats. Defaults to 0.0
    printflag : bool, optional
        Optional flag to print function progress. Defaults to False
    
    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing:
        - 'dem': Updated elevation matrix
        - 'mask': Processing mask
        - 'marked': Marked cells matrix
        - 'step': Step number matrix
        - 'direction': Flow direction matrix
        - 'basins': Basin assignment matrix
    
    Notes
    -----
    This function implements a priority flood algorithm that:
    1. Processes cells in order of increasing elevation
    2. Uses queue splitting for memory efficiency
    3. Traverses D4 neighbors (down, left, top, right)
    4. Assigns flow directions and basin numbers
    5. Handles border cells with special logic
    """
    t0 = time.time()
    nx, ny = dem.shape
    dem_new = dem.copy()
    
    # Setup matrices for anything that wasn't input
    if mask is None:
        mask = np.ones((nx, ny))  # Default to processing everything
    if step is None:
        step = np.zeros((nx, ny))  # Start all steps at zero
    if direction is None:
        direction = np.full((nx, ny), np.nan)  # Make a blank direction matrix
    if basins is None:
        basins = np.zeros((nx, ny))  # Make all the basins=0
    
    # D4 neighbors - ordered down, left, top, right
    # We are walking upstream so the direction needs to point opposite
    kd = np.array([
        [0, -1, d4[2]],    # Down
        [-1, 0, d4[3]],    # Left
        [0, 1, d4[0]],     # Top
        [1, 0, d4[1]]      # Right
    ])
    
    split = 0
    q1max = 0
    nqueue = len(queue) // 3
    nstep = 0
    queue_temp = None
    
    # Split the queue in 2 using the top nchunk values for the first
    # queue and the rest for the second
    if nqueue > nchunk:
        qsort = queue[queue[:, 2].argsort()]
        queue1 = qsort[:nchunk]
        queue2 = qsort[nchunk:]
        th = queue2[0, 2]
        nqueue2 = len(queue2) // 3
        nqueue = len(queue1)
        if printstep:
            print(f'Initial queue: {len(queue) // 3} splitting. Q1={nqueue}, Q2={nqueue2}')
        # Adding the matrix step so it doesn't become a vector if it's only one
        if nqueue2 == 1:
            queue2 = queue2.reshape(1, 3)
    else:
        if printstep:
            print(f'Initial queue: {nqueue} Not splitting')
        queue1 = queue.copy()
        queue2 = None
        nqueue2 = 0
        th = queue1[nqueue - 1, 2] * 1.1
    
    t0 = time.time()
    
    while nqueue > 0:
        ##############
        # Pick the lowest DEM cell on the queue
        pick = np.argmin(queue1[:, 2])
        xc = int(queue1[pick, 0])
        yc = int(queue1[pick, 1])
        demc = queue1[pick, 2]
        
        ##############
        # Look for D4 neighbor cells that are on the mask and add to queue
        
        count = 0
        # Check if the original cell is on the border
        bdrchk = direction[xc, yc]
        
        for k in range(4):
            xk = xc + kd[k, 0]
            yk = yc + kd[k, 1]
            
            # Check that the neighbor is inside the domain and on the mask
            if 0 <= yk < ny and 0 <= xk < nx:
                if mask[xk, yk] == 1 and marked[xk, yk] == 0:
                    demtemp = max((demc + epsilon), dem[xk, yk])
                    dem_new[xk, yk] = demtemp
                    
                    # If it's less than the threshold add it to Q1, if not add it to Q2
                    if demtemp < th:
                        queue1 = np.vstack([queue1, [xk, yk, demtemp]])
                    else:
                        if queue_temp is None:
                            queue_temp = np.array([[xk, yk, demtemp]])
                        else:
                            queue_temp = np.vstack([queue_temp, [xk, yk, demtemp]])
                    
                    marked[xk, yk] = 1
                    step[xk, yk] = step[xc, yc] + 1
                    direction[xk, yk] = kd[k, 2]
                    basins[xk, yk] = basins[xc, yc]
                    count += 1
                    
                    # If the original cell is on the border and lacking a flow direction then
                    # give it the direction of the cell it just added as long as this points out of the domain
                    if np.isnan(bdrchk):
                        # xy location of the cell opposite the cell being processed
                        xo = xc - kd[k, 0]
                        yo = yc - kd[k, 1]
                        # If this is outside the grid or the mask then apply the direction to this cell
                        if (yo * xo == 0 or yo >= ny or xo >= nx or 
                            (0 <= xo < nx and 0 <= yo < ny and mask[xo, yo] == 0)):
                            direction[xc, yc] = kd[k, 2]
        
        ##############
        # Remove from the queue and move on
        
        nqueue = len(queue1) // 3
        nqueue_temp = nqueue2 + (len(queue_temp) // 3 if queue_temp is not None else 0)
        
        # If you have 2 or more items in Q1 then you will have 1 left after
        # deleting the current cell so continue with Q1
        if nqueue > 1:
            queue1 = np.delete(queue1, pick, axis=0)
            nqueue = len(queue1) // 3
            q1max = max(nqueue, q1max)
        # If you have only 1 item left in Q1 then it will be empty after deleting the
        # current cell so see if you are done or if you need to grab another chunk from Q2
        else:
            # Look and see if there are still values in Q2 to merge
            if nqueue_temp > nchunk:
                t1 = time.time()
                split += 1
                if printstep:
                    print(f'P{split} Q2 {nqueue_temp} nstep={nstep} Q1 Max: {q1max} time {round(t1 - t0, 2)}')
                
                if queue_temp is not None:
                    queue2 = np.vstack([queue2, queue_temp]) if queue2 is not None else queue_temp
                queue_temp = None
                qsort = queue2[queue2[:, 2].argsort()]
                queue1 = qsort[:nchunk]
                queue2 = qsort[nchunk:]
                th = queue2[0, 2]
                nqueue = len(queue1) // 3
                nqueue2 = len(queue2) // 3
                q1max = 0
                t0 = time.time()
                
            elif nqueue_temp <= nchunk and nqueue2 > 0:
                split += 1
                if printstep:
                    print(f'Split: {split} Q2 {nqueue_temp} taking last chunk, nstep={nstep} Q1 Max: {q1max}')
                
                if queue_temp is not None:
                    queue1 = np.vstack([queue2, queue_temp])
                queue_temp = None
                queue2 = None
                th = np.max(dem[mask == 1]) * 1.1
                nqueue = len(queue1) // 3
                q1max = 0
                nqueue2 = 0
            else:
                if printstep:
                    print(f'Q1 depleted, Q2 {nqueue2} done!!')
                nqueue = 0
        
        # If there is only one row it will treat it as a vector and things will break
        if nqueue == 1:
            queue1 = queue1.reshape(1, 3)
        
        nstep += 1
        if printstep:
            print(f"Step: {nstep} NQueue: {nqueue} Queue2: {nqueue2} Qtemp: {len(queue_temp) // 3 if queue_temp is not None else 0}")
    
    # After it's all done do a final pass to fill in flow directions for any
    # of the cells on the initial queue which didn't get a flow direction
    # assigned because they didn't add any neighbors to the queue
    border_miss = np.where(np.isnan(direction) & (mask == 1))
    
    # Make a padded dem repeating the edge values
    dem_pad = dem.copy()
    dem_pad = np.vstack([dem[0, :], dem_pad, dem[-1, :]])
    dem_pad = np.column_stack([dem_pad[:, 0], dem_pad, dem_pad[:, -1]])
    
    # Loop over border cells and assign a flow direction based on steepest neighbor
    if len(border_miss[0]) > 0:
        for b in range(len(border_miss[0])):
            bx = border_miss[0][b] + 1
            by = border_miss[1][b] + 1
            # Calculate the elevation difference to every D4 neighboring cell
            dem_negh = np.array([
                dem[bx - 1, by - 1] - dem_pad[bx + kd[0, 0], by + kd[0, 1]],
                dem[bx - 1, by - 1] - dem_pad[bx + kd[1, 0], by + kd[1, 1]],
                dem[bx - 1, by - 1] - dem_pad[bx + kd[2, 0], by + kd[2, 1]],
                dem[bx - 1, by - 1] - dem_pad[bx + kd[3, 0], by + kd[3, 1]]
            ])
            # Pick the neighbor with the biggest elevation difference
            # Note if there is a tie this will just pick the first
            pick = np.argmax(np.abs(dem_negh))
            # Assign the flow direction
            if dem_negh[pick] < 0:
                direction[bx - 1, by - 1] = kd[pick, 2]
            else:
                direction[bx - 1, by - 1] = d4[pick]
    
    output_list = {
        "dem": dem_new,
        "mask": mask,
        "marked": marked,
        "step": step,
        "direction": direction,
        "basins": basins
    }
    
    return output_list 