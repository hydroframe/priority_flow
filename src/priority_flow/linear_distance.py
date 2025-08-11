"""
Linear Distance functions for PriorityFlow.

This module provides functions to calculate the minimum linear distance between
every point in a 2D array and a mask of target points using a step-by-step
distance checking algorithm.
"""

import numpy as np
from typing import Optional


def lin_dist(
    target_points: np.ndarray,
    mask: Optional[np.ndarray] = None,
    cell_size: float = 1.0,
    max_dist: Optional[float] = None,
    printflag: bool = False
) -> np.ndarray:
    """
    Calculate the minimum linear distance from a feature set.
    
    Calculates the minimum linear distance between every point in a 2D array
    and a mask of target points. Note that this function assumes dx=dy, i.e.,
    square grid cells. If this is not the case, it will not work correctly.
    
    Parameters
    ----------
    target_points : np.ndarray
        Matrix with a value of 1 for all cells that you would like to calculate
        the distance to and 0 for all other cells
    mask : np.ndarray, optional
        Processing mask. Defaults to processing everything
    cell_size : float, optional
        The size of a grid cell. Note that this function assumes dx=dy, i.e.,
        square grid cells. Defaults to 1.0
    max_dist : float, optional
        Maximum distance to check. All cells with no distance found <= max_dist
        will be assigned max_dist. If None, checks out to the total size of the domain
    printflag : bool, optional
        Whether to print progress information. Defaults to False
    
    Returns
    -------
    np.ndarray
        Matrix containing the minimum distance from each cell to the nearest target point.
        Cells outside the mask are assigned NaN.
    
    Notes
    -----
    This function implements a step-by-step distance checking algorithm that:
    1. Creates an ordered list of distance steps and rotations to check
    2. Iteratively checks each distance incrementally
    3. Uses array shifting to efficiently count target points at each distance
    4. Assigns distances to cells as they are found
    5. Continues until all cells have been assigned distances or max_dist is reached
    
    The algorithm handles:
    - Square grid cells (dx = dy)
    - Configurable maximum distance limits
    - Efficient array operations for distance checking
    - Mask-based domain processing
    - Progressive distance assignment
    
    The function assumes square grid cells and uses Euclidean distance calculations.
    For non-square grids, results may be incorrect.
    """
    nx, ny = target_points.shape
    
    if mask is None:
        mask = np.ones((nx, ny))  # Default to processing everything
    
    # Incomplete matrix for cells still needing to be processed
    # Target points and points outside the mask are initialized to 0
    incomplete = (1 - target_points) * mask
    
    n_missing = np.sum(incomplete)
    distance = np.zeros((nx, ny))
    
    # Figure out the order of steps (horizontal and vertical) and rotations
    # (distance from horizontal and vertical) to check
    # If max_dist is unspecified, check out to the total size of the domain
    if max_dist is None:
        max_dist = np.sqrt((nx * cell_size)**2 + (ny * cell_size)**2)
    
    order_list = []
    for s in range(1, max(nx, ny) + 1):
        for r in range(s + 1):
            dist = np.sqrt((r * cell_size)**2 + (s * cell_size)**2)
            if dist <= max_dist:
                order_list.append([s, r, dist])
    
    # Convert to numpy array and sort by distance
    if order_list:
        order = np.array(order_list)
        sort_indices = np.argsort(order[:, 2])
        order = order[sort_indices]
    else:
        order = np.empty((0, 3))
    
    ndist = order.shape[0]  # Number of unique distances to test
    
    if printflag:
        print("Orders created")
    
    while n_missing > 0:
        for d in range(ndist):
            s = int(order[d, 0])
            r = int(order[d, 1])
            d_temp = order[d, 2]
            
            if printflag:
                print(f"Step={s}, Rotation={r}, N_missing={n_missing}, distance={d_temp:.4f}")
            
            # Counting up the number of target points within a given step (s = later distance from center)
            # and rotation (r = distance off vertical or horizontal axes)
            temp_count = np.zeros((nx, ny))
            
            # Check if s and r are within domain bounds
            if s < nx and r < ny:
                # Apply target point counting with array shifting
                temp_count[0:(nx-s), 0:(ny-r)] += target_points[s:nx, r:ny]
                temp_count[0:(nx-s), r:ny] += target_points[s:nx, 0:(ny-r)]
                temp_count[s:nx, 0:(ny-r)] += target_points[0:(nx-s), r:ny]
                temp_count[s:nx, r:ny] += target_points[0:(nx-s), 0:(ny-r)]
            
            # Check if r and s are within domain bounds (for rotation)
            if r < nx and s < ny:
                temp_count[0:(nx-r), 0:(ny-s)] += target_points[r:nx, s:ny]
                temp_count[0:(nx-r), s:ny] += target_points[r:nx, 0:(ny-s)]
                temp_count[r:nx, 0:(ny-s)] += target_points[0:(nx-r), s:ny]
                temp_count[r:nx, s:ny] += target_points[0:(nx-r), 0:(ny-s)]
            
            # Convert to binary (any target points found = 1)
            temp_count[temp_count > 0] = 1
            
            # Record the distance for any cell with a temp_count=1 that is still incomplete
            distance += temp_count * incomplete * d_temp
            
            # Set all the cells that had a temp_count=1 to complete
            incomplete = incomplete - temp_count
            incomplete[incomplete < 0] = 0
            
            # Count up the number of cells still needing a distance
            n_missing = np.sum(incomplete)
            
            # Stop if you are out of missing cells
            if n_missing == 0:
                break
    
    if printflag:
        print("Distance calculated")
    
    # Set cells outside the mask to NaN
    distance[mask == 0] = np.nan
    
    return distance 