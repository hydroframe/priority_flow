"""
Downstream Extract functions for PriorityFlow.

This module provides functions to walk downstream from a point and extract values
from a matrix, creating a path mask and ordered list of cells along the flow path.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def path_extract(
    input_matrix: np.ndarray,
    direction: np.ndarray,
    mask: Optional[np.ndarray] = None,
    startpoint: Union[List[int], Tuple[int, int], np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4)
) -> Dict[str, np.ndarray]:
    """
    Walk downstream from a point and extract values from a matrix.
    
    This function grabs out values from a matrix by walking downstream from a point,
    following the flow direction until reaching the domain boundary or masked area.
    
    Parameters
    ----------
    input_matrix : np.ndarray
        The matrix of values that you would like to extract the stream path from
    direction : np.ndarray
        Flow direction matrix
    mask : np.ndarray, optional
        Processing mask. Defaults to processing everything
    startpoint : Union[List[int], Tuple[int, int], np.ndarray]
        The x,y index of the grid cell you would like to start from
    d4 : Tuple[int, int, int, int], optional
        Directional numbering system: down, left, top, right. Defaults to (1, 2, 3, 4)
    
    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing:
        - 'data': Numpy array of values extracted along the downstream path
        - 'path_mask': Matrix mapping the path with step numbers
        - 'path_list': Numpy array of cells on the path in order
    
    Notes
    -----
    This function implements a downstream walking algorithm that:
    1. Starts from a specified grid cell
    2. Follows flow directions downstream
    3. Extracts values from the input matrix along the path
    4. Creates a path mask and ordered cell list
    5. Stops when reaching domain boundary or masked area
    
    The algorithm handles different D4 numbering schemes by internally renumbering
    to a standard system (1=down, 2=left, 3=up, 4=right).
    """
    nx, ny = direction.shape
    
    if mask is None:
        mask = np.ones((nx, ny))  # Default to processing everything
    
    if startpoint is None:
        raise ValueError("startpoint must be provided")
    
    # Convert startpoint to proper format
    if isinstance(startpoint, (list, tuple)):
        startpoint = np.array(startpoint)
    if startpoint.ndim == 1:
        startpoint = startpoint.reshape(1, -1)
    
    # Initialize path tracking
    path_mask = np.zeros((nx, ny))  # Matrix mapping the path
    path = []  # List of cells on the path in order
    
    # D4 neighbors
    # Rows: down, left, top, right
    # Columns: (1) deltax, (2) deltay, direction number if you are walking downstream
    kd = np.array([
        [0, -1, 1],    # Down
        [-1, 0, 2],    # Left
        [0, 1, 3],     # Top
        [1, 0, 4]      # Right
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
    
    # Initialize tracking variables
    indx = int(startpoint[0, 0])
    indy = int(startpoint[0, 1])
    step = 1
    active = True
    output = []
    
    # Walking downstream
    while active:
        # Extract value from current cell
        output.append(input_matrix[indx, indy])
        path_mask[indx, indy] = step
        path.append([indx, indy])
        
        # Look downstream
        dirtemp = int(dir2[indx, indy])
        downindx = indx + kd[dirtemp - 1, 0] 
        downindy = indy + kd[dirtemp - 1, 1]
        
        # Check if we have made it out of the domain
        if (downindx < 0 or downindx >= nx or downindy < 0 or downindy >= ny):
            active = False
        else:
            # Check if the downstream cell is masked
            if mask[downindx, downindy] == 0:
                active = False
        
        # Update the new indices
        indx = downindx
        indy = downindy
        step += 1
    
    # Convert output and path to numpy arrays for consistency
    output_array = np.array(output) if output else np.empty(0)
    path_array = np.array(path) if path else np.empty((0, 2))
    
    output_list = {
        "data": output_array,
        "path_mask": path_mask,
        "path_list": path_array
    }
    
    return output_list 