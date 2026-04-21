"""
Find Orphan functions for PriorityFlow.

This module provides functions to find orphan branches - unprocessed river cells
that have D8 neighbors on the river network or on the boundary. This is useful
for identifying missed cells during river network processing.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Union


def find_orphan(
    dem: np.ndarray,
    mask: np.ndarray,
    marked: np.ndarray,
) -> Dict[str, Union[int, np.ndarray, None]]:
    """
    Find orphan branches in river network processing.
    
    Function to look for unprocessed river cells that have D8 neighbors on the
    river network or on the boundary.
    
    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model matrix
    mask : np.ndarray
        River network mask (1 for river cells, 0 for non-river cells)
    marked : np.ndarray
        Matrix of grid cells that have been processed (1 for processed, 0 for unprocessed)
    
    Returns
    -------
    Dict[str, Union[int, np.ndarray, None]]
        A dictionary containing:
        - 'norphan': Number of orphaned branches found
        - 'queue': Queue of marked neighbors to be processed (numpy array with columns: x, y, elevation)
    
    Notes
    -----
    This function implements an orphan detection algorithm that:
    1. Identifies unprocessed river cells (masked but not marked)
    2. Checks D8 connectivity to find marked neighbors
    3. Counts marked neighbors for each orphan cell
    4. Creates a processing queue from marked neighbors of orphan cells
    
    The algorithm uses D8 connectivity (8-directional) to ensure complete
    neighbor checking and proper orphan identification.
    """
    nx, ny = dem.shape

    kd = np.zeros((8, 2), dtype=float)
    kd[:, 0] = [0, -1, -1, -1, 0, 1, 1, 1]
    kd[:, 1] = [-1, -1, 0, 1, 1, 1, 0, -1]

    queue: np.ndarray | None = None

    missed = np.zeros((nx, ny), dtype=float)
    missed[(mask == 1) & (marked == 0)] = 1.0

    ncount = np.zeros((nx, ny), dtype=float)
    ncount[1 : (nx - 1), 1 : (ny - 1)] = (
        marked[2:nx, 2:ny]
        + marked[0 : (nx - 2), 0 : (ny - 2)]
        + marked[2:nx, 0 : (ny - 2)]
        + marked[0 : (nx - 2), 2:ny]
        + marked[1 : (nx - 1), 2:ny]
        + marked[1 : (nx - 1), 0 : (ny - 2)]
        + marked[2:nx, 1 : (ny - 1)]
        + marked[0 : (nx - 2), 1 : (ny - 1)]
    )

    ncount = ncount * missed
    norphan = int(np.sum(ncount > 0))

    if norphan > 0:
        flat = np.flatnonzero(np.ravel(ncount > 0, order="F"))
        ii, jj = np.unravel_index(flat, (nx, ny), order="F")
        addloc = np.column_stack((ii, jj))

        queue_list: list[list[float]] = []
        for n in range(norphan):
            xn = int(addloc[n, 0])
            yn = int(addloc[n, 1])
            for k in range(8):
                xtemp = xn + int(kd[k, 0])
                ytemp = yn + int(kd[k, 1])
                if marked[xtemp, ytemp] == 1:
                    queue_list.append(
                        [float(xtemp), float(ytemp), float(dem[xtemp, ytemp])]
                    )
        if queue_list:
            queue = np.array(queue_list, dtype=float)
        else:
            queue = np.empty((0, 3), dtype=float)
    else:
        print("No Orphans Found")
        queue = None

    return {"norphan": norphan, "queue": queue}
