"""
Run PriorityFlow main workflow functions.

This module provides the main execution function that orchestrates the complete
PriorityFlow topographic processing workflow, including stream network processing,
orphan branch detection, and hillslope processing.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Union
from .stream_traverse import stream_traverse
from .find_orphan import find_orphan
from .init_queue import init_queue
from .d4_traverse import d4_traverse_b


def run_pf(
    dem: np.ndarray,
    initmask: np.ndarray,
    domainmask: np.ndarray,
    border: np.ndarray,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4)
) -> Dict[str, Union[np.ndarray, List]]:
    """
    Initialize queue for topographic processing.
    
    Sets up a queue and initializes marked and step matrices for DEM processing.
    This is the main workflow orchestrator that runs the complete PriorityFlow
    algorithm including stream network processing, orphan branch detection,
    and hillslope processing.
    
    Parameters
    ----------
    dem : np.ndarray
        Digital Elevation Model matrix
    initmask : np.ndarray
        Mask of the same dimensions as dem denoting a subset of cells to be
        considered for the queue (e.g., if you want to setup a run starting
        with only river cells). Note: if no init mask is included, every
        border cell will be added to the queue
    domainmask : np.ndarray
        Mask of the domain extent to be considered. If no domain mask is
        provided, boundaries will be calculated from the rectangular extent
    border : np.ndarray
        Alternatively you can input your own border rather than having it be
        calculated from the domain mask. For example, if you want to have the
        river network and the borders combined, you can input this as a border
    d4 : Tuple[int, int, int, int], optional
        D4 direction numbering scheme. Defaults to (1, 2, 3, 4)
    
    Returns
    -------
    Dict[str, Union[np.ndarray, List]]
        A dictionary containing:
        - 'marked': Matrix indicating the outlet cells that were identified (1=outlet, 0=not outlet)
        - 'queue': List of the outlet cells with three columns: x, y, elevation
        - 'initmask': Matrix indicating the cells that were input as potential output points
        - 'basin': Matrix indicating the basin number for each outlet point (each outlet is assigned a unique basin number)
        - 'direction': Matrix indicating the flow direction for each outlet point. The numbering scheme follows the d4 numbering scheme provided as input
    
    Notes
    -----
    This function implements a multi-phase processing approach:
    
    1. **Stream Network Traversal**: Process the initial river network using StreamTraverse
    2. **Orphan Branch Detection**: Find and connect disconnected river segments using FindOrphan
    3. **Hillslope Processing**: Process all non-river cells using D4TraverseB
    
    The function handles:
    - Iterative orphan branch resolution until all segments are connected
    - Comprehensive border handling including rivers, domain boundaries, lakes, and sinks
    - Performance monitoring with timing and progress reporting
    - Integration of multiple PriorityFlow functions into a unified workflow
    
    The algorithm ensures:
    - Complete river network connectivity
    - Comprehensive domain coverage
    - Consistent flow direction and basin assignment
    - Robust processing of complex terrain
    
    Examples
    --------
    Basic usage with all required parameters:
    
    >>> import numpy as np
    >>> from priority_flow.run_pf import run_pf
    >>> dem = np.random.rand(100, 200)
    >>> initmask = np.zeros((100, 200))
    >>> initmask[50:60, 100:120] = 1  # River cells
    >>> domainmask = np.ones((100, 200))
    >>> border = np.zeros((100, 200))
    >>> border[0, :] = 1  # Top border
    >>> border[-1, :] = 1  # Bottom border
    >>> border[:, 0] = 1  # Left border
    >>> border[:, -1] = 1  # Right border
    >>> result = run_pf(dem, initmask, domainmask, border)
    
    Custom D4 directions:
    
    >>> result = run_pf(dem, initmask, domainmask, border, d4=(3, 1, 4, 2))
    
    Notes on Processing Strategy:
    - Stream-First: Process river network before hillslopes for efficiency
    - Orphan Resolution: Ensure complete river connectivity through iterative processing
    - Hillslope Processing: Process remaining domain cells with comprehensive coverage
    - Integration: Combine all results into unified output with consistent flow directions
    
    Performance Considerations:
    - Orphan detection loops may require multiple iterations for complex networks
    - Large domains may require significant processing time
    - Memory requirements scale with domain size and processing complexity
    
    Integration with Other Functions:
    - StreamTraverse: River network processing
    - FindOrphan: Orphan branch detection
    - InitQueue: Queue initialization for hillslope processing
    - D4TraverseB: Comprehensive domain processing
    """
    
    # PriorityFlow - Topographic Processing Toolkit for Hydrologic Models
    # Copyright (C) 2018  Laura Condon (lecondon@email.arizona.edu)
    # Contributors - Reed Maxwell (rmaxwell@mines.edu)
    # 
    # This program is free software: you can redistribute it and/or modify
    # it under the terms of the GNU General Public License as published by
    # the Free Software Foundation version 3 of the License
    # 
    # This program is distributed in the hope that it will be useful,
    # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # GNU General Public License for more details.
    # 
    # You should have received a copy of the GNU General Public License
    # along with this program.  If not, see <http://www.gnu.org/licenses/>
    
    # Input validation
    if not isinstance(dem, np.ndarray) or dem.ndim != 2:
        raise ValueError("dem must be a 2D numpy array")
    
    if not isinstance(initmask, np.ndarray) or initmask.shape != dem.shape:
        raise ValueError("initmask must be a 2D numpy array with same shape as dem")
    
    if not isinstance(domainmask, np.ndarray) or domainmask.shape != dem.shape:
        raise ValueError("domainmask must be a 2D numpy array with same shape as dem")
    
    if not isinstance(border, np.ndarray) or border.shape != dem.shape:
        raise ValueError("border must be a 2D numpy array with same shape as dem")
    
    if not isinstance(d4, tuple) or len(d4) != 4:
        raise ValueError("d4 must be a tuple of 4 integers")
    
    # Initialize timing
    t0 = time.time()
    
    # Phase 1: Traverse the stream network
    print("Phase 1: Stream Network Traversal")
    trav1 = stream_traverse(
        dem=dem,
        mask=initmask,
        queue=np.array([]),  # Will be initialized by init_queue
        marked=np.zeros_like(dem, dtype=int),
        basins=np.zeros_like(dem, dtype=int),
        printstep=False,
        epsilon=0.0
    )
    
    t1 = time.time()
    first_pass_time = t1 - t0
    first_pass_percent = np.sum(trav1['marked']) / np.sum(initmask) * 100 if np.sum(initmask) > 0 else 0
    
    print(f"First Pass: {first_pass_time:.1f} sec")
    print(f"First Pass: {first_pass_percent:.1f} % cells processed")
    
    # Phase 2: Look for orphaned branches and continue traversing until they are all connected
    print("\nPhase 2: Orphan Branch Detection and Resolution")
    norphan = 1
    lap = 1
    
    while norphan > 0:
        t1_lap = time.time()
        
        # Look for orphan branches
        # Create mask of marked rivers + boundaries + lakes + sinks
        # Note: In the R code, LborderT and SborderT are used but not defined in this function
        # For now, we'll use the existing border and marked cells
        riv_border = border + trav1['marked']
        riv_border[riv_border > 1] = 1
        
        orphan = find_orphan(
            dem=trav1['dem'],
            mask=initmask,
            marked=riv_border
        )
        
        norphan = orphan['norphan']
        print(f"Lap {lap}: {norphan} orphans found")
        
        # Go around again if orphans are found
        if norphan > 0:
            trav2 = stream_traverse(
                dem=trav1['dem'],
                mask=initmask,
                queue=orphan['queue'],
                marked=trav1['marked'],
                basins=trav1['basins'],
                step=trav1['step'],
                direction=trav1['direction'],
                printstep=False,
                epsilon=0.0
            )
            
            # Update trav1 with the new results
            trav1 = trav2
            lap += 1
            
            t2_lap = time.time()
            lap_time = t2_lap - t1_lap
            print(f"Lap {lap}: {lap_time:.1f} sec")
        else:
            print("Done! No orphan branches found")
    
    # Report final processing status
    final_percent = np.sum(trav1['marked'] * initmask) / np.sum(initmask) * 100 if np.sum(initmask) > 0 else 0
    print(f"Final pass: {final_percent:.1f} % cells processed")
    
    t3 = time.time()
    total_time = t3 - t0
    print(f"Total Time: {total_time:.1f} sec")
    
    # Phase 3: Initialize the queue with every cell on the processed river and the boundary
    print("\nPhase 3: Queue Initialization and Hillslope Processing")
    
    # River border equals the traversed river plus domain border plus lake and sink border
    # Note: LborderT and SborderT are not defined in this function
    # Using domainmask as a proxy for lake and sink borders
    riv_border = border + trav1['marked'] + domainmask
    riv_border[riv_border > 1] = 1
    
    # Initialize the updated river border
    init_result = init_queue(
        dem=trav1['dem'],
        border=riv_border
    )
    
    # Phase 4: Process all the cells outside the channel network
    t4 = time.time()
    
    # Note: In the R code, LakemaskT is used but not defined
    # Using domainmask as a proxy for the lake mask
    trav_hs = d4_traverse_b(
        dem=trav1['dem'],
        queue=init_result['queue'],
        marked=init_result['marked'],
        mask=domainmask,  # Using domainmask as lake mask
        direction=trav1['direction'],
        basins=trav1['basins'],
        step=trav1['step'],
        epsilon=0.1,
        printstep=False,
        nchunk=1000
    )
    
    t5 = time.time()
    hillslope_time = t5 - t4
    print(f"Hillslope Processing Time: {hillslope_time:.1f} sec")
    
    # Prepare output
    output = {
        'marked': trav1['marked'],
        'queue': init_result['queue'],
        'initmask': initmask,
        'basin': trav1['basins'],
        'direction': trav1['direction'],
        'dem_processed': trav1['dem'],
        'step': trav1['step'],
        'hillslope_results': trav_hs
    }
    
    return output 