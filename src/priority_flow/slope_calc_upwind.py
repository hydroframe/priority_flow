"""
Upwind Slope Calculation functions for PriorityFlow.

This module provides functions to calculate slopes from a DEM using upwind methods
and apply various river processing options including watershed and stream mean slopes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def slope_calc_upwind(
    dem: np.ndarray,
    direction: np.ndarray,
    dx: float,
    dy: float,
    mask: Optional[np.ndarray] = None,
    borders: Optional[np.ndarray] = None,
    borderdir: int = 1,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    minslope: float = 1e-5,
    maxslope: float = -1.0,
    secondary_th: float = -1.0,
    river_method: int = 0,
    river_secondary_th: float = 0.0,
    rivermask: Optional[np.ndarray] = None,
    subbasins: Optional[np.ndarray] = None,
    printflag: bool = False,
    upflag: bool = True
) -> Dict[str, np.ndarray]:
    """
    Calculate the slopes from a DEM using upwind methods.
    
    This function will calculate slopes using upwind methods and apply a range
    of smoothing options including minimum/maximum slope thresholds, secondary
    slope scaling, and river-specific processing methods.
    
    Parameters
    ----------
    dem : np.ndarray
        Digital Elevation Model matrix
    direction : np.ndarray
        Flow direction matrix
    dx : float
        Lateral grid cell resolution in x direction
    dy : float
        Lateral grid cell resolution in y direction
    mask : np.ndarray, optional
        Domain mask matrix. Defaults to processing the complete domain if not provided.
    borders : np.ndarray, optional
        Matrix with 1's for border cells pointing out, 2 for pointing in, and 0 for non-border cells.
    borderdir : int, optional
        Default value for border cells: 1=point out, 2=point in. Defaults to 1.
    d4 : Tuple[int, int, int, int], optional
        D4 direction numbering scheme. Defaults to (1, 2, 3, 4) for (down, left, up, right).
    minslope : float, optional
        Minimum absolute slope value to apply to flat cells if needed. Defaults to 1e-5.
    maxslope : float, optional
        Maximum absolute value of slopes. If set to -1, slopes will not be limited. Defaults to -1.0.
    secondary_th : float, optional
        Secondary threshold - maximum ratio of |secondary|/|primary| to be enforced.
        If set to -1, no scaling will be applied. Defaults to -1.0.
    river_method : int, optional
        Optional method to treat river cells differently from the rest of the domain:
        0: Default value, no special treatment for river cells
        1: Scale secondary slopes along the river (requires river mask and river_secondary_th)
        2: Apply watershed mean slope to each river reach (requires river mask and subbasins)
        3: Apply the stream mean slope to each reach (requires river mask and subbasins)
        Defaults to 0.
    river_secondary_th : float, optional
        Secondary threshold to apply to river cells if river method 1-3 is chosen.
        Maximum ratio of |secondary|/|primary| to be enforced. Defaults to 0.0.
    rivermask : np.ndarray, optional
        Mask with 1 for river cells and 0 for other cells. Required for river methods 1-3.
    subbasins : np.ndarray, optional
        Matrix of subbasin values. Required for river methods 2-3.
    printflag : bool, optional
        Flag to enable debug printing. Defaults to False.
    upflag : bool, optional
        Flag indicating whether slope calc should be upwinded to be consistent with
        the upwinding in the ParFlow OverlandFlow Boundary condition. Defaults to True.
        If set to False, all slopes will be calculated as [i+1]-[i] with no adjustments.
    
    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing:
        - 'slopex': Matrix of slopes in the x direction
        - 'slopey': Matrix of slopes in the y direction
        - 'direction': The adjusted flow direction matrix
    
    Notes
    -----
    This function implements a comprehensive upwind slope calculation algorithm that:
    1. Calculates upwind slopes for consistent flow direction handling
    2. Handles boundary conditions with configurable border directions
    3. Applies minimum and maximum slope thresholds
    4. Enforces secondary slope scaling options
    5. Provides river-specific slope processing methods
    6. Validates slope consistency and quality
    
    The algorithm handles:
    - Upwind slope calculations for flow consistency
    - Border cell slope assignments with directional control
    - Primary flow direction slope identification
    - Slope consistency validation and correction
    - Threshold-based slope adjustments
    - Secondary slope scaling and management
    - River-specific slope processing methods
    - Flat cell detection and correction
    
    River Methods:
    0: Default value, no special treatment for river cells
    1: Scale secondary slopes along the river (requires river mask and river_secondary_th)
    2: Apply watershed mean slope to each river reach (requires river mask and subbasins)
    3: Apply the stream mean slope to each reach (requires river mask and subbasins)
    
    Note: The river mask can be different from the rivers used to create the subbasins
    if desired (e.g., if you want to use a threshold of 100 to create subbasins but
    then apply to river cells with a threshold of 50).
    """
    ny, nx = dem.shape  # Note: R uses ncol/nrow, Python uses shape
    
    # If no mask is provided, default to the rectangular domain
    if mask is None:
        if printflag:
            print("No mask provided, initializing mask for complete domain")
        mask = np.ones((nx, ny))
        borders = np.zeros((nx, ny))
        borders[:, [0, ny-1]] = 1 * borderdir
        borders[[0, nx-1], :] = 1 * borderdir
    
    # If no border is provided, create a border with everything pointing in or out according to borderdir
    if borders is None:
        borders = np.ones((nx, ny))
        borders[1:nx-1, 1:ny-1] = (mask[0:nx-2, 1:ny-1] + 
                                   mask[2:nx, 1:ny-1] + 
                                   mask[1:nx-1, 0:ny-2] + 
                                   mask[1:nx-1, 2:ny])
        borders = borders * mask
        borders[(borders < 4) & (borders != 0)] = 1 * borderdir
        borders[borders == 4] = 0
    
    # Assign NA values to the DEM outside the mask
    inmask = np.where(mask == 1)
    dem_mask = dem.copy()
    dem_mask[~mask.astype(bool)] = np.nan
    
    # First pass: calculate the x and y slopes as (i+1)-(i)
    # slopex = dem[i+1,j] - dem[i,j]
    # slopey = dem[i,j+1] - dem[i,j]
    slopex1 = np.full((nx, ny), np.nan)
    slopey1 = np.full((nx, ny), np.nan)
    slopex1[0:nx-1, :] = (dem_mask[1:nx, :] - dem_mask[0:nx-1, :]) / dx
    slopex1[nx-1, :] = slopex1[nx-2, :]
    slopey1[:, 0:ny-1] = (dem_mask[:, 1:ny] - dem_mask[:, 0:ny-1]) / dy
    slopey1[:, ny-1] = slopey1[:, ny-2]
    
    # Assign slopes based on upwinding for all non-border cells
    if upflag:
        if printflag:
            print("upwinding slopes")
        
        slopex2 = slopex1.copy()
        slopey2 = slopey1.copy()
        
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                if mask[i, j] == 1 and borders[i, j] == 0:
                    # X-Direction
                    ###################
                    # SINK: If slopex[i-1]<0 & slopex[i]>0
                    # Assign slope=0
                    if slopex1[i-1, j] < 0 and slopex1[i, j] > 0:
                        slopex2[i, j] = 0
                        if direction[i, j] == d4[1] or direction[i, j] == d4[3]:
                            if printflag:
                                print(f"Problem! Local X sink found in primary direction. x={i} j={j}")
                    
                    # Local Maximum: If slopex[i-1]>0 & slopex[i]<0
                    # If the primary flow direction is in the x
                    # Then assign the slope consistent with this.
                    # If not, choose the maximum absolute value
                    if slopex1[i-1, j] >= 0 and slopex1[i, j] <= 0:
                        # If the primary direction is left, use the i-1 slope
                        if direction[i, j] == d4[1]:
                            slopex2[i, j] = slopex1[i-1, j]
                        elif direction[i, j] == d4[3]:
                            # If the primary direction is right, use the i slope
                            slopex2[i, j] = slopex1[i, j]
                        else:
                            # If the primary direction is in the y, then just choose the steepest one
                            if abs(slopex1[i-1, j]) > abs(slopex1[i, j]):
                                slopex2[i, j] = slopex1[i-1, j]
                    
                    # Flow Through from right: If slopex[i-1]>0 & slopex[i]>0
                    if slopex1[i-1, j] >= 0 and slopex1[i, j] >= 0:
                        slopex2[i, j] = slopex1[i-1, j]
                    
                    # NOTE: Nothing to do for flow through from left
                    # This would get assigned slopex[i,j] which is how slopex2 is initialized
                    
                    # Y-Direction
                    ###################
                    # SINK: If slopey[j-1]<0 & slopey[j]>0
                    # Assign slope=0
                    if slopey1[i, j-1] < 0 and slopey1[i, j] > 0:
                        slopey2[i, j] = 0
                        if direction[i, j] == d4[0] or direction[i, j] == d4[2]:
                            if printflag:
                                print(f"Problem! Local Y sink found in primary direction. x={i} j={j}")
                    
                    # Local Maximum: If slopey[j-1]>0 & slopey[j]<0
                    # If the primary flow direction is in the y
                    # Then assign the slope consistent with this.
                    # If not, choose the maximum absolute value
                    if slopey1[i, j-1] >= 0 and slopey1[i, j] <= 0:
                        # If the primary direction is down, use the j-1 slope
                        if direction[i, j] == d4[0]:
                            slopey2[i, j] = slopey1[i, j-1]
                        elif direction[i, j] == d4[2]:
                            # If the primary direction is up, use the j slope
                            slopey2[i, j] = slopey1[i, j]
                        else:
                            # If the primary direction is in the x, then just choose the steepest one
                            if abs(slopey1[i, j-1]) > abs(slopey1[i, j]):
                                slopey2[i, j] = slopey1[i, j-1]
                    
                    # Flow Through from top: If slopey[j-1]>0 & slopey[j]>0
                    if slopey1[i, j-1] >= 0 and slopey1[i, j] >= 0:
                        slopey2[i, j] = slopey1[i, j-1]
                    
                    # NOTE: Nothing to do for flow through from bottom
                    # This would get assigned slopey[i,j] which is how slopey2 is initialized
    else:
        # If you are not using the upwinded slopes, just use the [i+1]-i calculations
        if printflag:
            print("standard slope calc")
        slopex2 = slopex1.copy()
        slopey2 = slopey1.copy()
    
    ###################
    # Assign flow directions and slopes for border cells
    bordi = borders.copy()  # Making a border indicator file with all 1's for any border cell
    bordi[bordi > 1] = 1
    interior = mask - bordi  # Mask of non-boundary cells inside the domain
    
    # Find the left border cells
    border_l = np.zeros((nx, ny))
    border_l[0, :] = 1
    border_l2 = border_l.copy()
    # Looking for cells where mask[i-1]=0 and mask[i]=1 (mask[2:nx]+mask[1:nx-1]=1)
    # Filtering out cells where there isn't a cell to the right to calculate the slope from (i.e., *interior[3:nx])
    border_l[1:nx-1, :] = (mask[1:nx-1, :] + mask[0:nx-2, :]) * interior[2:nx, :]
    border_l = border_l * bordi  # Getting rid of any edge not on the border
    border_l[border_l > 1] = 0
    # Second version allows for any neighbor in the mask for slope calc (i.e., not just non-border cells)
    border_l2[1:nx-1, :] = (mask[1:nx-1, :] + mask[0:nx-2, :]) * mask[2:nx, :]
    border_l2 = border_l2 * bordi  # Getting rid of any edge not on the border
    border_l2[border_l2 > 1] = 0
    # Left borders get value of 2
    border_l = border_l * 2
    border_l2 = border_l2 * 2
    
    # Find the right border cells
    border_r = np.zeros((nx, ny))
    border_r[nx-1, :] = 1
    border_r2 = border_r.copy()
    # Looking for cells where mask[i+1]=0 and mask[i]=1 (mask[2:nx]+mask[1:nx-1]=1)
    # Filtering out cells where there isn't a cell to the left to calculate the slope from (i.e., *interior[1:nx-2])
    border_r[1:nx-1, :] = (mask[1:nx-1, :] + mask[2:nx, :]) * interior[0:nx-2, :]
    border_r = border_r * bordi  # Getting rid of any edge not on the border
    border_r[border_r > 1] = 0
    # Second version allows for any neighbor in the mask for slope calc (i.e., not just non-border cells)
    border_r2[1:nx-1, :] = (mask[1:nx-1, :] + mask[2:nx, :]) * mask[0:nx-2, :]
    border_r2 = border_r2 * bordi  # Getting rid of any edge not on the border
    border_r2[border_r2 > 1] = 0
    # Right borders get value of 4
    border_r = border_r * 4
    border_r2 = border_r2 * 4
    
    # Find the lower (bottom) border cells
    border_b = np.zeros((nx, ny))
    border_b[:, 0] = 1
    border_b2 = border_b.copy()
    # Looking for cells where mask[j-1]=0 and mask[j]=1 (mask[2:ny]+mask[1:ny-1]=1)
    # Filtering out cells where there isn't a cell above to calculate the slope from
    border_b[:, 1:ny-1] = (mask[:, 1:ny-1] + mask[:, 0:ny-2]) * interior[:, 2:ny]
    border_b = border_b * bordi  # Getting rid of any edge not on the border
    border_b[border_b > 1] = 0
    # Second version allows for any neighbor in the mask for slope calc (i.e., not just non-border cells)
    border_b2[:, 1:ny-1] = (mask[:, 1:ny-1] + mask[:, 0:ny-2]) * mask[:, 2:ny]
    border_b2 = border_b2 * bordi  # Getting rid of any edge not on the border
    border_b2[border_b2 > 1] = 0
    # Bottom borders get value of 1
    border_b = border_b * 1
    border_b2 = border_b2 * 1
    
    # Find the upper (top) border cells
    border_t = np.zeros((nx, ny))
    border_t[:, ny-1] = 1
    border_t2 = border_t.copy()
    # Looking for cells where mask[j+1]=0 and mask[j]=1 (mask[2:ny]+mask[1:ny-1]=1)
    # Filtering out cells where there isn't a cell below to calculate the slope from
    border_t[:, 1:ny-1] = (mask[:, 1:ny-1] + mask[:, 2:ny]) * interior[:, 0:ny-2]
    border_t = border_t * bordi  # Getting rid of any edge not on the border
    border_t[border_t > 1] = 0
    # Second version allows for any neighbor in the mask for slope calc (i.e., not just non-border cells)
    border_t2[:, 1:ny-1] = (mask[:, 1:ny-1] + mask[:, 2:ny]) * mask[:, 0:ny-2]
    border_t2 = border_t2 * bordi  # Getting rid of any edge not on the border
    border_t2[border_t2 > 1] = 0
    # Top borders get value of 3
    border_t = border_t * 3
    border_t2 = border_t2 * 3
    
    # Assign the edge face to be calculated as the max of the borders. In most cases there is just one
    # non-zero border so this is not really a choice
    # In cases where there is more than one boundary, it will have an arbitrary preference in the following order - Right, Top, Left, Bottom
    borlist = np.where(bordi == 1)
    borlisti = np.where(bordi == 1)[0]
    bordsum = border_b + border_t + border_l + border_r
    borddir = np.zeros((nx, ny))
    
    for k in range(len(borlist[0])):
        i = borlist[0][k]
        j = borlist[1][k]
        # If you can choose a border edge with an internal cell opposite, do this first
        if bordsum[i, j] > 0:
            borddir[i, j] = max(border_b[i, j], border_t[i, j], border_l[i, j], border_r[i, j])
        # If not, resort to picking an edge with a boundary cell opposite (this mostly happens on corners where both the d4 neighbors are also borders)
        else:
            borddir[i, j] = max(border_b2[i, j], border_t2[i, j], border_l2[i, j], border_r2[i, j])
    
    test = np.sum((borddir - bordi) < 0)
    missinglist = np.where((borddir - bordi) < 0)[0]
    missinglist_a = np.where((borddir - bordi) < 0)
    
    # These errors occur with sinks or some special cases where there aren't ANY
    # usable neighbors. These cells get assigned an arbitrary direction and minimum slope below
    
    # Calculate the number of cells draining to any cell
    down = np.zeros((nx, ny))
    up = np.zeros((nx, ny))
    left = np.zeros((nx, ny))
    right = np.zeros((nx, ny))
    down[np.where(direction == d4[0])] = 1
    left[np.where(direction == d4[1])] = 1
    up[np.where(direction == d4[2])] = 1
    right[np.where(direction == d4[3])] = 1
    
    draincount = np.zeros((nx, ny))
    draincount[:, 0:ny-1] = draincount[:, 0:ny-1] + down[:, 1:ny]
    draincount[:, 1:ny] = draincount[:, 1:ny] + up[:, 0:ny-1]
    draincount[0:nx-1, :] = draincount[0:nx-1, :] + left[1:nx, :]
    draincount[1:nx, :] = draincount[1:nx, :] + right[0:nx-1, :]
    
    # If a cell on the border has another interior cell draining to it, then it should be pointing out
    drainout = np.where(draincount[borlisti] > 0)[0]
    borders[borlisti[drainout]] = 1
    
    # Calculate the slopes
    for k in range(len(borlist[0])):
        i = borlist[0][k]
        j = borlist[1][k]
        
        # Border on the top
        if borddir[i, j] == 3:
            slopex2[i, j] = 0
            # If it's pointing out, make slopey negative
            if borders[i, j] == 1:
                slopey2[i, j] = -abs(slopey1[i, j-1])
                direction[i, j] = 3
            else:
                # If it's pointing in, make slopey positive
                slopey2[i, j] = abs(slopey1[i, j-1])
                direction[i, j] = 1
        
        # Border on the right
        if borddir[i, j] == 4:
            slopey2[i, j] = 0
            # If it's pointing out, make slopex negative
            if borders[i, j] == 1:
                slopex2[i, j] = -abs(slopex1[i-1, j])
                direction[i, j] = 4
            else:
                # If it's pointing in, make slopex positive
                slopex2[i, j] = abs(slopex1[i-1, j])
                direction[i, j] = 2
        
        # Border on the bottom
        if borddir[i, j] == 1:
            slopex2[i, j] = 0
            # If it's pointing out, make slopey positive
            if borders[i, j] == 1:
                slopey2[i, j] = abs(slopey1[i, j])
                direction[i, j] = 1
            else:
                # If it's pointing in, make slopey negative
                slopey2[i, j] = -abs(slopey1[i, j])
                direction[i, j] = 3
        
        # Border on the left
        if borddir[i, j] == 2:
            slopey2[i, j] = 0
            # If it's pointing out, make slopex positive
            if borders[i, j] == 1:
                slopex2[i, j] = abs(slopex1[i, j])
                direction[i, j] = 2
            else:
                # If it's pointing in, make slopex negative
                slopex2[i, j] = -abs(slopex1[i, j])
                direction[i, j] = 4
    
    # Filling in the missing values with an arbitrary direction
    # Need to circle back to this
    direction[missinglist] = 4
    slopex2[missinglist] = -minslope
    slopey2[missinglist] = 0
    
    # Make lists of the primary directions now that directions are all filled in
    downlist = np.where(direction == d4[0])[0]
    leftlist = np.where(direction == d4[1])[0]
    uplist = np.where(direction == d4[2])[0]
    rightlist = np.where(direction == d4[3])[0]
    ylist = np.concatenate([uplist, downlist])  # List of cells with primary flow in the y direction
    xlist = np.concatenate([rightlist, leftlist])  # List of cells with primary flow in the x direction
    
    ymask = np.zeros((nx, ny))  # Mask of cells with primary flow in x and y direction, signs indicate direction
    xmask = np.zeros((nx, ny))
    ymask[downlist] = 1
    ymask[uplist] = -1
    xmask[leftlist] = 1
    xmask[rightlist] = -1
    
    ###################################
    # If an upper limit on slopes is set (i.e., maxslope is positive)
    if maxslope >= 0:
        if printflag:
            print(f"Limiting slopes to maximum absolute value of {maxslope}")
        
        # x slopes
        xclip_p = np.where(slopex2 > maxslope)[0]
        slopex2[xclip_p] = maxslope
        xclip_n = np.where(slopex2 < -maxslope)[0]
        slopex2[xclip_n] = -maxslope
        
        # y slopes
        yclip_p = np.where(slopey2 > maxslope)[0]
        slopey2[yclip_p] = maxslope
        yclip_n = np.where(slopey2 < -maxslope)[0]
        slopey2[yclip_n] = -maxslope
    
    ###################################
    # If a maximum secondary/primary slope ratio is set (i.e., secondary_th >= 0)
    if secondary_th >= 0:
        if printflag:
            print(f"Limiting the ratio of secondary to primary slopes to {secondary_th}")
        
        # Make matrices of primary and secondary slopes and ratios
        primary = np.abs(slopex2)
        primary[ylist] = np.abs(slopey2[ylist])
        secondary = np.abs(slopey2)
        secondary[ylist] = np.abs(slopex2[ylist])
        ratio = secondary / primary
        
        scalelist = np.where(ratio > secondary_th)[0]
        for i in range(len(scalelist)):
            temp = scalelist[i]
            # If primary direction is in x, then scale the y slopes
            if direction[temp] == d4[1] or direction[temp] == d4[3]:
                slopey2[temp] = np.sign(slopey2[temp]) * secondary_th * abs(slopex2[temp])
            # If primary direction is in y, then scale the x slopes
            if direction[temp] == d4[0] or direction[temp] == d4[2]:
                slopex2[temp] = np.sign(slopex2[temp]) * secondary_th * abs(slopey2[temp])
    
    ###################################
    # Separate processing for the river cells
    # Option 1: Just turn off secondary slopes in the river cells
    if river_method == 1:
        if rivermask is None:
            if printflag:
                print("WARNING: No rivermask provided, slopes not adjusted")
        else:
            if printflag:
                print("River Method 1: setting secondary slopes to zero along the river")
                print(f"Scaling secondary slopes along river mask to {river_secondary_th} * primary slope")
            
            # Otherwise scale by whatever river secondary threshold is provided
            xtemp = slopex2.copy()
            xtemp_scaled = slopey2 * river_secondary_th  # x slopes = y slopes * scaler
            xtemp[ylist] = xtemp_scaled[ylist]  # Where primary flow is in the y direction, fill in x slopes with the scaled values
            
            # Repeat in y direction
            ytemp = slopey2.copy()
            ytemp_scaled = slopex2 * river_secondary_th  # y slopes = x slopes * scaler
            ytemp[xlist] = ytemp_scaled[xlist]
            
            # Merge in these slopes over the river mask
            rivlist = np.where(rivermask == 1)[0]
            slopex2[rivlist] = xtemp[rivlist]
            slopey2[rivlist] = ytemp[rivlist]
    
    # Option 2: Assign the subbasin mean slope to the river cells in the primary direction and set the secondary direction to 0
    if river_method == 2:
        if printflag:
            print("River Method 2: assigning average watershed slope to river cells by watershed")
            print(f"Scaling secondary slopes along river mask to {river_secondary_th} * primary slope")
        
        nbasin = int(np.max(subbasins))
        savg = np.zeros(nbasin)
        count = np.zeros(nbasin)
        
        # Get the signs of the slopes before averaging to use in the secondary slope scaling
        xsign = np.sign(slopex2)
        ysign = np.sign(slopey2)
        xsign[xsign == 0] = 1  # For zero slopes, default to positive
        ysign[ysign == 0] = 1  # For zero slopes, default to positive
        
        # Calculate the subbasin average slope
        # NEED to sort out for x list and ylist so just averaging the primary
        for i in range(nx):
            for j in range(ny):
                if subbasins[i, j] > 0:
                    savg[int(subbasins[i, j])-1] = (savg[int(subbasins[i, j])-1] + 
                                                   abs(slopex2[i, j]) * abs(xmask[i, j]) + 
                                                   abs(slopey2[i, j]) * abs(ymask[i, j]))
                    count[int(subbasins[i, j])-1] = count[int(subbasins[i, j])-1] + 1
        
        savg = savg / count
        for b in range(nbasin):
            savg[b] = max(minslope, savg[b])  # Make sure the average slope is >= minslope
        
        savg[np.isnan(savg)] = minslope
        
        rivlist = np.where(rivermask == 1)[0]
        nriv = len(rivlist)
        # Fill in the river slopes with the average for their subbasin
        for i in range(nriv):
            rtemp = rivlist[i]
            sbtemp = int(subbasins[rtemp // nx, rtemp % nx]) - 1  # Convert linear index to 2D
            if sbtemp >= 0:
                # Setting the primary slopes along the river (xmask and y masks are masks of primary flow direction so secondary will be set to zero)
                slopex2[rtemp] = savg[sbtemp] * xmask[rtemp // nx, rtemp % nx]
                slopey2[rtemp] = savg[sbtemp] * ymask[rtemp // nx, rtemp % nx]
                
                # Setting the secondary slopes along the river
                # By taking the inverse of the mask and adding the scale slope to the initial slope
                # This means it will add zero if it's the primary flow direction and it will add the scaled secondary if not
                slopex2[rtemp] = (slopex2[rtemp] + 
                                 savg[sbtemp] * abs(1 - abs(xmask[rtemp // nx, rtemp % nx])) * 
                                 xsign[rtemp // nx, rtemp % nx] * river_secondary_th)
                slopey2[rtemp] = (slopey2[rtemp] + 
                                 savg[sbtemp] * abs(1 - abs(ymask[rtemp // nx, rtemp % nx])) * 
                                 ysign[rtemp // nx, rtemp % nx] * river_secondary_th)
    
    # Option 3: Assign the subbasin mean river slope to the river cells in the primary direction and set the secondary direction to 0
    if river_method == 3:
        if printflag:
            print("River Method 3: assigning average river slope to river cells by watershed")
            print(f"Scaling secondary slopes along river mask to {river_secondary_th} * primary slope")
        
        nbasin = int(np.max(subbasins))
        savg = np.zeros(nbasin)
        count = np.zeros(nbasin)
        
        # Get the signs of the slopes before averaging to use in the secondary slope scaling
        xsign = np.sign(slopex2)
        ysign = np.sign(slopey2)
        xsign[xsign == 0] = 1  # For zero slopes, default to positive
        ysign[ysign == 0] = 1  # For zero slopes, default to positive
        
        # Calculate the subbasin average slope
        # NEED to sort out for x list and ylist so just averaging the primary
        for i in range(nx):
            for j in range(ny):
                if subbasins[i, j] > 0 and rivermask[i, j] == 1:
                    savg[int(subbasins[i, j])-1] = (savg[int(subbasins[i, j])-1] + 
                                                   abs(slopex2[i, j]) * abs(xmask[i, j]) + 
                                                   abs(slopey2[i, j]) * abs(ymask[i, j]))
                    count[int(subbasins[i, j])-1] = count[int(subbasins[i, j])-1] + 1
        
        savg = savg / count
        for b in range(nbasin):
            savg[b] = max(minslope, savg[b])  # Make sure the average slope is >= minslope
        
        savg[np.isnan(savg)] = minslope
        
        rivlist = np.where(rivermask == 1)[0]
        nriv = len(rivlist)
        # Fill in the river slopes with the average for their subbasin
        for i in range(nriv):
            rtemp = rivlist[i]
            sbtemp = int(subbasins[rtemp // nx, rtemp % nx]) - 1  # Convert linear index to 2D
            if sbtemp >= 0:
                # Setting the primary slopes along the river (xmask and y masks are masks of primary flow direction so secondary will be set to zero)
                slopex2[rtemp] = savg[sbtemp] * xmask[rtemp // nx, rtemp % nx]
                slopey2[rtemp] = savg[sbtemp] * ymask[rtemp // nx, rtemp % nx]
                
                # Setting the secondary slopes along the river
                # By taking the inverse of the mask and adding the scale slope to the initial slope
                # This means it will add zero if it's the primary flow direction and it will add the scaled secondary if not
                slopex2[rtemp] = (slopex2[rtemp] + 
                                 savg[sbtemp] * abs(1 - abs(xmask[rtemp // nx, rtemp % nx])) * 
                                 xsign[rtemp // nx, rtemp % nx] * river_secondary_th)
                slopey2[rtemp] = (slopey2[rtemp] + 
                                 savg[sbtemp] * abs(1 - abs(ymask[rtemp // nx, rtemp % nx])) * 
                                 ysign[rtemp // nx, rtemp % nx] * river_secondary_th)
    
    ###################################
    # Check for flat cells
    nflat = np.sum((slopex2 == 0) & (slopey2 == 0))
    if nflat != 0:
        if printflag:
            print(f"WARNING: {nflat} Flat cells found")
        flatloc = np.where((slopex2 == 0) & (slopey2 == 0))
        flatlist = np.where((slopex2 == 0) & (slopey2 == 0))[0]
        if printflag:
            print("Flat locations (note this is x,y)")
            print(flatloc)
        
        # Impose a minimum slope in the primary direction for flat cells
        for i in range(nflat):
            dtemp = direction[flatlist[i] // nx, flatlist[i] % nx]
            if dtemp == d4[0]:
                slopey2[flatlist[i] // nx, flatlist[i] % nx] = minslope  # Down
            if dtemp == d4[1]:
                slopex2[flatlist[i] // nx, flatlist[i] % nx] = minslope  # Left
            if dtemp == d4[2]:
                slopey2[flatlist[i] // nx, flatlist[i] % nx] = -minslope  # Up
            if dtemp == d4[3]:
                slopex2[flatlist[i] // nx, flatlist[i] % nx] = -minslope  # Right
        
        nflat = np.sum((slopex2 == 0) & (slopey2 == 0))
        if printflag:
            print(f"After processing: {nflat} Flat cells left")
    
    ###################################
    # If a lower limit on slopes is set (i.e., minslope is greater than zero)
    if minslope >= 0:
        if printflag:
            print(f"Limiting slopes to minimum {minslope}")
        
        # x slopes
        xclip_p = np.where((slopex2 < minslope) & (slopex2 > 0) & (xmask == 1))[0]
        slopex2[xclip_p // nx, xclip_p % nx] = minslope
        xclip_n = np.where((slopex2 > -minslope) & (slopex2 < 0) & (xmask == 1))[0]
        slopex2[xclip_n // nx, xclip_n % nx] = -minslope
        
        # y slopes
        yclip_p = np.where((slopey2 < minslope) & (slopey2 > 0) & (ymask == 1))[0]
        slopey2[yclip_p // nx, yclip_p % nx] = minslope
        yclip_n = np.where((slopey2 > -minslope) & (slopey2 < 0) & (ymask == 1))[0]
        slopey2[yclip_n // nx, yclip_n % nx] = -minslope
    
    ###########################
    # Replace the NA's with 0s
    nax = np.where(np.isnan(slopex2))[0]
    nay = np.where(np.isnan(slopey2))[0]
    slopex2[nax] = 0
    slopey2[nay] = 0
    
    output_list = {
        "slopex": slopex2,
        "slopey": slopey2,
        "direction": direction
    }
    
    return output_list