"""
Standard Slope Calculation functions for PriorityFlow.

This module provides functions to calculate slopes from a DEM using standard
finite difference methods with various smoothing and threshold options.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def slope_calc_standard(
    dem: np.ndarray,
    direction: np.ndarray,
    dx: float,
    dy: float,
    mask: Optional[np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    minslope: float = 0.0,
    maxslope: float = -1.0,
    secondary_th: float = -1.0,
    printflag: bool = False
) -> Dict[str, Union[np.ndarray, List]]:
    """
    Calculate the slopes from a DEM using standard finite difference methods.
    
    This function will calculate slopes using standard methods and apply a range
    of smoothing options including minimum/maximum slope thresholds and secondary
    slope scaling.
    
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
    d4 : Tuple[int, int, int, int], optional
        D4 direction numbering scheme. Defaults to (1, 2, 3, 4) for (down, left, up, right).
    minslope : float, optional
        Minimum absolute slope value to apply to flat cells if needed. Defaults to 0.0.
    maxslope : float, optional
        Maximum absolute value of slopes. If set to -1, slopes will not be limited. Defaults to -1.0.
    secondary_th : float, optional
        Secondary threshold - maximum ratio of |secondary|/|primary| to be enforced.
        If set to -1, no scaling will be applied.
        If set to 0, all secondary slopes will be zero.
        Defaults to -1.0.
    printflag : bool, optional
        Flag to enable debug printing. Defaults to False.
    
    Returns
    -------
    Dict[str, Union[np.ndarray, List]]
        A dictionary containing:
        - 'slopex': Matrix of slopes in the x direction
        - 'slopey': Matrix of slopes in the y direction
        - 'direction': The input flow direction matrix
        - 'Sinks': List of cells where slope directions were inconsistent with flow directions
    
    Notes
    -----
    This function implements a comprehensive slope calculation algorithm that:
    1. Calculates standard finite difference slopes
    2. Handles boundary conditions and edge cases
    3. Applies minimum and maximum slope thresholds
    4. Enforces secondary slope scaling options
    5. Validates slope consistency with flow directions
    6. Identifies and reports flat cells
    
    The algorithm handles:
    - Face-centered slope calculations
    - Border cell slope assignments
    - Primary flow direction slope identification
    - Slope consistency validation
    - Threshold-based slope adjustments
    - Secondary slope scaling
    - Flat cell detection and reporting
    
    River Methods (not yet implemented but documented):
    0: Default value, no special treatment for river cells
    1: Scale secondary slopes along the river (requires river mask and river_secondaryTH)
    2: Apply watershed mean slope to each river reach (requires river mask and subbasins)
    3: Apply the stream mean slope to each reach (requires river mask and subbasins)
    """
    ny, nx = dem.shape  # Note: R uses ncol/nrow, Python uses shape
    
    # If no mask is provided, default to the rectangular domain
    if mask is None:
        if printflag:
            print("No mask provided, initializing mask for complete domain")
        mask = np.ones((nx, ny))
        borders = np.zeros((nx, ny))
    
    # Identify the border cells
    borders = np.ones((nx, ny))
    borders[1:nx-1, 1:ny-1] = (mask[0:nx-2, 1:ny-1] + 
                                mask[2:nx, 1:ny-1] + 
                                mask[1:nx-1, 0:ny-2] + 
                                mask[1:nx-1, 2:ny])
    borders = borders * mask
    borders[(borders < 4) & (borders != 0)] = 1
    borders[borders == 4] = 0
    
    bordi = borders.copy()  # Making a border indicator file with all 1's for any border cell
    bordi[bordi > 1] = 1
    bordlist = np.where(borders > 0)
    
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
    slopey1[:, 0:ny-1] = (dem_mask[:, 1:ny] - dem_mask[:, 0:ny-1]) / dy
    
    # Assign slopes for the upper and right border cells (i.e., where slopes can't be calculated)
    # Look for any upper and left borders where the slope couldn't be calculated and
    # repeat the i-1 or j-1 slope
    slopex2 = slopex1.copy()
    slopey2 = slopey1.copy()
    
    ### Right border
    border_r = np.zeros((nx, ny))
    slope_r = np.zeros((nx, ny))
    
    # Turn the NA's into zeros for all the right border cells
    border_r[0:nx-1, :] = (mask[0:nx-1, :] + mask[1:nx, :]) * mask[0:nx-1, :]
    border_r[nx-1, :] = 1
    border_r = border_r * bordi
    r_list = np.where(border_r == 1)
    slopex2[r_list] = 0
    
    # Find the right border cells that also have a cell to their left in the mask
    border_r2 = np.zeros((nx, ny))
    border_r2[nx-1, :] = 1
    border_r2[1:nx-1, :] = (mask[1:nx-1, :] + mask[2:nx, :]) * mask[0:nx-2, :]
    border_r2 = border_r2 * bordi  # Getting rid of any edge not on the border
    border_r2[border_r2 > 1] = 0
    
    # If there is a cell to the left to use then fill in the slope for the border cell with this value
    slope_r[1:nx, :] = slopex2[0:nx-1, :] * border_r2[1:nx, :]
    slope_r[np.isnan(slope_r)] = 0
    slopex2 = slopex2 + slope_r
    
    ### Top border
    border_t = np.zeros((nx, ny))
    slope_t = np.zeros((nx, ny))
    
    # Turn the NA's into zeros for all the top border cells
    border_t[:, 0:ny-1] = (mask[:, 0:ny-1] + mask[:, 1:ny]) * mask[:, 0:ny-1]
    border_t[:, ny-1] = 1
    border_t = border_t * bordi
    t_list = np.where(border_t == 1)
    slopey2[t_list] = 0
    
    # Find the top border cells that also have a cell below them in the mask
    border_t2 = np.zeros((nx, ny))
    border_t2[:, ny-1] = 1
    border_t2[:, 1:ny-1] = (mask[:, 1:ny-1] + mask[:, 2:ny]) * mask[:, 0:ny-2]
    border_t2 = border_t2 * bordi  # Getting rid of any edge not on the border
    border_t2[border_t2 > 1] = 0
    
    # If there is a cell below to use then fill in the slope for the border cell with this value
    slope_t[:, 1:ny] = slopey2[:, 0:ny-1] * border_t2[:, 1:ny]
    slope_t[np.isnan(slope_t)] = 0
    slopey2 = slopey2 + slope_t
    
    # Make masks of the primary flow directions
    # Make lists of the primary directions now that directions are all filled in
    downlist = np.where(direction == d4[0])[0]
    leftlist = np.where(direction == d4[1])[0]
    uplist = np.where(direction == d4[2])[0]
    rightlist = np.where(direction == d4[3])[0]
    
    downlist_arr = np.where(direction == d4[0])
    leftlist_arr = np.where(direction == d4[1])
    uplist_arr = np.where(direction == d4[2])
    rightlist_arr = np.where(direction == d4[3])
    
    # Masks of which cells contain primary flow direction slope calculations
    # Because the slopes are face centered - a primary flow direction up or right will mean that the slope for
    # that cell is a primary slope
    # However a flow direction down or left indicates that the slope of the i-1 (or j-1) cell is a primary slope
    # Thus some cells may have primary slopes in x and y while some may have them in neither even though every 
    # grid cell only has one direction.
    ymask = np.zeros((nx, ny))
    xmask = np.zeros((nx, ny))
    
    # Mask of cells with primary flow in x and y direction,
    # signs indicate sign of the slope consistent - i.e., for flow in the positive x direction (right) you need a negative slope
    ymask[uplist_arr] = -1
    xmask[rightlist_arr] = -1
    
    if len(leftlist_arr[0]) > 0:
        for ii in range(len(leftlist_arr[0])):
            xindex = max((leftlist_arr[0][ii] - 1), 0)  # max statement means that if you have a left facing cell on the left border of the domain keep the slope at [i] as primary
            if mask[xindex, leftlist_arr[1][ii]] == 0:
                xindex = xindex + 1  # if the left cell falls outside the mask use the current cell for this border
            xmask[xindex, leftlist_arr[1][ii]] = 1
    
    if len(downlist_arr[0]) > 0:
        for ii in range(len(downlist_arr[0])):
            yindex = max((downlist_arr[1][ii] - 1), 0)  # max statement means that if you have a down facing cell on the lower border of the domain keep the slope at [i] as primary
            if mask[downlist_arr[0][ii], yindex] == 0:
                yindex = yindex + 1  # if the lower cell falls outside the mask use the current cell for this border
            ymask[downlist_arr[0][ii], yindex] = 1
    
    ylist = np.where(ymask != 0)  # primary flow direction y slope calculations
    xlist = np.where(xmask != 0)  # primary flow direction x slope calculations
    
    # Do a check to see that slope directions are consistent with flow directions for primary flows
    fix_px = np.where((np.sign(slopex2) == -1) & (xmask == 1))[0]
    slopex2[fix_px] = np.abs(slopex2[fix_px])
    fix_nx = np.where((np.sign(slopex2) == 1) & (xmask == -1))[0]
    slopex2[fix_nx] = -np.abs(slopex2[fix_nx])
    fix_py = np.where((np.sign(slopey2) == -1) & (ymask == 1))[0]
    slopey2[fix_py] = np.abs(slopey2[fix_py])
    fix_ny = np.where((np.sign(slopey2) == 1) & (ymask == -1))[0]
    slopey2[fix_ny] = -np.abs(slopey2[fix_ny])
    
    sinklist = np.concatenate([fix_px, fix_nx, fix_py, fix_ny])
    
    # If a lower limit on slopes is set (i.e., minslope is greater than zero)
    # Then apply the minimum slope threshold to any primary flow direction slope
    # while maintaining the direction consistent with the flow direction file
    if minslope >= 0:
        if printflag:
            print(f"Limiting slopes to minimum {minslope}")
        
        # x slopes
        xclip_p = np.where((np.abs(slopex2) < minslope) & (xmask == 1))[0]
        slopex2[xclip_p] = minslope
        xclip_n = np.where((np.abs(slopex2) < minslope) & (xmask == -1))[0]
        slopex2[xclip_n] = -minslope
        
        # y slopes
        yclip_p = np.where((np.abs(slopey2) < minslope) & (ymask == 1))[0]
        slopey2[yclip_p] = minslope
        yclip_n = np.where((np.abs(slopey2) < minslope) & (ymask == -1))[0]
        slopey2[yclip_n] = -minslope
        
        if printflag:
            print(f'min adjustment: x+={len(xclip_p)}, x-={len(xclip_n)}, y+={len(yclip_p)}, y-={len(yclip_n)}')
    
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
    
    # If a maximum secondary/primary slope ratio is set (i.e., secondary_th >= 0)
    if secondary_th >= 0:
        if secondary_th == 0:
            if printflag:
                print(f"Limiting the ratio of secondary to primary slopes to {secondary_th}")
            slopex2[~np.isin(np.arange(slopex2.size), xlist)] = 0
            slopey2[~np.isin(np.arange(slopey2.size), ylist)] = 0
        else:
            if printflag:
                print("Options for nonzero secondary scaling not currently available, please set secondary_th to -1 or 0")
    
    # Check for flat cells
    # A cell is only flat if all 4 faces of the cell are flat
    flattest = (np.abs(slopex2[1:nx, 1:ny]) + 
                np.abs(slopex2[0:nx-1, 1:ny]) + 
                np.abs(slopey2[1:nx, 1:ny]) + 
                np.abs(slopey2[1:nx, 0:ny-1]))
    
    nflat = np.sum((flattest == 0) & mask[1:nx, 1:ny])
    flats = np.where(flattest == 0)
    if len(flats[0]) > 0:
        flats = (flats[0] + 1, flats[1] + 1)  # Adjust for 1-indexing in R
    
    if nflat > 0:
        if printflag:
            print(f"WARNING: {nflat} Flat cells found")
    
    # Replace the NA's with 0s
    nax = np.where(np.isnan(slopex2))[0]
    nay = np.where(np.isnan(slopey2))[0]
    slopex2[nax] = 0
    slopey2[nay] = 0
    
    output_list = {
        "slopex": slopex2,
        "slopey": slopey2,
        "direction": direction,
        "Sinks": sinklist
    }
    
    return output_list 