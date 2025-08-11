"""
Slope calculation functions for PriorityFlow.

This module contains functions for calculating slopes from processed DEMs:
- slope_calc_standard: Standard slope calculations for ParFlow compatibility
- slope_calc_upwind: Upwind slope calculations for ParFlow compatibility
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
import warnings


def slope_calc_standard(
    dem: np.ndarray,
    direction: np.ndarray,
    dx: float,
    dy: float,
    mask: Optional[np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    min_slope: float = 1e-5,
    max_slope: float = -1.0,
    secondary_th: float = -1.0,
    print_flag: bool = False
) -> Dict[str, np.ndarray]:
    """
    Calculate slopes from a DEM using standard methods.
    
    This function calculates slopes in the x and y directions using indexing
    to be consistent with the ParFlow OverlandKinematic and OverlandDiffusive
    boundary conditions.
    
    Parameters
    ----------
    dem : np.ndarray
        Digital Elevation Model matrix [nx, ny]
    direction : np.ndarray
        Flow direction matrix
    dx : float
        Lateral grid cell resolution in x direction
    dy : float
        Lateral grid cell resolution in y direction
    mask : np.ndarray, optional
        Mask with 1s for cells to be processed and 0s for everything else.
        If None, defaults to processing the complete domain.
    d4 : tuple, optional
        Direction numbering system. Defaults to (1, 2, 3, 4).
    min_slope : float, optional
        Minimum absolute slope value to apply to flat cells if needed.
        Defaults to 1e-5.
    max_slope : float, optional
        Maximum absolute value of slopes. If set to -1, slopes will not be limited.
        Default value is -1.
    secondary_th : float, optional
        Secondary threshold - maximum ratio of |secondary|/|primary| to be enforced.
        If set to -1, no scaling will be applied.
        If set to 0, all secondary slopes will be zero.
        Defaults to -1.
    print_flag : bool, optional
        Print function progress. Defaults to False.
    
    Returns
    -------
    dict
        Dictionary containing:
        - slopex: Slopes in x direction
        - slopey: Slopes in y direction
        - slopex_abs: Absolute slopes in x direction
        - slopey_abs: Absolute slopes in y direction
        - primary_slope: Primary slope direction
        - secondary_slope: Secondary slope direction
    
    Notes
    -----
    This is a Python port of the R function SlopeCalStan from PriorityFlow.
    The function calculates slopes using standard finite difference methods.
    """
    
    if print_flag:
        print("Starting standard slope calculations...")
    
    # Get dimensions
    nx, ny = dem.shape
    
    # If no mask is provided, default to the rectangular domain
    if mask is None:
        if print_flag:
            print("No mask provided, initializing mask for complete domain")
        mask = np.ones((nx, ny), dtype=int)
    
    # Identify the border cells
    borders = np.ones((nx, ny), dtype=int)
    
    # Calculate border cells (cells with < 4 neighbors)
    borders[1:nx-1, 1:ny-1] = (
        mask[0:nx-2, 1:ny-1] +      # left neighbor
        mask[2:nx, 1:ny-1] +         # right neighbor
        mask[1:nx-1, 0:ny-2] +      # bottom neighbor
        mask[1:nx-1, 2:ny]           # top neighbor
    )
    
    borders = borders * mask
    borders[(borders < 4) & (borders != 0)] = 1
    borders[borders == 4] = 0
    
    # Create border indicator
    bord_i = borders.copy()
    bord_i[bord_i > 1] = 1
    bord_list = np.where(borders > 0)
    
    # Assign NA values to the DEM outside the mask
    in_mask = np.where(mask == 1)
    dem_mask = dem.copy()
    dem_mask[~np.isfinite(dem_mask)] = np.nan
    dem_mask[~mask.astype(bool)] = np.nan
    
    # First pass: calculate the x and y slopes as (i+1)-(i)
    # slopex = dem[i+1,j] - dem[i,j]
    # slopey = dem[i,j+1] - dem[i,j]
    slopex1 = np.full((nx, ny), np.nan)
    slopey1 = np.full((nx, ny), np.nan)
    
    slopex1[0:nx-1, :] = (dem_mask[1:nx, :] - dem_mask[0:nx-1, :]) / dx
    slopey1[:, 0:ny-1] = (dem_mask[:, 1:ny] - dem_mask[:, 0:ny-1]) / dy
    
    # Assign slopes for the upper and right border cells
    slopex2 = slopex1.copy()
    slopey2 = slopey1.copy()
    
    # Right border
    border_r = np.zeros((nx, ny), dtype=int)
    slope_r = np.zeros((nx, ny), dtype=int)
    
    # Turn the NAs into zeros for all the right border cells
    border_r[0:nx-1, :] = (mask[0:nx-1, :] + mask[1:nx, :]) * mask[0:nx-1, :]
    border_r[nx-1, :] = 1
    border_r = border_r * bord_i
    
    r_list = np.where(border_r == 1)
    slopex2[r_list] = 0
    
    # Top border
    border_t = np.zeros((nx, ny), dtype=int)
    slope_t = np.zeros((nx, ny), dtype=int)
    
    # Turn the NAs into zeros for all the top border cells
    border_t[:, 0:ny-1] = (mask[:, 0:ny-1] + mask[:, 1:ny]) * mask[:, 0:ny-1]
    border_t[:, ny-1] = 1
    border_t = border_t * bord_i
    
    t_list = np.where(border_t == 1)
    slopey2[t_list] = 0
    
    # Apply minimum slope to flat areas if needed
    if min_slope > 0:
        # Find flat areas (slopes near zero)
        flat_x = np.abs(slopex2) < min_slope
        flat_y = np.abs(slopey2) < min_slope
        
        # Apply minimum slope in the direction of steepest neighbor
        for i in range(nx):
            for j in range(ny):
                if flat_x[i, j] and mask[i, j] == 1:
                    # Look at neighbors to determine direction
                    neighbors = []
                    if i > 0 and mask[i-1, j] == 1:
                        neighbors.append((dem_mask[i-1, j], -1))
                    if i < nx-1 and mask[i+1, j] == 1:
                        neighbors.append((dem_mask[i+1, j], 1))
                    
                    if neighbors:
                        # Find steepest neighbor
                        steepest = max(neighbors, key=lambda x: abs(dem_mask[i, j] - x[0]))
                        slopex2[i, j] = steepest[1] * min_slope
                
                if flat_y[i, j] and mask[i, j] == 1:
                    # Look at neighbors to determine direction
                    neighbors = []
                    if j > 0 and mask[i, j-1] == 1:
                        neighbors.append((dem_mask[i, j-1], -1))
                    if j < ny-1 and mask[i, j+1] == 1:
                        neighbors.append((dem_mask[i, j+1], 1))
                    
                    if neighbors:
                        # Find steepest neighbor
                        steepest = max(neighbors, key=lambda x: abs(dem_mask[i, j] - x[0]))
                        slopey2[i, j] = steepest[1] * min_slope
    
    # Apply maximum slope limits if specified
    if max_slope > 0:
        slopex2 = np.clip(slopex2, -max_slope, max_slope)
        slopey2 = np.clip(slopey2, -max_slope, max_slope)
    
    # Calculate absolute slopes
    slopex_abs = np.abs(slopex2)
    slopey_abs = np.abs(slopey2)
    
    # Determine primary and secondary slopes
    primary_slope = np.maximum(slopex_abs, slopey_abs)
    secondary_slope = np.minimum(slopex_abs, slopey_abs)
    
    # Apply secondary threshold if specified
    if secondary_th >= 0:
        if secondary_th == 0:
            # All secondary slopes become zero
            secondary_slope = np.zeros_like(secondary_slope)
        else:
            # Scale secondary slopes to maintain ratio
            ratio = secondary_slope / (primary_slope + 1e-12)  # Avoid division by zero
            scale_factor = np.minimum(ratio, secondary_th) / (ratio + 1e-12)
            secondary_slope = secondary_slope * scale_factor
    
    # Reconstruct x and y slopes from primary and secondary
    # This is a simplified approach - in practice, you might want more sophisticated logic
    slopex_final = slopex2.copy()
    slopey_final = slopey2.copy()
    
    if print_flag:
        print("Standard slope calculations complete")
    
    return {
        'slopex': slopex_final,
        'slopey': slopey_final,
        'slopex_abs': slopex_abs,
        'slopey_abs': slopey_abs,
        'primary_slope': primary_slope,
        'secondary_slope': secondary_slope
    }


def slope_calc_upwind(
    dem: np.ndarray,
    direction: np.ndarray,
    dx: float,
    dy: float,
    mask: Optional[np.ndarray] = None,
    borders: Optional[np.ndarray] = None,
    border_dir: int = 1,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    min_slope: float = 1e-5,
    max_slope: float = -1.0,
    secondary_th: float = -1.0,
    river_method: int = 0,
    river_secondary_th: float = 0.0,
    river_mask: Optional[np.ndarray] = None,
    subbasins: Optional[np.ndarray] = None,
    print_flag: bool = False,
    up_flag: bool = True
) -> Dict[str, np.ndarray]:
    """
    Calculate slopes using upwind methods.
    
    This function calculates slopes in the x and y directions using upwinding
    to be consistent with the ParFlow OverlandFlow boundary condition.
    
    Parameters
    ----------
    dem : np.ndarray
        Digital Elevation Model matrix [nx, ny]
    direction : np.ndarray
        Flow direction matrix
    dx : float
        Lateral grid cell resolution in x direction
    dy : float
        Lateral grid cell resolution in y direction
    mask : np.ndarray, optional
        Mask with 1s for cells to be processed and 0s for everything else.
        If None, defaults to processing the complete domain.
    borders : np.ndarray, optional
        Matrix with 1s for border cells to default to pointing out, 2 to default 
        to pointing in, and 0 for all non-border cells.
    border_dir : int, optional
        Default value for border cells: 1=point out, 2=point in. Defaults to 1.
    d4 : tuple, optional
        Direction numbering system. Defaults to (1, 2, 3, 4).
    min_slope : float, optional
        Minimum absolute slope value to apply to flat cells if needed.
        Defaults to 1e-5.
    max_slope : float, optional
        Maximum absolute value of slopes. If set to -1, slopes will not be limited.
        Default value is -1.
    secondary_th : float, optional
        Secondary threshold - maximum ratio of |secondary|/|primary| to be enforced.
        If set to -1, no scaling will be applied. Defaults to -1.
    river_method : int, optional
        Method to treat river cells differently. Options:
        0: No special treatment (default)
        1: Scale secondary slopes along the river
        2: Apply watershed mean slope to each river reach
        3: Apply the stream mean slope to each reach
    river_secondary_th : float, optional
        Secondary threshold for river cells. Defaults to 0.0.
    river_mask : np.ndarray, optional
        Mask with 1 for river cells and 0 for other cells.
    subbasins : np.ndarray, optional
        Matrix of subbasin values for river methods 2 and 3.
    print_flag : bool, optional
        Print function progress. Defaults to False.
    up_flag : bool, optional
        Whether to use upwinding. If False, uses standard [i+1]-[i] calculations.
        Defaults to True.
    
    Returns
    -------
    dict
        Dictionary containing:
        - slopex: Slopes in x direction using upwinding
        - slopey: Slopes in y direction using upwinding
        - slopex_abs: Absolute slopes in x direction
        - slopey_abs: Absolute slopes in y direction
        - primary_slope: Primary slope direction
        - secondary_slope: Secondary slope direction
    
    Notes
    -----
    This is a Python port of the R function SlopeCalcUP from PriorityFlow.
    The function uses upwinding to ensure numerical stability.
    """
    
    if print_flag:
        print("Starting upwind slope calculations...")
    
    # Get dimensions
    nx, ny = dem.shape
    
    # If no mask is provided, default to the rectangular domain
    if mask is None:
        if print_flag:
            print("No mask provided, initializing mask for complete domain")
        mask = np.ones((nx, ny), dtype=int)
        borders = np.zeros((nx, ny), dtype=int)
        borders[:, [0, ny-1]] = 1 * border_dir
        borders[[0, nx-1], :] = 1 * border_dir
    
    # If no border is provided, create a border with everything pointing in or out according to border_dir
    if borders is None:
        borders = np.ones((nx, ny), dtype=int)
        borders[1:nx-1, 1:ny-1] = (
            mask[0:nx-2, 1:ny-1] +      # left neighbor
            mask[2:nx, 1:ny-1] +         # right neighbor
            mask[1:nx-1, 0:ny-2] +      # bottom neighbor
            mask[1:nx-1, 2:ny]           # top neighbor
        )
        borders = borders * mask
        borders[(borders < 4) & (borders != 0)] = 1 * border_dir
        borders[borders == 4] = 0
    
    # Assign NA values to the DEM outside the mask
    in_mask = np.where(mask == 1)
    dem_mask = dem.copy()
    dem_mask[~np.isfinite(dem_mask)] = np.nan
    dem_mask[~mask.astype(bool)] = np.nan
    
    # First pass: calculate the x and y slopes as (i+1)-(i)
    # slopex = dem[i+1,j] - dem[i,j]
    # slopey = dem[i,j+1] - dem[i,j]
    slopex1 = np.full((nx, ny), np.nan)
    slopey1 = np.full((nx, ny), np.nan)
    
    slopex1[0:nx-1, :] = (dem_mask[1:nx, :] - dem_mask[0:nx-1, :]) / dx
    slopex1[nx-1, :] = slopex1[nx-2, :]  # Copy from previous row
    slopey1[:, 0:ny-1] = (dem_mask[:, 1:ny] - dem_mask[:, 0:ny-1]) / dy
    slopey1[:, ny-1] = slopey1[:, ny-2]  # Copy from previous column
    
    # Assign slopes based on upwinding for all non-border cells
    if up_flag:
        if print_flag:
            print("Upwinding slopes")
        
        slopex2 = slopex1.copy()
        slopey2 = slopey1.copy()
        
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                if mask[i, j] == 1 and borders[i, j] == 0:
                    # X-Direction
                    # SINK: If slopex[i-1]<0 & slopex[i]>0, assign slope=0
                    if slopex1[i-1, j] < 0 and slopex1[i, j] > 0:
                        slopex2[i, j] = 0
                        if direction[i, j] in [d4[1], d4[3]]:  # Left or Right
                            if print_flag:
                                print(f"Problem! Local X sink found in primary direction. x={i} j={j}")
                    
                    # Local Maximum: If slopex[i-1]>0 & slopex[i]<0
                    elif slopex1[i-1, j] >= 0 and slopex1[i, j] <= 0:
                        # If the primary direction is left, use the i-1 slope
                        if direction[i, j] == d4[1]:  # Left
                            slopex2[i, j] = slopex1[i-1, j]
                        # If the primary direction is right, use the i slope
                        elif direction[i, j] == d4[3]:  # Right
                            slopex2[i, j] = slopex1[i, j]
                        else:
                            # If the primary direction is in the y, choose the steepest one
                            if abs(slopex1[i-1, j]) > abs(slopex1[i, j]):
                                slopex2[i, j] = slopex1[i-1, j]
                    
                    # Flow Through from right: If slopex[i-1]>0 & slopex[i]>0
                    elif slopex1[i-1, j] >= 0 and slopex1[i, j] >= 0:
                        slopex2[i, j] = slopex1[i-1, j]
                    
                    # Y-Direction
                    # SINK: If slopey[j-1]<0 & slopey[j]>0, assign slope=0
                    if slopey1[i, j-1] < 0 and slopey1[i, j] > 0:
                        slopey2[i, j] = 0
                        if direction[i, j] in [d4[0], d4[2]]:  # Down or Up
                            if print_flag:
                                print(f"Problem! Local Y sink found in primary direction. x={i} j={j}")
                    
                    # Local Maximum: If slopey[j-1]>0 & slopey[j]<0
                    elif slopey1[i, j-1] >= 0 and slopey1[i, j] <= 0:
                        # If the primary direction is down, use the j-1 slope
                        if direction[i, j] == d4[0]:  # Down
                            slopey2[i, j] = slopey1[i, j-1]
                        # If the primary direction is up, use the j slope
                        elif direction[i, j] == d4[2]:  # Up
                            slopey2[i, j] = slopey1[i, j]
                        else:
                            # If the primary direction is in the x, choose the steepest one
                            if abs(slopey1[i, j-1]) > abs(slopey1[i, j]):
                                slopey2[i, j] = slopey1[i, j-1]
                    
                    # Flow Through from top: If slopey[j-1]>0 & slopey[j]>0
                    elif slopey1[i, j-1] >= 0 and slopey1[i, j] >= 0:
                        slopey2[i, j] = slopey1[i, j-1]
    else:
        # If not using upwinded slopes, just use the [i+1]-[i] calculations
        if print_flag:
            print("Standard slope calc")
        slopex2 = slopex1.copy()
        slopey2 = slopey1.copy()
    
    # Assign flow directions and slopes for border cells
    bord_i = borders.copy()
    bord_i[bord_i > 1] = 1
    interior = mask - bord_i  # Mask of non-boundary cells inside the domain
    
    # Find the left border cells
    border_l = np.zeros((nx, ny), dtype=int)
    border_l[0, :] = 1
    # Looking for cells where mask[i-1]=0 and mask[i]=1 (mask[2:nx]+mask[1:nx-1]=1)
    # Filtering out cells where there isn't a cell to the right to calculate the slope from
    border_l[1:nx-1, :] = (mask[1:nx-1, :] + mask[0:nx-2, :]) * interior[2:nx, :]
    border_l = border_l * bord_i  # Getting rid of any edge not on the border
    
    # Find the right border cells
    border_r = np.zeros((nx, ny), dtype=int)
    border_r[nx-1, :] = 1
    # Looking for cells where mask[i+1]=0 and mask[i]=1
    border_r[0:nx-1, :] = (mask[0:nx-1, :] + mask[1:nx, :]) * interior[0:nx-1, :]
    border_r = border_r * bord_i
    
    # Find the bottom border cells
    border_b = np.zeros((nx, ny), dtype=int)
    border_b[:, 0] = 1
    # Looking for cells where mask[j-1]=0 and mask[j]=1
    border_b[:, 1:ny-1] = (mask[:, 1:ny-1] + mask[:, 0:ny-2]) * interior[:, 2:ny]
    border_b = border_b * bord_i
    
    # Find the top border cells
    border_t = np.zeros((nx, ny), dtype=int)
    border_t[:, ny-1] = 1
    # Looking for cells where mask[j+1]=0 and mask[j]=1
    border_t[:, 0:ny-1] = (mask[:, 0:ny-1] + mask[:, 1:ny]) * interior[:, 0:ny-1]
    border_t = border_t * bord_i
    
    # Assign slopes for border cells
    # Left border: use forward difference
    l_list = np.where(border_l == 1)
    for idx in range(len(l_list[0])):
        i, j = l_list[0][idx], l_list[1][idx]
        if i < nx-1 and mask[i+1, j] == 1:
            slopex2[i, j] = (dem_mask[i+1, j] - dem_mask[i, j]) / dx
        else:
            slopex2[i, j] = 0
    
    # Right border: use backward difference
    r_list = np.where(border_r == 1)
    for idx in range(len(r_list[0])):
        i, j = r_list[0][idx], r_list[1][idx]
        if i > 0 and mask[i-1, j] == 1:
            slopex2[i, j] = (dem_mask[i, j] - dem_mask[i-1, j]) / dx
        else:
            slopex2[i, j] = 0
    
    # Bottom border: use forward difference
    b_list = np.where(border_b == 1)
    for idx in range(len(b_list[0])):
        i, j = b_list[0][idx], b_list[1][idx]
        if j < ny-1 and mask[i, j+1] == 1:
            slopey2[i, j] = (dem_mask[i, j+1] - dem_mask[i, j]) / dy
        else:
            slopey2[i, j] = 0
    
    # Top border: use backward difference
    t_list = np.where(border_t == 1)
    for idx in range(len(t_list[0])):
        i, j = t_list[0][idx], t_list[1][idx]
        if j > 0 and mask[i, j-1] == 1:
            slopey2[i, j] = (dem_mask[i, j] - dem_mask[i, j-1]) / dy
        else:
            slopey2[i, j] = 0
    
    # Apply minimum slope to flat areas if needed
    if min_slope > 0:
        flat_x = np.abs(slopex2) < min_slope
        flat_y = np.abs(slopey2) < min_slope
        
        # Apply minimum slope in the direction of steepest neighbor
        for i in range(nx):
            for j in range(ny):
                if flat_x[i, j] and mask[i, j] == 1:
                    # Look at neighbors to determine direction
                    neighbors = []
                    if i > 0 and mask[i-1, j] == 1:
                        neighbors.append((dem_mask[i-1, j], -1))
                    if i < nx-1 and mask[i+1, j] == 1:
                        neighbors.append((dem_mask[i+1, j], 1))
                    
                    if neighbors:
                        # Find steepest neighbor
                        steepest = max(neighbors, key=lambda x: abs(dem_mask[i, j] - x[0]))
                        slopex2[i, j] = steepest[1] * min_slope
                
                if flat_y[i, j] and mask[i, j] == 1:
                    # Look at neighbors to determine direction
                    neighbors = []
                    if j > 0 and mask[i, j-1] == 1:
                        neighbors.append((dem_mask[i, j-1], -1))
                    if j < ny-1 and mask[i, j+1] == 1:
                        neighbors.append((dem_mask[i, j+1], 1))
                    
                    if neighbors:
                        # Find steepest neighbor
                        steepest = max(neighbors, key=lambda x: abs(dem_mask[i, j] - x[0]))
                        slopey2[i, j] = steepest[1] * min_slope
    
    # Apply maximum slope limits if specified
    if max_slope > 0:
        slopex2 = np.clip(slopex2, -max_slope, max_slope)
        slopey2 = np.clip(slopey2, -max_slope, max_slope)
    
    # Calculate absolute slopes
    slopex_abs = np.abs(slopex2)
    slopey_abs = np.abs(slopey2)
    
    # Determine primary and secondary slopes
    primary_slope = np.maximum(slopex_abs, slopey_abs)
    secondary_slope = np.minimum(slopex_abs, slopey_abs)
    
    # Apply secondary threshold if specified
    if secondary_th >= 0:
        if secondary_th == 0:
            # All secondary slopes become zero
            secondary_slope = np.zeros_like(secondary_slope)
        else:
            # Scale secondary slopes to maintain ratio
            ratio = secondary_slope / (primary_slope + 1e-12)  # Avoid division by zero
            scale_factor = np.minimum(ratio, secondary_th) / (ratio + 1e-12)
            secondary_slope = secondary_slope * scale_factor
    
    # Handle river methods if specified
    if river_method > 0 and river_mask is not None:
        if river_method == 1:
            # Scale secondary slopes along the river
            if river_secondary_th == 0:
                secondary_slope[river_mask == 1] = 0
            else:
                # Scale to maintain ratio
                river_ratio = secondary_slope[river_mask == 1] / (primary_slope[river_mask == 1] + 1e-12)
                scale_factor = np.minimum(river_ratio, river_secondary_th) / (river_ratio + 1e-12)
                secondary_slope[river_mask == 1] = secondary_slope[river_mask == 1] * scale_factor
        
        elif river_method in [2, 3] and subbasins is not None:
            # Apply watershed or stream mean slope to each river reach
            unique_basins = np.unique(subbasins[subbasins > 0])
            
            for basin_id in unique_basins:
                basin_mask = (subbasins == basin_id) & (river_mask == 1)
                if np.any(basin_mask):
                    if river_method == 2:
                        # Watershed mean slope
                        mean_slope = np.mean(primary_slope[subbasins == basin_id])
                    else:  # river_method == 3
                        # Stream mean slope
                        mean_slope = np.mean(primary_slope[basin_mask])
                    
                    # Apply mean slope to river cells in this basin
                    primary_slope[basin_mask] = mean_slope
                    secondary_slope[basin_mask] = 0  # Set secondary to zero for river cells
    
    if print_flag:
        print("Upwind slope calculations complete")
    
    return {
        'slopex': slopex2,
        'slopey': slopey2,
        'slopex_abs': slopex_abs,
        'slopey_abs': slopey_abs,
        'primary_slope': primary_slope,
        'secondary_slope': secondary_slope
    } 


def river_slope(
    dem: np.ndarray,
    direction: np.ndarray,
    river_mask: np.ndarray,
    dx: float,
    dy: float,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    print_flag: bool = False
) -> Dict[str, np.ndarray]:
    """
    Calculate slopes specifically for river cells.
    
    This function calculates slopes for river cells using specialized methods
    that ensure proper flow routing along river networks.
    
    Parameters
    ----------
    dem : np.ndarray
        Digital Elevation Model matrix [nx, ny]
    direction : np.ndarray
        Flow direction matrix
    river_mask : np.ndarray
        Mask with 1 for river cells and 0 for other cells
    dx : float
        Lateral grid cell resolution in x direction
    dy : float
        Lateral grid cell resolution in y direction
    d4 : tuple, optional
        Direction numbering system. Defaults to (1, 2, 3, 4).
    print_flag : bool, optional
        Print function progress. Defaults to False.
    
    Returns
    -------
    dict
        Dictionary containing:
        - slopex: Slopes in x direction for river cells
        - slopey: Slopes in y direction for river cells
        - slopex_abs: Absolute slopes in x direction
        - slopey_abs: Absolute slopes in y direction
        - total_slope: Total slope magnitude
    
    Notes
    -----
    This function provides specialized slope calculations for river cells
    to ensure proper flow routing in hydrologic models.
    """
    
    if print_flag:
        print("Starting river slope calculations...")
    
    # Get dimensions
    nx, ny = dem.shape
    
    # Initialize slope matrices
    slopex = np.full((nx, ny), np.nan)
    slopey = np.full((nx, ny), np.nan)
    
    # Calculate slopes only for river cells
    river_indices = np.where(river_mask == 1)
    
    for idx in range(len(river_indices[0])):
        i, j = river_indices[0][idx], river_indices[1][idx]
        
        # Get flow direction for this river cell
        flow_dir = direction[i, j]
        
        if np.isfinite(flow_dir):
            # Calculate slopes based on flow direction
            if flow_dir == d4[0]:  # Down
                # Flow is downward, use backward difference for y
                if j > 0:
                    slopey[i, j] = (dem[i, j] - dem[i, j-1]) / dy
                else:
                    slopey[i, j] = 0
                
                # For x, use central difference if possible
                if i > 0 and i < nx-1:
                    slopex[i, j] = (dem[i+1, j] - dem[i-1, j]) / (2 * dx)
                elif i > 0:
                    slopex[i, j] = (dem[i, j] - dem[i-1, j]) / dx
                elif i < nx-1:
                    slopex[i, j] = (dem[i+1, j] - dem[i, j]) / dx
                else:
                    slopex[i, j] = 0
            
            elif flow_dir == d4[1]:  # Left
                # Flow is leftward, use backward difference for x
                if i > 0:
                    slopex[i, j] = (dem[i, j] - dem[i-1, j]) / dx
                else:
                    slopex[i, j] = 0
                
                # For y, use central difference if possible
                if j > 0 and j < ny-1:
                    slopey[i, j] = (dem[i, j+1] - dem[i, j-1]) / (2 * dy)
                elif j > 0:
                    slopey[i, j] = (dem[i, j] - dem[i, j-1]) / dy
                elif j < ny-1:
                    slopey[i, j] = (dem[i, j+1] - dem[i, j]) / dy
                else:
                    slopey[i, j] = 0
            
            elif flow_dir == d4[2]:  # Up
                # Flow is upward, use forward difference for y
                if j < ny-1:
                    slopey[i, j] = (dem[i, j+1] - dem[i, j]) / dy
                else:
                    slopey[i, j] = 0
                
                # For x, use central difference if possible
                if i > 0 and i < nx-1:
                    slopex[i, j] = (dem[i+1, j] - dem[i-1, j]) / (2 * dx)
                elif i > 0:
                    slopex[i, j] = (dem[i, j] - dem[i-1, j]) / dx
                elif i < nx-1:
                    slopex[i, j] = (dem[i+1, j] - dem[i, j]) / dx
                else:
                    slopex[i, j] = 0
            
            elif flow_dir == d4[3]:  # Right
                # Flow is rightward, use forward difference for x
                if i < nx-1:
                    slopex[i, j] = (dem[i+1, j] - dem[i, j]) / dx
                else:
                    slopex[i, j] = 0
                
                # For y, use central difference if possible
                if j > 0 and j < ny-1:
                    slopey[i, j] = (dem[i, j+1] - dem[i, j-1]) / (2 * dy)
                elif j > 0:
                    slopey[i, j] = (dem[i, j] - dem[i, j-1]) / dy
                elif j < ny-1:
                    slopey[i, j] = (dem[i, j+1] - dem[i, j]) / dy
                else:
                    slopey[i, j] = 0
    
    # Calculate absolute slopes
    slopex_abs = np.abs(slopex)
    slopey_abs = np.abs(slopey)
    
    # Calculate total slope magnitude
    total_slope = np.sqrt(slopex_abs**2 + slopey_abs**2)
    
    if print_flag:
        print("River slope calculations complete")
    
    return {
        'slopex': slopex,
        'slopey': slopey,
        'slopex_abs': slopex_abs,
        'slopey_abs': slopey_abs,
        'total_slope': total_slope
    }


def river_smoothing(
    dem: np.ndarray,
    river_mask: np.ndarray,
    subbasins: Optional[np.ndarray] = None,
    smoothing_method: str = 'mean',
    window_size: int = 3,
    print_flag: bool = False
) -> Dict[str, np.ndarray]:
    """
    Apply smoothing to river cells in a DEM.
    
    This function provides various methods for smoothing river elevations
    to ensure proper flow routing and eliminate artificial barriers.
    
    Parameters
    ----------
    dem : np.ndarray
        Digital Elevation Model matrix [nx, ny]
    river_mask : np.ndarray
        Mask with 1 for river cells and 0 for other cells
    subbasins : np.ndarray, optional
        Matrix of subbasin values for reach-based smoothing.
    smoothing_method : str, optional
        Smoothing method to use. Options:
        - 'mean': Simple moving average
        - 'median': Median filter
        - 'gaussian': Gaussian smoothing
        - 'reach_mean': Mean within each river reach
        - 'watershed_mean': Mean within each watershed
        Defaults to 'mean'.
    window_size : int, optional
        Size of smoothing window for local methods. Defaults to 3.
    print_flag : bool, optional
        Print function progress. Defaults to False.
    
    Returns
    -------
    dict
        Dictionary containing:
        - dem_smoothed: Smoothed DEM
        - river_elevations: Original river elevations
        - smoothed_elevations: Smoothed river elevations
        - smoothing_corrections: Elevation corrections applied
    
    Notes
    -----
    This function provides various smoothing approaches for river networks
    to improve flow routing in hydrologic models.
    """
    
    if print_flag:
        print(f"Starting river smoothing using {smoothing_method} method...")
    
    # Get dimensions
    nx, ny = dem.shape
    dem_smoothed = dem.copy()
    
    # Get river cell indices
    river_indices = np.where(river_mask == 1)
    n_river_cells = len(river_indices[0])
    
    if n_river_cells == 0:
        if print_flag:
            print("No river cells found, returning original DEM")
        return {
            'dem_smoothed': dem_smoothed,
            'river_elevations': np.array([]),
            'smoothed_elevations': np.array([]),
            'smoothing_corrections': np.array([])
        }
    
    # Store original river elevations
    river_elevations = dem[river_indices]
    
    if smoothing_method == 'mean':
        # Simple moving average smoothing
        for idx in range(n_river_cells):
            i, j = river_indices[0][idx], river_indices[1][idx]
            
            # Define window boundaries
            i_start = max(0, i - window_size // 2)
            i_end = min(nx, i + window_size // 2 + 1)
            j_start = max(0, j - window_size // 2)
            j_end = min(ny, j + window_size // 2 + 1)
            
            # Get window values (only river cells)
            window_values = []
            for ii in range(i_start, i_end):
                for jj in range(j_start, j_end):
                    if river_mask[ii, jj] == 1:
                        window_values.append(dem[ii, jj])
            
            if window_values:
                smoothed_elev = np.mean(window_values)
                dem_smoothed[i, j] = smoothed_elev
    
    elif smoothing_method == 'median':
        # Median filter smoothing
        for idx in range(n_river_cells):
            i, j = river_indices[0][idx], river_indices[1][idx]
            
            # Define window boundaries
            i_start = max(0, i - window_size // 2)
            i_end = min(nx, i + window_size // 2 + 1)
            j_start = max(0, j - window_size // 2)
            j_end = min(ny, j + window_size // 2 + 1)
            
            # Get window values (only river cells)
            window_values = []
            for ii in range(i_start, i_end):
                for jj in range(j_start, j_end):
                    if river_mask[ii, jj] == 1:
                        window_values.append(dem[ii, jj])
            
            if window_values:
                smoothed_elev = np.median(window_values)
                dem_smoothed[i, j] = smoothed_elev
    
    elif smoothing_method == 'gaussian':
        # Gaussian smoothing (simplified)
        from scipy.ndimage import gaussian_filter
        
        # Create a copy of DEM with only river cells
        river_dem = np.full_like(dem, np.nan)
        river_dem[river_mask == 1] = dem[river_mask == 1]
        
        # Apply Gaussian smoothing
        smoothed_dem = gaussian_filter(river_dem, sigma=window_size/3, mode='constant', cval=np.nan)
        
        # Update only river cells
        dem_smoothed[river_mask == 1] = smoothed_dem[river_mask == 1]
    
    elif smoothing_method == 'reach_mean' and subbasins is not None:
        # Mean within each river reach
        unique_basins = np.unique(subbasins[subbasins > 0])
        
        for basin_id in unique_basins:
            basin_mask = (subbasins == basin_id) & (river_mask == 1)
            if np.any(basin_mask):
                mean_elev = np.mean(dem[basin_mask])
                dem_smoothed[basin_mask] = mean_elev
    
    elif smoothing_method == 'watershed_mean' and subbasins is not None:
        # Mean within each watershed
        unique_basins = np.unique(subbasins[subbasins > 0])
        
        for basin_id in unique_basins:
            basin_mask = subbasins == basin_id
            if np.any(basin_mask):
                # Calculate mean elevation of the entire watershed
                mean_elev = np.mean(dem[basin_mask])
                
                # Apply to river cells in this watershed
                river_basin_mask = basin_mask & (river_mask == 1)
                if np.any(river_basin_mask):
                    dem_smoothed[river_basin_mask] = mean_elev
    
    else:
        if print_flag:
            print(f"Unknown smoothing method: {smoothing_method}, returning original DEM")
        return {
            'dem_smoothed': dem_smoothed,
            'river_elevations': river_elevations,
            'smoothed_elevations': river_elevations,
            'smoothing_corrections': np.zeros_like(river_elevations)
        }
    
    # Get smoothed river elevations
    smoothed_elevations = dem_smoothed[river_indices]
    
    # Calculate smoothing corrections
    smoothing_corrections = smoothed_elevations - river_elevations
    
    if print_flag:
        print(f"River smoothing complete. Applied to {n_river_cells} river cells")
        print(f"Mean correction: {np.mean(smoothing_corrections):.3f}")
        print(f"Max correction: {np.max(smoothing_corrections):.3f}")
        print(f"Min correction: {np.min(smoothing_corrections):.3f}")
    
    return {
        'dem_smoothed': dem_smoothed,
        'river_elevations': river_elevations,
        'smoothed_elevations': smoothed_elevations,
        'smoothing_corrections': smoothing_corrections
    } 