"""
Flat Fix functions for PriorityFlow.

This module provides functions to identify and fix stagnation points by finding
cells where the total output slope is much less than the input slopes and
adjusting the outlet slope to address this issue.
"""

import numpy as np
from typing import Dict, Optional


def fix_flat(
    direction: np.ndarray,
    slopex: np.ndarray,
    slopey: np.ndarray,
    adj_th: float,
    adj_ratio: float = -1.0,
    mask: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Identify and fix stagnation points.
    
    A function that finds cells where the total output slope is much less than the input
    slopes and adjusts the outlet slope to address this. This is useful for fixing
    flat areas and stagnation points in flow calculations.
    
    Parameters
    ----------
    direction : np.ndarray
        Nx by Ny matrix of flow directions following the convention (1=down, 2=left, 3=up, 4=right)
    slopex : np.ndarray
        Nx by Ny matrix of slopes in the x direction (should be face centered slopes as calculated with SlopeCalcStan)
    slopey : np.ndarray
        Nx by Ny matrix of slopes in the y direction (should be face centered slopes as calculated with SlopeCalcStan)
    adj_th : float
        Threshold for slope adjustment. If the total slopes out divided by the total slopes into a given cell is less than adj_th, the outlet will be scaled by adj_ratio.
    adj_ratio : float, optional
        Scaler value for slope adjustment. New outlet slopes will be set to initial outlet slope times adj_ratio. If no adj_ratio is provided, then the outlet slope will be set to the total inlets slopes times the adjustment threshold. Defaults to -1.0.
    mask : np.ndarray, optional
        Nx by Ny matrix indicating the active domain, 1=active, 0=inactive. If no mask is provided, the function will be applied over the entire input matrices.
    
    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing:
        - 'outslope': Nx by Ny matrix with the sum of the slopes pointed out of every cell (sum of slope magnitudes, all positive)
        - 'inslope': Nx by Ny matrix with the sum of the slopes pointed into every cell (sum of slope magnitudes, all positive)
        - 'OutIn_ratio': Nx by Ny matrix with the ratio of outslope to inslope (cells outside mask or with 0 inslope are assigned -1)
        - 'slopex': The adjusted Nx by Ny matrix of slopex values
        - 'slopey': The adjusted Nx by Ny matrix of slopey values
        - 'adj_mask': Nx by Ny matrix indicating cells that were adjusted (1=adjusted, 0=not adjusted)
        - 'SlopeOutlet': Nx by Ny matrix of the outlet slope for every grid cell
        - 'SlopeOutletNew': Nx by Ny matrix of the outlet slope for every grid cell after processing
    
    Notes
    -----
    This function implements a stagnation point detection and correction algorithm that:
    1. Calculates total input and output slopes for each cell
    2. Identifies cells with low output/input slope ratios
    3. Adjusts outlet slopes to meet minimum threshold requirements
    4. Provides comprehensive slope analysis and adjustment tracking
    
    The algorithm handles:
    - Face-centered slope calculations
    - Direction-based slope adjustments
    - Threshold-based correction criteria
    - Comprehensive output reporting
    - Mask-based domain processing
    """
    # HydroFrame layout -> internal R-style layout
    direction = direction.T.copy()
    slopex = slopex.T.copy()
    slopey = slopey.T.copy()
    if mask is not None:
        mask = mask.T.copy()

    # R: nx=dim(direction)[1] ny=dim(direction)[2]
    nx = direction.shape[0]
    ny = direction.shape[1]

    # R: if(missing(mask)){ ... }
    if mask is None:
        print("No domain mask provided using entire domain")
        mask = np.ones((nx, ny))

    # R: intot=outtot=outdirslop=ratio_mask=matrix(0, nrow=nx, ncol=ny)
    intot = np.zeros((nx, ny))
    outtot = np.zeros((nx, ny))
    ratio_mask = np.zeros((nx, ny))
    # R: outdirslope=outdirslopeNew=matrix(0, nrow=nx, ncol=ny)
    outdirslope = np.zeros((nx, ny))
    outdirslope_new = np.zeros((nx, ny))
    # R: ratio=matrix(-1, nrow=nx, ncol=ny)
    ratio = np.full((nx, ny), -1.0)
    # R: slopexNew=slopex ; slopeyNew=slopey
    slopex_new = slopex.copy()
    slopey_new = slopey.copy()

    # R: for(j in 2:(ny-1)){ for(i in 2:(nx-1)){ ... } }
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            if mask[i, j] == 1:
                # R: intot[i,j]=... ; outtot[i,j]=...
                intot[i, j] = (
                    max(slopex[i, j], 0)
                    + max(slopey[i, j], 0)
                    - min(slopex[i - 1, j], 0)
                    - min(slopey[i, j - 1], 0)
                )
                outtot[i, j] = (
                    max(slopex[i - 1, j], 0)
                    + max(slopey[i, j - 1], 0)
                    - min(slopex[i, j], 0)
                    - min(slopey[i, j], 0)
                )

                # R: ratio[i,j]=outtot[i,j]/intot[i,j]
                ratio[i, j] = outtot[i, j] / intot[i, j]

                # R: if(abs(ratio[i,j])==Inf){ratio[i,j]=(-1)}
                if np.abs(ratio[i, j]) == np.inf:
                    ratio[i, j] = -1

                # R: if(ratio[i,j]<adj_th & ratio[i,j]>0){ratio_mask[i,j]=1}
                if ratio[i, j] < adj_th and ratio[i, j] > 0:
                    ratio_mask[i, j] = 1

                # R: if(is.na(direction[i,j])==F){ ... }
                if not np.isnan(direction[i, j]):
                    if direction[i, j] == 1:
                        outdirslope[i, j] = slopey[i, j - 1]
                        if ratio_mask[i, j] == 1:
                            if adj_ratio > 0:
                                slopey_new[i, j - 1] = slopey[i, j - 1] * adj_ratio
                            else:
                                slopey_new[i, j - 1] = (
                                    intot[i, j] * adj_th * np.sign(slopey[i, j - 1])
                                )
                            outdirslope_new[i, j] = slopey_new[i, j - 1]

                    elif direction[i, j] == 2:
                        outdirslope[i, j] = slopex[i - 1, j]
                        if ratio_mask[i, j] == 1:
                            if adj_ratio > 0:
                                slopex_new[i - 1, j] = slopex[i - 1, j] * adj_ratio
                            else:
                                slopex_new[i - 1, j] = (
                                    intot[i, j] * adj_th * np.sign(slopex[i - 1, j])
                                )
                            outdirslope_new[i, j] = slopex_new[i - 1, j]

                    elif direction[i, j] == 3:
                        outdirslope[i, j] = slopey[i, j]
                        if ratio_mask[i, j] == 1:
                            if adj_ratio > 0:
                                slopey_new[i, j] = slopey[i, j] * adj_ratio
                            else:
                                slopey_new[i, j] = intot[i, j] * adj_th * np.sign(slopey[i, j])
                            outdirslope_new[i, j] = slopey_new[i, j]

                    elif direction[i, j] == 4:
                        outdirslope[i, j] = slopex[i, j]
                        if ratio_mask[i, j] == 1:
                            if adj_ratio > 0:
                                slopex_new[i, j] = slopex[i, j] * adj_ratio
                            else:
                                slopex_new[i, j] = intot[i, j] * adj_th * np.sign(slopex[i, j])
                            outdirslope_new[i, j] = slopex_new[i, j]

    # Internal layout -> HydroFrame layout (transpose 2D arrays)
    output_list = {
        "outslope": outtot.T,
        "inslope": intot.T,
        "OutIn_ratio": ratio.T,
        "slopex": slopex_new.T,
        "slopey": slopey_new.T,
        "adj_mask": ratio_mask.T,
        "SlopeOutlet": outdirslope.T,
        "SlopeOutletNew": outdirslope_new.T,
    }

    return output_list