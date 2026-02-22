"""
Apply minimum slope and secondary scaling to river cells.

Line-by-line translation of River_Slope.R (RivSlope) from the R PriorityFlow
package. R uses 1-based indexing; we use 0-based.
"""

import numpy as np
from typing import Dict, Optional, Tuple

####################################################################
# PriorityFlow - Topographic Processing Toolkit for Hydrologic Models
# Copyright (C) 2018  Laura Condon (lecondon@email.arizona.edu)
# Contributors - Reed Maxwell (rmaxwell@mines.edu)
####################################################################


def riv_slope(
    direction: np.ndarray,
    slopex: np.ndarray,
    slopey: np.ndarray,
    minslope: float,
    river_mask: Optional[np.ndarray] = None,
    remove_sec: bool = False,
) -> Dict[str, np.ndarray]:
    # R: nx=dim(direction)[1]  ny=dim(direction)[2]
    nx = direction.shape[0]
    ny = direction.shape[1]

    # Columns: (1)deltax, (2)deltay, (3) direction number assuming you are looking downstream
    # R: kd=matrix(0, nrow=4, ncol=3)  kd[,1]=c(0,-1,0,0)  kd[,2]=c(-1,0,0,0)  kd[,3]=c(1,2,3,4)
    kd = np.zeros((4, 3))
    kd[:, 0] = [0, -1, 0, 0]
    kd[:, 1] = [-1, 0, 0, 0]
    kd[:, 2] = [1, 2, 3, 4]

    # R: setup outputs  outdirslope=outdirslopeNew=adj_mask=matrix(0, ...)
    outdirslope = np.zeros((nx, ny))
    outdirslopeNew = np.zeros((nx, ny))
    adj_mask = np.zeros((nx, ny))
    # R: slopexNew=slopex  slopeyNew=slopey
    slopexNew = slopex.copy()
    slopeyNew = slopey.copy()

    # R: Loop over the domain adjusting slopes along river cells as needed
    # R: for(j in 2:(ny-1)){ for(i in 2:(nx-1)){
    # R 1-based: interior cells 2..ny-1, 2..nx-1. Python 0-based: 1..ny-2, 1..nx-2
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            # R: if(RiverMask[i,j]==1){
            if river_mask[i, j] == 1:
                # R: sec_out=c( max(slopey[i,(j-1)],0), max(slopex[(i-1),j],0), -min(slopey[i,j],0), -min(slopex[i,j],0))
                sec_out = np.array([
                    max(slopey[i, j - 1], 0),
                    max(slopex[i - 1, j], 0),
                    -min(slopey[i, j], 0),
                    -min(slopex[i, j], 0),
                ])

                # R: if(is.na(direction[i,j])==F){
                if not np.isnan(direction[i, j]):
                    # R: sec_out[direction[i,j]]=0
                    dir_val = int(direction[i, j])
                    sec_out[dir_val - 1] = 0  # R 1-based direction 1..4

                    # R: Set the primary direction slope to be >=minslope
                    # R: if(direction[i,j]==1 & abs(slopey[i,(j-1)])<minslope){
                    if dir_val == 1 and np.abs(slopey[i, j - 1]) < minslope:
                        # R: slopeyNew[i,(j-1)]=sign(slopey[i,(j-1)])*minslope
                        slopeyNew[i, j - 1] = np.sign(slopey[i, j - 1]) * minslope
                        outdirslope[i, j] = slopey[i, j - 1]
                        outdirslopeNew[i, j] = slopeyNew[i, j - 1]
                        adj_mask[i, j] = 0.5
                    elif dir_val == 2 and np.abs(slopex[i - 1, j]) < minslope:
                        slopexNew[i - 1, j] = np.sign(slopex[i - 1, j]) * minslope
                        outdirslope[i, j] = slopex[i - 1, j]
                        outdirslopeNew[i, j] = slopexNew[i - 1, j]
                        adj_mask[i, j] = 0.5
                    elif dir_val == 3 and np.abs(slopey[i, j]) < minslope:
                        slopeyNew[i, j] = np.sign(slopey[i, j]) * minslope
                        outdirslopeNew[i, j] = slopeyNew[i, j]
                        adj_mask[i, j] = 0.5
                    elif dir_val == 4 and np.abs(slopex[i, j]) < minslope:
                        slopexNew[i, j] = np.sign(slopex[i, j]) * minslope
                        outdirslopeNew[i, j] = slopexNew[i, j]
                        adj_mask[i, j] = 0.5

                    # R: if(Remove.Sec==TRUE){ ... }
                    if remove_sec:
                        # R: if(max(sec_out)>0){
                        if np.max(sec_out) > 0:
                            # R: if(sec_out[1]>0){ slopeyNew[(i+kd[1,1]), (j+kd[1,2])]=0  adj_mask[i,j]=adj_mask[i,j]+1 }
                            if sec_out[0] > 0:
                                slopeyNew[i + int(kd[0, 0]), j + int(kd[0, 1])] = 0
                                adj_mask[i, j] = adj_mask[i, j] + 1
                            if sec_out[1] > 0:
                                slopexNew[i + int(kd[1, 0]), j + int(kd[1, 1])] = 0
                                adj_mask[i, j] = adj_mask[i, j] + 1
                            if sec_out[2] > 0:
                                slopeyNew[i + int(kd[2, 0]), j + int(kd[2, 1])] = 0
                                adj_mask[i, j] = adj_mask[i, j] + 1
                            if sec_out[3] > 0:
                                slopexNew[i + int(kd[3, 0]), j + int(kd[3, 1])] = 0
                                adj_mask[i, j] = adj_mask[i, j] + 1

    # R: output_list=list("slopex"=slopexNew, "slopey"=slopeyNew, "adj_mask"=adj_mask, "SlopeOutlet"=outdirslope, "SlopeOutletNew"=outdirslopeNew)
    output_list = {
        "slopex": slopexNew,
        "slopey": slopeyNew,
        "adj_mask": adj_mask,
        "SlopeOutlet": outdirslope,
        "SlopeOutletNew": outdirslopeNew,
    }
    return output_list
