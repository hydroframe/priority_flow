"""
Calculate slopes from a DEM.

Line-by-line translation of Slope_Calc_Standard.R (SlopeCalStan) from the R
PriorityFlow package. R uses 1-based indexing; we use 0-based.
"""

import numpy as np
from typing import Dict, Optional, Tuple

####################################################################
# PriorityFlow - Topographic Processing Toolkit for Hydrologic Models
# Copyright (C) 2018  Laura Condon (lecondon@email.arizona.edu)
# Contributors - Reed Maxwell (rmaxwell@mines.edu)
####################################################################


def slope_calc_standard(
    dem: np.ndarray,
    direction: np.ndarray,
    dx: float,
    dy: float,
    mask: Optional[np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    minslope: float = 0,
    maxslope: float = -1,
    secondary_th: float = -1,
    printflag: bool = False,
) -> Dict[str, np.ndarray]:
    # R: ny=ncol(dem)  nx=nrow(dem)
    nx = dem.shape[0]
    ny = dem.shape[1]

    # If no mask is provided default to the rectangular domain
    # R: if(missing(mask)){ print(...); mask=matrix(1,...); borders=matrix(0,...) }
    if mask is None:
        if printflag:
            print("No mask provided, initializing mask for complete domain")
        mask = np.ones((nx, ny))
        borders = np.zeros((nx, ny))
    else:
        # R: Identify the border cells
        # R: borders=matrix(1, nrow=nx, ncol=ny)
        borders = np.ones((nx, ny))
        # R: borders[2:(nx-1), 2:(ny-1)]= mask[1:(nx-2), 2:(ny-1)] + mask[3:nx, 2:(ny-1)] + ...
        borders[1 : (nx - 1), 1 : (ny - 1)] = (
            mask[0 : (nx - 2), 1 : (ny - 1)]
            + mask[2:nx, 1 : (ny - 1)]
            + mask[1 : (nx - 1), 0 : (ny - 2)]
            + mask[1 : (nx - 1), 2:ny]
        )
        # R: borders=borders*mask  borders[which(borders<4 & borders!=0)]=1  borders[borders==4]=0
        borders = borders * mask
        borders[(borders < 4) & (borders != 0)] = 1
        borders[borders == 4] = 0

    # R: bordi=borders  bordi[bordi>1]=1  bordlist=which(borders>0)
    bordi = borders.copy()
    bordi[bordi > 1] = 1
    bordlist = np.where(borders > 0)

    # R: assign NA values to the DEM outside the mask
    # R: inmask=which(mask==1)  demMask=dem  demMask[-inmask]=NA
    demMask = dem.copy().astype(float)
    demMask[mask != 1] = np.nan

    # R: First pass calculate the x and y slopes as (i+1)-(i)
    # R: slopex1[1:(nx-1),]=(demMask[2:nx,]-demMask[1:(nx-1),])/dx
    slopex1 = np.full((nx, ny), np.nan)
    slopey1 = np.full((nx, ny), np.nan)
    slopex1[0 : (nx - 1), :] = (demMask[1:nx, :] - demMask[0 : (nx - 1), :]) / dx
    # R: slopey1[,1:(ny-1)]=(demMask[,2:ny]-demMask[,1:(ny-1)])/dy
    slopey1[:, 0 : (ny - 1)] = (demMask[:, 1:ny] - demMask[:, 0 : (ny - 1)]) / dy

    # R: slopex2=slopex1  slopey2=slopey1
    slopex2 = slopex1.copy()
    slopey2 = slopey1.copy()

    ### Right
    # R: borderR=slopeR=matrix(0, nrow=nx, ncol=ny)
    borderR = np.zeros((nx, ny))
    slopeR = np.zeros((nx, ny))
    # R: borderR[1:(nx-1),]=(mask[1:(nx-1),]+mask[2:nx,])*mask[1:(nx-1),]
    borderR[0 : (nx - 1), :] = (
        (mask[0 : (nx - 1), :] + mask[1:nx, :]) * mask[0 : (nx - 1), :]
    )
    # R: borderR[nx,]=1
    borderR[nx - 1, :] = 1
    # R: borderR=borderR*bordi  Rlist=which(borderR==1)  slopex2[Rlist]=0
    borderR = borderR * bordi
    Rlist = np.where(borderR == 1)
    slopex2[Rlist] = 0

    # R: borderR2=matrix(0,...)  borderR2[nx,]=1
    borderR2 = np.zeros((nx, ny))
    borderR2[nx - 1, :] = 1
    # R: borderR2[2:(nx-1),]=(mask[2:(nx-1),]+mask[3:nx,])*mask[1:(nx-2),]
    borderR2[1 : (nx - 1), :] = (
        (mask[1 : (nx - 1), :] + mask[2:nx, :]) * mask[0 : (nx - 2), :]
    )
    borderR2 = borderR2 * bordi
    borderR2[borderR2 > 1] = 0
    # R: slopeR[2:nx,]=slopex2[1:(nx-1),]*borderR2[2:nx,]
    slopeR[1:nx, :] = slopex2[0 : (nx - 1), :] * borderR2[1:nx, :]
    slopeR[np.isnan(slopeR)] = 0
    slopex2 = slopex2 + slopeR

    ### Top
    # R: borderT=slopeT=matrix(0,...)
    borderT = np.zeros((nx, ny))
    slopeT = np.zeros((nx, ny))
    # R: borderT[,1:(ny-1)]=(mask[,1:(ny-1)]+mask[,2:ny])*mask[,1:(ny-1)]
    borderT[:, 0 : (ny - 1)] = (
        (mask[:, 0 : (ny - 1)] + mask[:, 1:ny]) * mask[:, 0 : (ny - 1)]
    )
    # R: borderT[,ny]=1
    borderT[:, ny - 1] = 1
    borderT = borderT * bordi
    Tlist = np.where(borderT == 1)
    slopey2[Tlist] = 0

    # R: borderT2[,ny]=1  borderT2[,2:(ny-1)]=(mask[,2:(ny-1)]+mask[,3:ny])*mask[,1:(ny-2)]
    borderT2 = np.zeros((nx, ny))
    borderT2[:, ny - 1] = 1
    borderT2[:, 1 : (ny - 1)] = (
        (mask[:, 1 : (ny - 1)] + mask[:, 2:ny]) * mask[:, 0 : (ny - 2)]
    )
    borderT2 = borderT2 * bordi
    borderT2[borderT2 > 1] = 0
    # R: slopeT[,2:ny]=slopey2[,1:(ny-1)]*borderT2[,2:ny]
    slopeT[:, 1:ny] = slopey2[:, 0 : (ny - 1)] * borderT2[:, 1:ny]
    slopeT[np.isnan(slopeT)] = 0
    slopey2 = slopey2 + slopeT

    ######################################
    # R: Make masks of the primary flow directions
    # R: downlist=which(direction==d4[1])  etc.
    downlist = np.where(direction == d4[0])
    leftlist = np.where(direction == d4[1])
    uplist = np.where(direction == d4[2])
    rightlist = np.where(direction == d4[3])

    # R: downlist.arr=which(direction==d4[1], arr.ind=T)  etc.
    downlist_arr = np.argwhere(direction == d4[0])
    leftlist_arr = np.argwhere(direction == d4[1])
    uplist_arr = np.argwhere(direction == d4[2])
    rightlist_arr = np.argwhere(direction == d4[3])

    # R: xmask=ymask=matrix(0,...)  ymask[uplist]=-1  xmask[rightlist]=-1
    xmask = np.zeros((nx, ny))
    ymask = np.zeros((nx, ny))
    ymask[uplist] = -1
    xmask[rightlist] = -1
    # R: if(length(leftlist.arr>0)){ for(ii in 1:nrow(leftlist.arr)){ ... } }
    if leftlist_arr.shape[0] > 0:
        for ii in range(leftlist_arr.shape[0]):
            # R: xindex=max((leftlist.arr[ii,1]-1),1)
            xindex = max(leftlist_arr[ii, 0] - 1, 0)
            # R: if(mask[xindex, leftlist.arr[ii,2]]==0){xindex=xindex+1}
            if mask[xindex, leftlist_arr[ii, 1]] == 0:
                xindex = xindex + 1
            # R: xmask[xindex, leftlist.arr[ii,2]]=1
            xmask[xindex, leftlist_arr[ii, 1]] = 1
    if downlist_arr.shape[0] > 0:
        for ii in range(downlist_arr.shape[0]):
            # R: yindex=max((downlist.arr[ii,2]-1),1)
            yindex = max(downlist_arr[ii, 1] - 1, 0)
            # R: if(mask[downlist.arr[ii,1],yindex]==0){yindex=yindex+1}
            if mask[downlist_arr[ii, 0], yindex] == 0:
                yindex = yindex + 1
            # R: ymask[downlist.arr[ii,1], yindex]=1
            ymask[downlist_arr[ii, 0], yindex] = 1
    # R: ylist=which(ymask!=0)  xlist=which(xmask!=0)
    ylist = np.where(ymask != 0)
    xlist = np.where(xmask != 0)

    ###################################
    # R: fixPx=which(sign(slopex2)==-1 & xmask==1)  slopex2[fixPx]=abs(slopex2[fixPx])
    fixPx = np.where((np.sign(slopex2) == -1) & (xmask == 1))
    slopex2[fixPx] = np.abs(slopex2[fixPx])
    fixNx = np.where((np.sign(slopex2) == 1) & (xmask == -1))
    slopex2[fixNx] = -np.abs(slopex2[fixNx])
    fixPy = np.where((np.sign(slopey2) == -1) & (ymask == 1))
    slopey2[fixPy] = np.abs(slopey2[fixPy])
    fixNy = np.where((np.sign(slopey2) == 1) & (ymask == -1))
    slopey2[fixNy] = -np.abs(slopey2[fixNy])
    # R: Sinklist=c(fixPx, fixNx, fixPy, fixNy)  (linear indices)
    fixPx_flat = np.ravel_multi_index(fixPx, (nx, ny))
    fixNx_flat = np.ravel_multi_index(fixNx, (nx, ny))
    fixPy_flat = np.ravel_multi_index(fixPy, (nx, ny))
    fixNy_flat = np.ravel_multi_index(fixNy, (nx, ny))
    Sinklist = np.concatenate([fixPx_flat, fixNx_flat, fixPy_flat, fixNy_flat])

    ###################################
    # R: if(minslope>=0){ ... }
    if minslope >= 0:
        if printflag:
            print(f"Limiting slopes to minimum {minslope}")
        xclipP = np.where((np.abs(slopex2) < minslope) & (xmask == 1))
        slopex2[xclipP] = minslope
        xclipN = np.where((np.abs(slopex2) < minslope) & (xmask == -1))
        slopex2[xclipN] = -minslope
        yclipP = np.where((np.abs(slopey2) < minslope) & (ymask == 1))
        slopey2[yclipP] = minslope
        yclipN = np.where((np.abs(slopey2) < minslope) & (ymask == -1))
        slopey2[yclipN] = -minslope

    ###################################
    # R: if(maxslope>=0){ ... }
    if maxslope >= 0:
        if printflag:
            print(f"Limiting slopes to maximum absolute value of {maxslope}")
        xclipP = np.where(slopex2 > maxslope)
        slopex2[xclipP] = maxslope
        xclipN = np.where(slopex2 < -maxslope)
        slopex2[xclipN] = -maxslope
        yclipP = np.where(slopey2 > maxslope)
        slopey2[yclipP] = maxslope
        yclipN = np.where(slopey2 < -maxslope)
        slopey2[yclipN] = -maxslope

    ###################################
    # R: if(secondaryTH>=0){ if(secondaryTH==0){ slopex2[-xlist]=0  slopey2[-ylist]=0 } ... }
    if secondary_th >= 0:
        if secondary_th == 0:
            if printflag:
                print(f"Limiting the ratio of secondary to primary slopes {secondary_th}")
            # R: slopex2[-xlist]=0  (set non-primary x slopes to 0)
            slopex2[xmask == 0] = 0
            # R: slopey2[-ylist]=0
            slopey2[ymask == 0] = 0
        else:
            if printflag:
                print(
                    "Options for nonzero secondary scaling not currently available please set secondaryTH to -1 or 0"
                )

    ###################################
    # R: flattest=abs(slopex2[2:nx,2:ny])+abs(slopex2[1:(nx-1),2:ny])+abs(slopey2[2:nx,2:ny])+abs(slopey2[2:nx,1:(ny-1)])
    flattest = (
        np.abs(slopex2[1:nx, 1:ny])
        + np.abs(slopex2[0 : (nx - 1), 1:ny])
        + np.abs(slopey2[1:nx, 1:ny])
        + np.abs(slopey2[1:nx, 0 : (ny - 1)])
    )
    # R: nflat=length(which(flattest==0 & mask[2:nx, 2:ny]))
    nflat = np.sum((flattest == 0) & (mask[1:nx, 1:ny] == 1))
    # R: flats=which(flattest==0, arr.ind=T)+1  (R +1 for 1-based; we keep 0-based)
    flats = np.argwhere(flattest == 0)
    if nflat > 0 and printflag:
        print(f"WARNING: {nflat} Flat cells found")

    ###########################
    # R: nax=which(is.na(slopex2==T))  slopex2[nax]=0  etc.
    slopex2[np.isnan(slopex2)] = 0
    slopey2[np.isnan(slopey2)] = 0

    # R: output_list=list("slopex"=slopex2, "slopey"=slopey2, "direction"=direction, "Sinks"=Sinklist)
    output_list = {
        "slopex": slopex2,
        "slopey": slopey2,
        "direction": direction,
        "Sinks": Sinklist,
    }
    return output_list
