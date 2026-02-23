"""
Calculate slopes from a DEM using upwinding.

Translated line-by-line from PriorityFlow R package Slope_Calc_Upwind.R.
This function calculates slopes using standard or upwinding options and
applies a range of smoothing options, consistent with ParFlow OverlandFlow BC.
"""

from __future__ import annotations

import numpy as np


def slope_calc_upwind(
    dem: np.ndarray,
    direction: np.ndarray,
    dx: float,
    dy: float,
    mask: np.ndarray | None = None,
    borders: np.ndarray | None = None,
    borderdir: int = 1,
    d4: tuple[int, int, int, int] = (1, 2, 3, 4),
    minslope: float = 1e-5,
    maxslope: float = -1,
    secondary_th: float = -1,
    river_method: int = 0,
    river_secondary_th: float = 0,
    rivermask: np.ndarray | None = None,
    subbasins: np.ndarray | None = None,
    printflag: bool = False,
    upflag: bool = True,
) -> dict[str, np.ndarray]:
    """
    Calculate slopes from a DEM with upwinding consistent with ParFlow OverlandFlow.

    Parameters
    ----------
    dem : np.ndarray
        Digital elevation model (nx x ny).
    direction : np.ndarray
        Flow direction (1=down, 2=left, 3=up, 4=right). May be modified in place for borders.
    dx, dy : float
        Lateral grid cell resolution.
    mask : np.ndarray or None
        Domain mask (1=active, 0=inactive). If None, full rectangular domain is used.
    borders : np.ndarray or None
        Matrix with 1 for border cells default out, 2 default in, 0 for non-border. Built from mask if None.
    borderdir : int
        Default for border cells: 1=point out, 2=point in.
    d4 : tuple
        Direction codes (down, left, up, right). Default (1, 2, 3, 4).
    minslope : float
        Minimum absolute slope for flat cells. Use -1 to disable minimum enforcement.
    maxslope : float
        Maximum absolute slope. Use -1 to disable.
    secondary_th : float
        Max ratio |secondary|/|primary|. Use -1 to disable.
    river_method : int
        0=none, 1=scale secondary along river, 2=watershed mean slope, 3=stream mean slope.
    river_secondary_th : float
        Secondary ratio for river cells when river_method 1-3.
    rivermask : np.ndarray or None
        River mask (1=river). Required for river_method 1, 2, 3.
    subbasins : np.ndarray or None
        Subbasin ids. Required for river_method 2, 3.
    printflag : bool
        Print progress.
    upflag : bool
        If True, downwind slopes for OverlandFlow; if False, standard [i+1]-[i] (OverlandKin).

    Returns
    -------
    dict
        "slopex", "slopey", "direction" (nx x ny arrays).
    """
    ####################################################################
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
    ####################################################################

    d4 = tuple(d4)
    direction = np.asarray(direction, dtype=float).copy()
    secondaryTH = secondary_th
    river_secondaryTH = river_secondary_th

    # R: ny=ncol(dem)  nx=nrow(dem)
    ny = dem.shape[1]
    nx = dem.shape[0]

    # R: if(missing(mask)){ ... }
    if mask is None:
        print("No mask provided, initializing mask for complete domain")
        mask = np.ones((nx, ny))
        borders = np.zeros((nx, ny))
        borders[:, [0, ny - 1]] = 1 * borderdir
        borders[[0, nx - 1], :] = 1 * borderdir

    # R: if(missing(borders)){ ... }
    if borders is None:
        borders = np.ones((nx, ny))
        borders[1 : (nx - 1), 1 : (ny - 1)] = (
            mask[0 : (nx - 2), 1 : (ny - 1)]
            + mask[2:nx, 1 : (ny - 1)]
            + mask[1 : (nx - 1), 0 : (ny - 2)]
            + mask[1 : (nx - 1), 2:ny]
        )
        borders = borders * mask
        borders[np.where((borders < 4) & (borders != 0))] = 1 * borderdir
        borders[borders == 4] = 0

    # R: inmask=which(mask==1)  demMask=dem  demMask[-inmask]=NA
    demMask = dem.copy().astype(float)
    demMask[mask != 1] = np.nan

    # R: First pass - slopex = dem[i+1,j]-dem[i,j], slopey = dem[i,j+1]-dem[i,j]
    # R: slopex1=matrix(NA, ncol=ny, nrow=nx)  slopey1=...
    slopex1 = np.full((nx, ny), np.nan)
    slopey1 = np.full((nx, ny), np.nan)
    # R: slopex1[1:(nx-1),]=(demMask[2:nx,]-demMask[1:(nx-1),])/dx
    slopex1[0 : (nx - 1), :] = (demMask[1:nx, :] - demMask[0 : (nx - 1), :]) / dx
    # R: slopex1[nx,]=slopex1[(nx-1),]
    slopex1[nx - 1, :] = slopex1[nx - 2, :]
    # R: slopey1[,1:(ny-1)]=(demMask[,2:ny]-demMask[,1:(ny-1)])/dy
    slopey1[:, 0 : (ny - 1)] = (demMask[:, 1:ny] - demMask[:, 0 : (ny - 1)]) / dy
    # R: slopey1[,ny]=slopey1[,(ny-1)]
    slopey1[:, ny - 1] = slopey1[:, ny - 2]

    # R: Assign slopes based on upwinding for all non border cells
    if upflag:
        print("upwinding slopes")
        slopex2 = slopex1.copy()
        slopey2 = slopey1.copy()
        # R: for(j in 2:(ny-1)){ for(i in 2:(nx-1)){
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                # R: if(mask[i,j]==1 & borders[i,j]==0){
                if mask[i, j] == 1 and borders[i, j] == 0:
                    # X-Direction
                    # R: SINK: If slopex[i-1]<0 & slopex[i]>0  -> slope=0
                    if slopex1[i - 1, j] < 0 and slopex1[i, j] > 0:
                        slopex2[i, j] = 0
                        if direction[i, j] == d4[1] or direction[i, j] == d4[3]:  # 2=left, 4=right
                            print(
                                f"Problem! Local X sink found in primary direction. x= {i+1} j= {j+1}"
                            )

                    # R: Local Maximum: slopex[i-1]>=0 & slopex[i]<=0
                    if slopex1[i - 1, j] >= 0 and slopex1[i, j] <= 0:
                        if direction[i, j] == d4[1]:  # left -> use i-1 slope
                            slopex2[i, j] = slopex1[i - 1, j]
                        elif direction[i, j] == d4[3]:  # right -> use i slope
                            slopex2[i, j] = slopex1[i, j]
                        else:
                            if np.abs(slopex1[i - 1, j]) > np.abs(slopex1[i, j]):
                                slopex2[i, j] = slopex1[i - 1, j]

                    # R: Flow Through from right: slopex[i-1]>=0 & slopex[i]>=0
                    if slopex1[i - 1, j] >= 0 and slopex1[i, j] >= 0:
                        slopex2[i, j] = slopex1[i - 1, j]

                    # Y-Direction
                    # R: SINK: slopey[j-1]<0 & slopey[j]>0
                    if slopey1[i, j - 1] < 0 and slopey1[i, j] > 0:
                        slopey2[i, j] = 0
                        if direction[i, j] == d4[0] or direction[i, j] == d4[2]:  # 1=down, 3=up
                            print(
                                f"Problem! Local Y sink found in primary direction. x= {i+1} j= {j+1}"
                            )

                    # R: Local Maximum: slopey[j-1]>=0 & slopey[j]<=0
                    if slopey1[i, j - 1] >= 0 and slopey1[i, j] <= 0:
                        if direction[i, j] == d4[0]:  # down -> use j-1 slope
                            slopey2[i, j] = slopey1[i, j - 1]
                        elif direction[i, j] == d4[2]:  # up -> use j slope
                            slopey2[i, j] = slopey1[i, j]
                        else:
                            # NOTE: Matching original R behavior (even though it is a bit odd):
                            # If primary direction is in x, choose steepest *y* slope,
                            # but then assign the corresponding *x*-direction slope value.
                            # This preserves exact numerical parity with the R implementation.
                            if np.abs(slopey1[i, j - 1]) > np.abs(slopey1[i, j]):
                                slopey2[i, j] = slopex1[i, j - 1]

                    # R: Flow Through from top: slopey[j-1]>=0 & slopey[j]>=0
                    if slopey1[i, j - 1] >= 0 and slopey1[i, j] >= 0:
                        slopey2[i, j] = slopey1[i, j - 1]
    else:
        print("standard slope calc")
        slopex2 = slopex1.copy()
        slopey2 = slopey1.copy()

    # R: bordi=borders  bordi[bordi>1]=1  interior=mask-bordi
    bordi = borders.copy()
    bordi[bordi > 1] = 1
    interior = mask - bordi

    # R: Find the left border cells
    borderL = np.zeros((nx, ny))
    borderL[0, :] = 1
    borderL2 = borderL.copy()
    # R: borderL[2:(nx-1),]=(mask[2:(nx-1),]+mask[1:(nx-2),])*interior[3:nx,]
    borderL[1 : (nx - 1), :] = (
        (mask[1 : (nx - 1), :] + mask[0 : (nx - 2), :]) * interior[2:nx, :]
    )
    borderL = borderL * bordi
    borderL[borderL > 1] = 0
    borderL2[1 : (nx - 1), :] = (
        (mask[1 : (nx - 1), :] + mask[0 : (nx - 2), :]) * mask[2:nx, :]
    )
    borderL2 = borderL2 * bordi
    borderL2[borderL2 > 1] = 0
    borderL = borderL * 2
    borderL2 = borderL2 * 2

    # R: Find the right border cells
    borderR = np.zeros((nx, ny))
    borderR[nx - 1, :] = 1
    borderR2 = borderR.copy()
    borderR[1 : (nx - 1), :] = (
        (mask[1 : (nx - 1), :] + mask[2:nx, :]) * interior[0 : (nx - 2), :]
    )
    borderR = borderR * bordi
    borderR[borderR > 1] = 0
    borderR2[1 : (nx - 1), :] = (mask[1 : (nx - 1), :] + mask[2:nx, :]) * mask[0 : (nx - 2), :]
    borderR2 = borderR2 * bordi
    borderR2[borderR2 > 1] = 0
    borderR = borderR * 4
    borderR2 = borderR2 * 4

    # R: Find the lower (bottom) border cells
    borderB = np.zeros((nx, ny))
    borderB[:, 0] = 1
    borderB2 = borderB.copy()
    borderB[:, 1 : (ny - 1)] = (
        (mask[:, 1 : (ny - 1)] + mask[:, 0 : (ny - 2)]) * interior[:, 2:ny]
    )
    borderB = borderB * bordi
    borderB[borderB > 1] = 0
    borderB2[:, 1 : (ny - 1)] = (
        (mask[:, 1 : (ny - 1)] + mask[:, 0 : (ny - 2)]) * mask[:, 2:ny]
    )
    borderB2 = borderB2 * bordi
    borderB2[borderB2 > 1] = 0
    borderB = borderB * 1
    borderB2 = borderB2 * 1

    # R: Find the upper (top) border cells
    borderT = np.zeros((nx, ny))
    borderT[:, ny - 1] = 1
    borderT2 = borderT.copy()
    borderT[:, 1 : (ny - 1)] = (
        (mask[:, 1 : (ny - 1)] + mask[:, 2:ny]) * interior[:, 0 : (ny - 2)]
    )
    borderT = borderT * bordi
    borderT[borderT > 1] = 0
    borderT2[:, 1 : (ny - 1)] = (
        (mask[:, 1 : (ny - 1)] + mask[:, 2:ny]) * mask[:, 0 : (ny - 2)]
    )
    borderT2 = borderT2 * bordi
    borderT2[borderT2 > 1] = 0
    borderT = borderT * 3
    borderT2 = borderT2 * 3

    # R: borlist=which(bordi==1, arr.ind=T)  borlisti=which(bordi==1)
    borlist = np.argwhere(bordi == 1)
    borlisti = np.flatnonzero(bordi == 1)
    bordsum = borderB + borderT + borderL + borderR
    borddir = np.zeros((nx, ny))

    for k in range(borlist.shape[0]):
        i = int(borlist[k, 0])
        j = int(borlist[k, 1])
        if bordsum[i, j] > 0:
            borddir[i, j] = np.max(
                [borderB[i, j], borderT[i, j], borderL[i, j], borderR[i, j]]
            )
        else:
            borddir[i, j] = np.max(
                [borderB2[i, j], borderT2[i, j], borderL2[i, j], borderR2[i, j]]
            )

    # R: missinglist=which((borddir-bordi)<0)  missinglistA=which(..., arr.ind=T)
    missinglist = np.flatnonzero((borddir - bordi) < 0)
    missinglistA = np.argwhere((borddir - bordi) < 0)

    # R: down=up=left=right=matrix(0,...)  direction==d4[1] etc.
    down = np.zeros((nx, ny))
    left = np.zeros((nx, ny))
    up = np.zeros((nx, ny))
    right = np.zeros((nx, ny))
    down[direction == d4[0]] = 1
    left[direction == d4[1]] = 1
    up[direction == d4[2]] = 1
    right[direction == d4[3]] = 1
    draincount = np.zeros((nx, ny))
    draincount[:, 0 : (ny - 1)] = draincount[:, 0 : (ny - 1)] + down[:, 1:ny]
    draincount[:, 1:ny] = draincount[:, 1:ny] + up[:, 0 : (ny - 1)]
    draincount[0 : (nx - 1), :] = draincount[0 : (nx - 1), :] + left[1:nx, :]
    draincount[1:nx, :] = draincount[1:nx, :] + right[0 : (nx - 1), :]

    # R: drainout=which(draincount[borlisti]>0)  borders[borlisti[drainout]]=1
    drainout = np.where(draincount.flat[borlisti] > 0)[0]
    borders.flat[borlisti[drainout]] = 1

    # R: Calculate the slopes for border cells
    for k in range(borlist.shape[0]):
        i = int(borlist[k, 0])
        j = int(borlist[k, 1])

        if borddir[i, j] == 3:  # border on the top
            slopex2[i, j] = 0
            if borders[i, j] == 1:
                slopey2[i, j] = -np.abs(slopey1[i, j - 1])
                direction[i, j] = 3
            else:
                slopey2[i, j] = np.abs(slopey1[i, j - 1])
                direction[i, j] = 1

        if borddir[i, j] == 4:  # border on the right
            slopey2[i, j] = 0
            if borders[i, j] == 1:
                slopex2[i, j] = -np.abs(slopex1[i - 1, j])
                direction[i, j] = 4
            else:
                slopex2[i, j] = np.abs(slopex1[i - 1, j])
                direction[i, j] = 2

        if borddir[i, j] == 1:  # border on the bottom
            slopex2[i, j] = 0
            if borders[i, j] == 1:
                slopey2[i, j] = np.abs(slopey1[i, j])
                direction[i, j] = 1
            else:
                slopey2[i, j] = -np.abs(slopey1[i, j])
                direction[i, j] = 3

        if borddir[i, j] == 2:  # border on the left
            slopey2[i, j] = 0
            if borders[i, j] == 1:
                slopex2[i, j] = np.abs(slopex1[i, j])
                direction[i, j] = 2
            else:
                slopex2[i, j] = -np.abs(slopex1[i, j])
                direction[i, j] = 4

    # R: direction[missinglist]=4  slopex2[missinglist]=-minslope  slopey2[missinglist]=0
    direction.flat[missinglist] = 4
    slopex2.flat[missinglist] = -minslope
    slopey2.flat[missinglist] = 0

    # R: downlist=which(direction==d4[1])  etc.  ylist=c(uplist, downlist)  xlist=c(rightlist, leftlist)
    downlist = np.where(direction == d4[0])
    leftlist = np.where(direction == d4[1])
    uplist = np.where(direction == d4[2])
    rightlist = np.where(direction == d4[3])
    ylist = np.concatenate([uplist[0], downlist[0]]), np.concatenate(
        [uplist[1], downlist[1]]
    )
    xlist = np.concatenate([rightlist[0], leftlist[0]]), np.concatenate(
        [rightlist[1], leftlist[1]]
    )
    ymask = np.zeros((nx, ny))
    xmask = np.zeros((nx, ny))
    ymask[downlist] = 1
    ymask[uplist] = -1
    xmask[leftlist] = 1
    xmask[rightlist] = -1

    # R: if(maxslope>=0){ ... }
    if maxslope >= 0:
        print(f"Limiting slopes to maximum absolute value of {maxslope}")
        xclipP = np.where(slopex2 > maxslope)
        slopex2[xclipP] = maxslope
        xclipN = np.where(slopex2 < (-maxslope))
        slopex2[xclipN] = -maxslope
        yclipP = np.where(slopey2 > maxslope)
        slopey2[yclipP] = maxslope
        yclipN = np.where(slopey2 < (-maxslope))
        slopey2[yclipN] = -maxslope

    # R: if(secondaryTH>=0){ ... }
    if secondaryTH >= 0:
        print(f"Limiting the ratio of secondary to primary slopes {secondaryTH}")
        primary = np.abs(slopex2)
        primary[ylist] = np.abs(slopey2[ylist])
        secondary = np.abs(slopey2)
        secondary[ylist] = np.abs(slopex2[ylist])
        ratio = secondary / primary
        scalelist = np.where(ratio > secondaryTH)
        for idx in range(len(scalelist[0])):
            ti = scalelist[0][idx]
            tj = scalelist[1][idx]
            temp = np.ravel_multi_index((ti, tj), (nx, ny))
            if direction.flat[temp] == d4[1] or direction.flat[temp] == d4[3]:
                slopey2[ti, tj] = (
                    np.sign(slopey2[ti, tj]) * secondaryTH * np.abs(slopex2[ti, tj])
                )
            if direction.flat[temp] == d4[0] or direction.flat[temp] == d4[2]:
                slopex2[ti, tj] = (
                    np.sign(slopex2[ti, tj]) * secondaryTH * np.abs(slopey2[ti, tj])
                )

    # R: river_method==1
    if river_method == 1:
        if rivermask is None:
            print("WARNING: No rivermask provided, slopes not adjusted")
        else:
            print("River Method 1: setting secondary slopes to zero along the river")
            print(
                f"Scaling secondary slopes along river mask to {river_secondaryTH} * primary slope"
            )
            xtemp = slopex2.copy()
            xtemp_scaled = slopey2 * river_secondaryTH
            xtemp[ylist] = xtemp_scaled[ylist]
            ytemp = slopey2.copy()
            ytemp_scaled = slopex2 * river_secondaryTH
            ytemp[xlist] = ytemp_scaled[xlist]
            rivlist = np.where(rivermask == 1)
            slopex2[rivlist] = xtemp[rivlist]
            slopey2[rivlist] = ytemp[rivlist]

    # R: river_method==2
    if river_method == 2:
        if rivermask is None or subbasins is None:
            raise ValueError("River method 2 requires rivermask and subbasins")
        print("River Method 2: assigning average watershed slope to river cells by watershed")
        print(
            f"Scaling secondary slopes along river mask to {river_secondaryTH} * primary slope"
        )
        nbasin = int(np.max(subbasins))
        savg = np.zeros(nbasin + 1)
        count = np.zeros(nbasin + 1)
        xsign = np.sign(slopex2)
        ysign = np.sign(slopey2)
        xsign[xsign == 0] = 1
        ysign[ysign == 0] = 1
        for i in range(nx):
            for j in range(ny):
                if subbasins[i, j] > 0:
                    sb = int(subbasins[i, j])
                    savg[sb] = (
                        savg[sb]
                        + np.abs(slopex2[i, j]) * np.abs(xmask[i, j])
                        + np.abs(slopey2[i, j]) * np.abs(ymask[i, j])
                    )
                    count[sb] = count[sb] + 1
        for b in range(1, nbasin + 1):
            if count[b] > 0:
                savg[b] = savg[b] / count[b]
            savg[b] = max(minslope, savg[b])
        savg[np.isnan(savg)] = minslope
        rivlist = np.where(rivermask == 1)
        nriv = len(rivlist[0])
        for idx in range(nriv):
            ri = rivlist[0][idx]
            rj = rivlist[1][idx]
            rtemp = np.ravel_multi_index((ri, rj), (nx, ny))
            sbtemp = int(subbasins.flat[rtemp])
            if sbtemp > 0:
                slopex2[ri, rj] = savg[sbtemp] * xmask[ri, rj]
                slopey2[ri, rj] = savg[sbtemp] * ymask[ri, rj]
                slopex2[ri, rj] = (
                    slopex2[ri, rj]
                    + savg[sbtemp]
                    * np.abs(1 - np.abs(xmask[ri, rj]))
                    * xsign[ri, rj]
                    * river_secondaryTH
                )
                slopey2[ri, rj] = (
                    slopey2[ri, rj]
                    + savg[sbtemp]
                    * np.abs(1 - np.abs(ymask[ri, rj]))
                    * ysign[ri, rj]
                    * river_secondaryTH
                )

    # R: river_method==3
    if river_method == 3:
        if rivermask is None or subbasins is None:
            raise ValueError("River method 3 requires rivermask and subbasins")
        print("River Method 3: assigning average river slope to river cells by watershed")
        print(
            f"Scaling secondary slopes along river mask to {river_secondaryTH} * primary slope"
        )
        nbasin = int(np.max(subbasins))
        savg = np.zeros(nbasin + 1)
        count = np.zeros(nbasin + 1)
        xsign = np.sign(slopex2)
        ysign = np.sign(slopey2)
        xsign[xsign == 0] = 1
        ysign[ysign == 0] = 1
        for i in range(nx):
            for j in range(ny):
                if subbasins[i, j] > 0 and rivermask[i, j] == 1:
                    sb = int(subbasins[i, j])
                    savg[sb] = (
                        savg[sb]
                        + np.abs(slopex2[i, j]) * np.abs(xmask[i, j])
                        + np.abs(slopey2[i, j]) * np.abs(ymask[i, j])
                    )
                    count[sb] = count[sb] + 1
        for b in range(1, nbasin + 1):
            if count[b] > 0:
                savg[b] = savg[b] / count[b]
            savg[b] = max(minslope, savg[b])
        savg[np.isnan(savg)] = minslope
        rivlist = np.where(rivermask == 1)
        nriv = len(rivlist[0])
        for idx in range(nriv):
            ri = rivlist[0][idx]
            rj = rivlist[1][idx]
            rtemp = np.ravel_multi_index((ri, rj), (nx, ny))
            sbtemp = int(subbasins.flat[rtemp])
            if sbtemp > 0:
                slopex2[ri, rj] = savg[sbtemp] * xmask[ri, rj]
                slopey2[ri, rj] = savg[sbtemp] * ymask[ri, rj]
                slopex2[ri, rj] = (
                    slopex2[ri, rj]
                    + savg[sbtemp]
                    * np.abs(1 - np.abs(xmask[ri, rj]))
                    * xsign[ri, rj]
                    * river_secondaryTH
                )
                slopey2[ri, rj] = (
                    slopey2[ri, rj]
                    + savg[sbtemp]
                    * np.abs(1 - np.abs(ymask[ri, rj]))
                    * ysign[ri, rj]
                    * river_secondaryTH
                )

    # R: Check for flat cells
    nflat = np.sum((slopex2 == 0) & (slopey2 == 0))
    if nflat != 0:
        print(f"WARNING: {nflat} Flat cells found")
        flatloc = np.argwhere((slopex2 == 0) & (slopey2 == 0))
        flatlist = np.flatnonzero((slopex2 == 0) & (slopey2 == 0))
        if printflag:
            print("Flat locations (note this is x,y)")
            print(flatloc)
        for idx in range(len(flatlist)):
            dtemp = direction.flat[flatlist[idx]]
            if dtemp == d4[0]:
                slopey2.flat[flatlist[idx]] = minslope
            if dtemp == d4[1]:
                slopex2.flat[flatlist[idx]] = minslope
            if dtemp == d4[2]:
                slopey2.flat[flatlist[idx]] = -minslope
            if dtemp == d4[3]:
                slopex2.flat[flatlist[idx]] = -minslope
        nflat = np.sum((slopex2 == 0) & (slopey2 == 0))
        print(f"After processing: {nflat} Flat cells left")

    # R: if(minslope>=0){ ... }  (minimum slope clip for primary direction)
    if minslope >= 0:
        print(f"Limiting slopes to minimum {minslope}")
        xclipP = np.where(
            (slopex2 < minslope) & (slopex2 > 0) & (xmask == 1)
        )
        slopex2[xclipP] = minslope
        xclipN = np.where(
            (slopex2 > (-minslope)) & (slopex2 < 0) & (xmask == 1)
        )
        slopex2[xclipN] = -minslope
        yclipP = np.where(
            (slopey2 < minslope) & (slopey2 > 0) & (ymask == 1)
        )
        slopey2[yclipP] = minslope
        yclipN = np.where(
            (slopey2 > (-minslope)) & (slopey2 < 0) & (ymask == 1)
        )
        slopey2[yclipN] = -minslope

    # R: nax=which(is.na(slopex2==T))  slopex2[nax]=0  etc.
    slopex2[np.isnan(slopex2)] = 0
    slopey2[np.isnan(slopey2)] = 0

    return {
        "slopex": slopex2,
        "slopey": slopey2,
        "direction": direction,
    }
