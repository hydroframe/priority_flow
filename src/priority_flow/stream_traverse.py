"""
Stream Traverse functions for PriorityFlow.

Line-by-line translation of `Stream_Traverse.R` (StreamTraverse) from the
PriorityFlow R package.

This processes stream networks by walking upstream on D4 neighbors in a river
mask. Where no D4 neighbors exist, it looks for D8 neighbors and creates D4
bridges to these diagonal cells.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, Tuple


def stream_traverse(
    dem: np.ndarray,
    mask: np.ndarray,
    queue: np.ndarray,
    marked: np.ndarray,
    step: Optional[np.ndarray] = None,
    direction: Optional[np.ndarray] = None,
    basins: Optional[np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    printstep: bool = False,
    epsilon: float = 0.0,
) -> Dict[str, np.ndarray]:
    """
    DEM processing of stream networks (R: StreamTraverse).

    Parameters
    ----------
    dem : np.ndarray
        Digital Elevation Model matrix (nx x ny).
    mask : np.ndarray
        Mask with zeros for non-river cells and 1 for river cells.
    queue : np.ndarray
        Priority queue to start from; columns: x (row), y (col), elevation.
    marked : np.ndarray
        Matrix of which cells have been marked already.
    step : np.ndarray, optional
        Matrix of the step number for processed cells. Defaults to zeros.
    direction : np.ndarray, optional
        Matrix of flow directions for processed cells. Defaults to NaN.
    basins : np.ndarray, optional
        Matrix of basin numbers (e.g., from InitQueue). If provided, every
        newly added cell inherits the basin of the cell that adds it.
    d4 : tuple of int, optional
        D4 direction numbering (down, left, up, right). Default (1,2,3,4).
    printstep : bool, optional
        If True, print step number and queue size.
    epsilon : float, optional
        Amount to add to filled areas to avoid creating flats.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys: "dem", "mask", "marked", "step", "direction",
        and "basins".
    """
    # R: nx=dim(dem)[1]  ny=dim(dem)[2]
    nx, ny = dem.shape
    # R: demnew=dem
    demnew = dem.copy()

    # setup matrices for anything that wasn't input
    # R: if(missing(step)){step=matrix(0, nrow=nx, ncol=ny)}
    if step is None:
        step = np.zeros((nx, ny))
    # R: if(missing(direction)){direction=matrix(NA,nrow=nx, ncol=ny)}
    if direction is None:
        direction = np.full((nx, ny), np.nan)
    # R: if(missing(basins)){basins=matrix(0,nrow=nx, ncol=ny)}
    if basins is None:
        basins = np.zeros((nx, ny))

    # D4 neighbors (ordered down, left, top, right; walking upstream so
    # direction points opposite)
    # R: kd=matrix(0, nrow=4, ncol=3)
    # R: kd[,1]=c(0,-1,0,1)
    # R: kd[,2]=c(-1,0,1,0)
    # R: kd[,3]=c(d4[3], d4[4], d4[1], d4[2])
    kd = np.zeros((4, 3), dtype=float)
    kd[:, 0] = [0, -1, 0, 1]
    kd[:, 1] = [-1, 0, 1, 0]
    kd[:, 2] = [d4[2], d4[3], d4[0], d4[1]]

    # D8 neighbors
    # R: kd8=matrix(0, nrow=4, ncol=4)
    # R: kd8[,1]=c(-1,-1,1,1)
    # R: kd8[,2]=c(-1,1,1,-1)
    # R: kd8[,3]= c(d4[3], d4[1], d4[1], d4[3])
    # R: kd8[,4]= c(d4[4], d4[4], d4[2], d4[2])
    kd8 = np.zeros((4, 4), dtype=float)
    kd8[:, 0] = [-1, -1, 1, 1]
    kd8[:, 1] = [-1, 1, 1, -1]
    kd8[:, 2] = [d4[2], d4[0], d4[0], d4[2]]
    kd8[:, 3] = [d4[3], d4[3], d4[1], d4[1]]

    # R: nqueue=nrow(queue)  nstep=0
    # Ensure queue is 2D
    queue = np.asarray(queue, dtype=float)
    if queue.ndim == 1:
        queue = queue.reshape(1, -1)
    nqueue = queue.shape[0]
    nstep = 0

    # Main traversal loop
    # R: while(nqueue>0){
    while nqueue > 0:
        # #############
        # pick the lowest DEM cell on the queue
        # R: pick=which.min(queue[,3])
        pick = int(np.argmin(queue[:, 2]))
        # R: xC=queue[pick,1]  yC=queue[pick,2]  demC=queue[pick,3]
        xC = int(queue[pick, 0])
        yC = int(queue[pick, 1])
        demC = float(queue[pick, 2])

        # #############
        # Look for D4 neighbor cells that are on the mask and add to queue
        # R: count=0
        count = 0
        # R: for(k in 1:4){
        for k in range(4):
            # R: xk=xC+kd[k,1]  yk=yC+kd[k,2]
            xk = xC + int(kd[k, 0])
            yk = yC + int(kd[k, 1])
            # check that the neighbor is inside the domain and on the mask
            # R: if(yk>=1 & yk<=ny & xk>=1 & xk<=nx ){
            if 0 <= xk < nx and 0 <= yk < ny:
                # R: if (mask[xk, yk]==1 & marked[xk,yk]==0){
                if mask[xk, yk] == 1 and marked[xk, yk] == 0:
                    # R: demtemp=max((demC+epsilon), dem[xk, yk])
                    demtemp = max(demC + epsilon, dem[xk, yk])
                    # R: demnew[xk, yk]=demtemp
                    demnew[xk, yk] = demtemp
                    # R: queue=rbind(queue, c(xk, yk, demtemp))
                    queue = np.vstack([queue, [xk, yk, demtemp]])
                    # R: marked[xk, yk]=1
                    marked[xk, yk] = 1
                    # R: step[xk, yk]=step[xC, yC]+1
                    step[xk, yk] = step[xC, yC] + 1
                    # R: direction[xk,yk]=kd[k,3]
                    direction[xk, yk] = kd[k, 2]
                    # R: basins[xk,yk]=basins[xC,yC]
                    basins[xk, yk] = basins[xC, yC]
                    # R: count=count+1  nqueue=nqueue+1
                    count += 1
                    nqueue += 1

        # #############
        # If you don't find any D4 neighbors, look for D8 neighbors and choose
        # the least-cost D4 option to reach that D8 cell
        if count == 0:
            # R: n4=matrix(NA,ncol=4, nrow=2)
            for k in range(4):
                # R: xk=xC+kd8[k,1]  yk=yC+kd8[k,2]
                xk = xC + int(kd8[k, 0])
                yk = yC + int(kd8[k, 1])
                # R: count4=0  n4=matrix(NA,ncol=4, nrow=2)
                count4 = 0
                n4 = np.full((2, 4), np.nan)

                # check that the neighbor is inside the domain and on the mask
                # R: if(yk>=1 & yk<=ny & xk>=1 & xk<=nx ){
                if 0 <= xk < nx and 0 <= yk < ny:
                    # R: if(marked[xk,yk]<1 & mask[xk,yk]==1){
                    if marked[xk, yk] < 1 and mask[xk, yk] == 1:
                        # look for available D4 neighbors to add instead
                        # first neighbor: (xC, yk)
                        # R: if(marked[xC,yk]<1){ n4[1,]=c(xC, yk, dem[xC, yk], kd8[k,3]) }
                        if marked[xC, yk] < 1:
                            n4[0, :] = [xC, yk, dem[xC, yk], kd8[k, 2]]
                            count4 += 1
                        # second neighbor: (xk, yC)
                        # R: if(marked[xk,yC]<1){ n4[2,]=c(xk, yC, dem[xk, yC], kd8[k,4]) }
                        if marked[xk, yC] < 1:
                            n4[1, :] = [xk, yC, dem[xk, yC], kd8[k, 3]]
                            count4 += 1

                        # choose the neighbor which is the lowest without going under
                        # and if not, fill the highest
                        if count4 > 0:
                            valid = ~np.isnan(n4[:, 2])
                            if np.any(valid):
                                vals = n4[valid, 2]
                                idxs = np.where(valid)[0]
                                if np.min(vals) >= demC:
                                    # R: npick=which.min(n4[,3])
                                    npick = int(idxs[np.argmin(vals)])
                                else:
                                    # R: npick=which.max(n4[,3])
                                    npick = int(idxs[np.argmax(vals)])
                            else:
                                npick = 0

                            rx = int(n4[npick, 0])
                            ry = int(n4[npick, 1])
                            # R: demtemp=max((demC+epsilon), dem[n4[npick,1], n4[npick,2]])
                            demtemp = max(demC + epsilon, dem[rx, ry])
                            # R: demnew[n4[npick,1], n4[npick,2]]=demtemp
                            demnew[rx, ry] = demtemp
                            # R: queue=rbind(queue, c(n4[npick,1], n4[npick,2], demtemp))
                            queue = np.vstack([queue, [rx, ry, demtemp]])
                            # R: marked[n4[npick,1], n4[npick,2]]=1
                            marked[rx, ry] = 1
                            # R: direction[n4[npick,1], n4[npick,2]]=n4[npick,4]
                            direction[rx, ry] = n4[npick, 3]
                            # R: step[n4[npick,1], n4[npick,2]]=step[xC, yC]+1
                            step[rx, ry] = step[xC, yC] + 1
                            # R: basins[n4[npick,1], n4[npick,2]]=basins[xC, yC]
                            basins[rx, ry] = basins[xC, yC]
                            # R: count=count+1
                            count += 1

        # #############
        # Remove from the queue and move on
        # R: nqueue=length(queue)/3
        nqueue = queue.shape[0]
        # R: if(nqueue>1){ queue=queue[-pick,]; nqueue=length(queue)/3 } else { nqueue=0 }
        if nqueue > 1:
            queue = np.delete(queue, pick, axis=0)
            nqueue = queue.shape[0]
        else:
            nqueue = 0

        # if there is only one row, force 2D structure
        # R: if(nqueue==1){ queue=matrix(queue, ncol=3,byrow=T)}
        if nqueue == 1 and queue.ndim == 1:
            queue = queue.reshape(1, 3)

        # R: nstep=nstep+1
        nstep += 1
        # R: if(printstep){print(paste("Step:", nstep, "NQueue:", nqueue))}
        if printstep:
            print(f"Step: {nstep} NQueue: {nqueue}")

    # R: output_list=list("dem"=demnew, "mask"=mask, "marked"=marked, "step"= step, "direction"=direction, "basins"=basins)
    output_list = {
        "dem": demnew,
        "mask": mask,
        "marked": marked,
        "step": step,
        "direction": direction,
        "basins": basins,
    }
    # R: return(output_list)
    return output_list

