"""
Priority flow processing of D4 stream networks.

Line-by-line translation of D4_Traverse.R (D4TraverseB) from the R PriorityFlow package.
R uses 1-based indexing; we use 0-based. Queue has columns (row, col, elevation).
"""

import numpy as np
from typing import Dict, Optional, Tuple

####################################################################
# PriorityFlow - Topographic Processing Toolkit for Hydrologic Models
# Copyright (C) 2018  Laura Condon (lecondon@email.arizona.edu)
# Contributors - Reed Maxwell (rmaxwell@mines.edu)
####################################################################


def d4_traverse_b(
    dem: np.ndarray,
    queue: np.ndarray,
    marked: np.ndarray,
    mask: Optional[np.ndarray] = None,
    step: Optional[np.ndarray] = None,
    direction: Optional[np.ndarray] = None,
    basins: Optional[np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    printstep: bool = False,
    nchunk: int = 100,
    epsilon: float = 0.0,
    printflag: bool = False,
    *,
    n_chunk: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    D4TraverseB: process all network cells walking upstream on D4 neighbors.
    Optional n_chunk for API compatibility (overrides nchunk if given).
    """
    if n_chunk is not None:
        nchunk = n_chunk

    # R: t0=proc.time()
    # R: nx=dim(dem)[1]  ny=dim(dem)[2]
    nx = dem.shape[0]
    ny = dem.shape[1]
    # R: demnew=dem
    demnew = dem.copy()

    # setup matrices for anything that wasn't input
    # R: if(missing(mask)){mask=matrix(1, nrow=nx, ncol=ny)}
    if mask is None:
        mask = np.ones((nx, ny))
    # R: if(missing(step)){step=matrix(0, nrow=nx, ncol=ny)}
    if step is None:
        step = np.zeros((nx, ny))
    # R: if(missing(direction)){direction=matrix(NA,nrow=nx, ncol=ny)}
    if direction is None:
        direction = np.full((nx, ny), np.nan)
    # R: if(missing(basins)){basins=matrix(0,nrow=nx, ncol=ny)}
    if basins is None:
        basins = np.zeros((nx, ny))

    # D4 neighbors
    # R: kd=matrix(0, nrow=4, ncol=3) #ordered down, left top right
    kd = np.zeros((4, 3))
    # R: kd[,1]=c(0,-1,0,1)  kd[,2]=c(-1,0,1,0)
    kd[:, 0] = [0, -1, 0, 1]
    kd[:, 1] = [-1, 0, 1, 0]
    # R: kd[,3]=c(d4[3], d4[4], d4[1], d4[2])  # walking upstream so direction points opposite
    kd[:, 2] = [d4[2], d4[3], d4[0], d4[1]]

    # R: split=0  q1max=0  nqueue=nrow(queue)  nstep=0  queuetemp=NULL
    split = 0
    q1max = 0
    nqueue = queue.shape[0]  # R: nrow(queue)
    nstep = 0
    queuetemp = None  # R: NULL

    # split the queue in 2 using the top nchunk values for the first queue and the rest for the second
    # R: if(nqueue>nchunk){
    if nqueue > nchunk:
        # R: qsort=queue[order(queue[,3]),]
        qsort = queue[np.argsort(queue[:, 2], kind="stable")]
        # R: queue1=qsort[1:nchunk,]  queue2=qsort[-(1:nchunk),]
        queue1 = qsort[:nchunk].copy()
        queue2 = qsort[nchunk:].copy()
        # R: th=queue2[1,3]
        th = float(queue2[0, 2])
        # R: nqueue2=length(queue2)/3
        nqueue2 = queue2.shape[0]
        # R: nqueue=nrow(queue1)
        nqueue = queue1.shape[0]
        # R: if(printstep){print(paste(...))}
        if printstep:
            print(
                "inital queue:",
                queue.shape[0],
                "splitting. Q1=",
                nqueue,
                "Q2=",
                nqueue2,
            )
        # R: if(nqueue2==1){ queue2=matrix(queue2, ncol=3,byrow=T) }
        if nqueue2 == 1:
            queue2 = queue2.reshape(1, 3)
    # R: } else{
    else:
        # R: print(paste('inital queue:', nqueue, "Not splitting"))
        print("inital queue:", nqueue, "Not splitting")
        # R: queue1=queue  queue2=NULL  nqueue2=0  th=queue1[nqueue,3]*1.1
        queue1 = queue.copy()
        queue2 = None
        nqueue2 = 0
        th = float(queue1[nqueue - 1, 2]) * 1.1
    # R: t0=proc.time()

    # R: while(nqueue>0){
    while nqueue > 0:
        # pick the lowest DEM cell on the queue
        # R: pick=which.min(queue1[,3])
        pick = int(np.argmin(queue1[:, 2]))
        # R: xC=queue1[pick,1]  yC=queue1[pick,2]  demC=queue1[pick,3]
        xC = int(queue1[pick, 0])
        yC = int(queue1[pick, 1])
        demC = float(queue1[pick, 2])

        # Look for D4 neighbor cells that are on the mask and add to queue
        # R: count=0
        count = 0
        # R: bdrchk=direction[xC,yC]
        bdrchk = direction[xC, yC]

        # R: for(k in 1:4){
        for k in range(4):
            # R: xk=xC+kd[k,1]  yk=yC+kd[k,2]
            xk = int(xC + kd[k, 0])
            yk = int(yC + kd[k, 1])

            # check that the neighbor is inside the domain and on the mask
            # R: if(yk>=1 & yk<=ny & xk>=1 & xk<=nx ){
            if 0 <= yk < ny and 0 <= xk < nx:
                # R: if (mask[xk, yk]==1 & marked[xk,yk]==0){
                if mask[xk, yk] == 1 and marked[xk, yk] == 0:
                    # R: demtemp=max((demC+epsilon), dem[xk, yk])
                    demtemp = max(demC + epsilon, float(dem[xk, yk]))
                    # R: demnew[xk, yk]=demtemp
                    demnew[xk, yk] = demtemp
                    # R: if(demtemp<th){ queue1=rbind(queue1, c(xk, yk, demtemp))
                    #     } else{ queuetemp=rbind(queuetemp, c(xk, yk, demtemp)) }
                    if demtemp < th:
                        queue1 = np.vstack([queue1, [[xk, yk, demtemp]]])
                    else:
                        if queuetemp is None:
                            queuetemp = np.array([[xk, yk, demtemp]])
                        else:
                            queuetemp = np.vstack([queuetemp, [[xk, yk, demtemp]]])

                    # R: marked[xk, yk]=1  step[xk, yk]=step[xC, yC]+1  direction[xk,yk]=kd[k,3]  basins[xk,yk]=basins[xC,yC]
                    marked[xk, yk] = 1
                    step[xk, yk] = step[xC, yC] + 1
                    direction[xk, yk] = kd[k, 2]
                    basins[xk, yk] = basins[xC, yC]
                    count += 1

                    # if the original cell is on the border and lacking a flow direction then give it the direction of the cell it just added
                    # R: if(is.na(bdrchk)==T){
                    if np.isnan(bdrchk):
                        # R: xO=xC-kd[k,1]  yO=yC-kd[k,2]
                        xO = int(xC - kd[k, 0])
                        yO = int(yC - kd[k, 1])
                        # R: if(yO*xO==0 | yO>ny | xO>nx ){direction[xC,yC]=kd[k,3]}else{ if(mask[xO,yO]==0){direction[xC,yC]=kd[k,3]} }
                        if xO < 0 or yO < 0 or xO >= nx or yO >= ny:
                            direction[xC, yC] = kd[k, 2]
                        else:
                            if mask[xO, yO] == 0:
                                direction[xC, yC] = kd[k, 2]

        # Remove from the queue and move on
        # R: nqueue=length(queue1)/3
        # R: nqueuetemp=nqueue2+length(queuetemp)/3
        nqueue = queue1.shape[0]
        nqueuetemp = nqueue2 + (queuetemp.shape[0] if queuetemp is not None else 0)

        # R: if(nqueue>1){
        if nqueue > 1:
            # R: queue1=queue1[-pick,]
            queue1 = np.delete(queue1, pick, axis=0)
            # R: nqueue=length(queue1)/3  q1max=max(nqueue, q1max)
            nqueue = queue1.shape[0]
            q1max = max(nqueue, q1max)
        # R: } else {
        else:
            # look and see if there are still values in Q2 to merge
            # R: if(nqueuetemp>nchunk){
            if nqueuetemp > nchunk:
                # R: split=split+1  print(...)  queue2=rbind(queue2, queuetemp)  queuetemp=NULL
                split += 1
                if printstep:
                    print(
                        "P", split, "Q2", nqueuetemp, "nstep=", nstep, "Q1 Max:", q1max
                    )
                if queuetemp is not None and queuetemp.size > 0:
                    if queue2 is None:
                        queue2 = queuetemp
                    else:
                        queue2 = np.vstack([queue2, queuetemp])
                queuetemp = None
                # R: qsort=queue2[order(queue2[,3]),]  queue1=qsort[1:nchunk,]  queue2=qsort[-(1:nchunk),]  th=queue2[1,3]
                qsort = queue2[np.argsort(queue2[:, 2], kind="stable")]
                queue1 = qsort[:nchunk].copy()
                queue2 = qsort[nchunk:].copy()
                th = float(queue2[0, 2])
                nqueue = queue1.shape[0]
                nqueue2 = queue2.shape[0]
                q1max = 0
                # R: t0=proc.time()
            # R: } else if(nqueuetemp<=nchunk & nqueue2 >0){
            elif nqueuetemp <= nchunk and nqueue2 > 0:
                split += 1
                if printstep:
                    print(
                        "Split:", split, "Q2", nqueuetemp, "taking last chunk, nstep=", nstep, "Q1 Max:", q1max
                    )
                # R: queue1=rbind(queue2, queuetemp)  queuetemp=NULL  queue2=NULL  th=max(dem[mask==1])*1.1
                parts = [queue2]
                if queuetemp is not None and queuetemp.size > 0:
                    parts.append(queuetemp)
                queue1 = np.vstack(parts)
                queuetemp = None
                queue2 = None
                th = float(np.max(dem[mask == 1])) * 1.1
                nqueue = queue1.shape[0]
                q1max = 0
                nqueue2 = 0
            # R: } else {
            else:
                if printstep:
                    print("Q1 depleted, Q2", nqueue2, "done!!")
                nqueue = 0

        # R: if(nqueue==1){ queue1=matrix(queue1, ncol=3,byrow=T) }
        if nqueue == 1:
            queue1 = queue1.reshape(1, 3)

        nstep += 1
        if printstep:
            nq2 = nqueue2
            nqt = queuetemp.shape[0] if queuetemp is not None else 0
            print("Step:", nstep, "NQueue:", nqueue, "Queue2:", nqueue2, "Qtemp:", nqt)

    # after its all done do a final pass to fill in flow directions for any cells on the initial queue which didn't get a flow direction
    # R: bordermiss=which(is.na(direction)==T & mask==1, arr.ind=T)
    bordermiss = np.argwhere(np.isnan(direction) & (mask == 1))

    # R: dem_pad=dem  dem_pad=rbind(dem[1,],dem_pad, dem[nx,])  dem_pad=cbind(dem_pad[,1],dem_pad, dem_pad[,ny])
    dem_pad = np.vstack([dem[0:1], dem, dem[-1:]])
    dem_pad = np.hstack([dem_pad[:, 0:1], dem_pad, dem_pad[:, -1:]])

    # R: if(length(bordermiss>0)){
    if bordermiss.shape[0] > 0:
        # R: for(b in 1:nrow(bordermiss)){
        for b in range(bordermiss.shape[0]):
            # R: bx=bordermiss[b,1]+1  by=bordermiss[b,2]+1  (1-based index into dem_pad; dem_pad has padding so dem[i,j]=dem_pad[i+1,j+1])
            bx = int(bordermiss[b, 0]) + 1
            by = int(bordermiss[b, 1]) + 1
            # R: dem_negh=c((dem[bx-1,by-1] - dem_pad[bx+kd[1,1], by+kd[1,2]]), ...)
            dem_negh = np.array(
                [
                    dem[bx - 1, by - 1] - dem_pad[int(bx + kd[0, 0]), int(by + kd[0, 1])],
                    dem[bx - 1, by - 1] - dem_pad[int(bx + kd[1, 0]), int(by + kd[1, 1])],
                    dem[bx - 1, by - 1] - dem_pad[int(bx + kd[2, 0]), int(by + kd[2, 1])],
                    dem[bx - 1, by - 1] - dem_pad[int(bx + kd[3, 0]), int(by + kd[3, 1])],
                ]
            )
            # R: pick=which.max(abs(dem_negh))
            pick = int(np.argmax(np.abs(dem_negh)))
            # R: if(dem_negh[pick]<0){ direction[bx-1,by-1]=kd[pick,3] }else{ direction[bx-1,by-1]=d4[pick] }
            if dem_negh[pick] < 0:
                direction[bx - 1, by - 1] = kd[pick, 2]
            else:
                direction[bx - 1, by - 1] = d4[pick]

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
