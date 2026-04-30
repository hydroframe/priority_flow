"""
Find the distance to the nearest stream point following drainage directions.

Line-by-line translation of Stream_Distance.R (StreamDist) from the R
PriorityFlow package. R uses 1-based indexing; we use 0-based.
"""

import numpy as np
from typing import Dict, Optional, Tuple


def stream_dist(
    direction: np.ndarray,
    streammask: np.ndarray,
    domainmask: Optional[np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
) -> Dict[str, np.ndarray]:
    """
    Find stream distance using flow-direction traversal upstream from streams.
    """
    # HydroFrame layout -> internal R-style layout
    direction = direction.T.copy()
    streammask = streammask.T.copy()
    if domainmask is not None:
        domainmask = domainmask.T.copy()

    # R: nx=dim(direction)[1] ; ny=dim(direction)[2]
    nx = direction.shape[0]
    ny = direction.shape[1]

    # R:
    # ku=matrix(0, nrow=4, ncol=3)
    # ku[,1]=c(0,1,0,-1)
    # ku[,2]=c(1,0,-1,0)
    # ku[,3]=c(1,2,3,4)
    ku = np.zeros((4, 3), dtype=int)
    ku[:, 0] = [0, 1, 0, -1]
    ku[:, 1] = [1, 0, -1, 0]
    ku[:, 2] = [1, 2, 3, 4]

    # R: if(missing(domainmask)){domainmask=matrix(1, nrow=nx, ncol=ny)}
    if domainmask is None:
        domainmask = np.ones((nx, ny))

    # R: renumber directions to 1,2,3,4 if custom d4 was passed
    dir2 = direction.copy()
    if d4[0] != 1:
        dir2[direction == d4[0]] = 1
    if d4[1] != 2:
        dir2[direction == d4[1]] = 2
    if d4[2] != 3:
        dir2[direction == d4[2]] = 3
    if d4[3] != 4:
        dir2[direction == d4[3]] = 4

    # R: queue=which(streammask==1, arr.ind=T)
    # R arr.ind gives 1-based matrix indices; convert to 0-based.
    queue = np.argwhere(streammask == 1)

    # R: distance=matrix(NA, nrow=nx, ncol=ny)
    distance = np.full((nx, ny), np.nan)

    # R: distance[which(streammask==1)]=0
    distance[streammask == 1] = 0

    # R:
    # streamy=matrix(rep(1:ny,nx), ncol=ny, byrow=T)
    # streamx=matrix(rep(1:nx,ny), ncol=ny, byrow=F)
    # Keep R's 1-based index values in outputs.
    streamy = np.tile(np.arange(1, ny + 1), (nx, 1)).astype(float)
    streamx = np.tile(np.arange(1, nx + 1), (ny, 1)).T.astype(float)
    streamx[streammask == 0] = np.nan
    streamy[streammask == 0] = np.nan

    # R: active=TRUE
    active = True

    # R: while(active==T){ ... }
    while active:
        # Safety for empty queue (R version assumes non-empty)
        if queue.shape[0] == 0:
            break

        # R: indx=queue[1,1] ; indy=queue[1,2]
        indx = int(queue[0, 0])
        indy = int(queue[0, 1])

        # R: queuetemp=NULL
        queuetemp = []

        # R: for(d in 1:4){ ... }
        for d in range(4):
            # R: tempx=indx+ku[d,1] ; tempy=indy+ku[d,2]
            tempx = indx + ku[d, 0]
            tempy = indy + ku[d, 1]

            # R: if(tempx*tempy>0 & tempx<nx & tempy<ny){
            # 1-based strict interior bounds => 0-based: tempx>0,tempy>0,tempx<nx-1,tempy<ny-1
            if tempx > 0 and tempy > 0 and tempx < (nx - 1) and tempy < (ny - 1):
                # R: if((d-dir2[tempx,tempy])==0 & streammask==0 & domainmask==1)
                if (
                    ((d + 1) - dir2[tempx, tempy]) == 0
                    and streammask[tempx, tempy] == 0
                    and domainmask[tempx, tempy] == 1
                ):
                    # R: distance[tempx,tempy]=distance[indx,indy]+1
                    distance[tempx, tempy] = distance[indx, indy] + 1
                    streamx[tempx, tempy] = streamx[indx, indy]
                    streamy[tempx, tempy] = streamy[indx, indy]
                    # R: queuetemp=rbind(c(tempx,tempy),queuetemp)
                    queuetemp.insert(0, [tempx, tempy])

        # R: if(length(queuetemp>0)){ queue=rbind(queuetemp,queue[-1,]) } else ...
        if len(queuetemp) > 0:
            queuetemp_arr = np.array(queuetemp, dtype=int)
            if queue.shape[0] > 1:
                queue = np.vstack((queuetemp_arr, queue[1:, :]))
            else:
                queue = queuetemp_arr
        else:
            # R: if(nrow(queue)>1){ queue=queue[-1,] ... } else {active=F}
            if queue.shape[0] > 1:
                queue = queue[1:, :]
                # R: if(length(queue)==2){queue=matrix(queue, ncol=2, byrow=T)}
                if queue.ndim == 1:
                    queue = queue.reshape(1, 2)
            else:
                active = False

    # Internal layout -> HydroFrame layout
    output_list = {
        "stream.dist": distance.T,
        "stream.xind": streamx.T,
        "stream.yind": streamy.T,
    }
    return output_list