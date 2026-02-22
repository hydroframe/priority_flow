"""
Walk downstream from a point and extract values from a matrix.

Line-by-line translation of Downstream_Extract.R (PathExtract) from the R
PriorityFlow package. R uses 1-based indexing; we use 0-based. startpoint
is (row, col) 0-based for array indexing.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union

####################################################################
# PriorityFlow - Topographic Processing Toolkit for Hydrologic Models
# Copyright (C) 2018  Laura Condon (lecondon@email.arizona.edu)
# Contributors - Reed Maxwell (rmaxwell@mines.edu)
####################################################################


def path_extract(
    input: np.ndarray,
    direction: np.ndarray,
    mask: Optional[np.ndarray] = None,
    startpoint: Optional[Union[Tuple[int, int], list, np.ndarray]] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
) -> Dict[str, np.ndarray]:
    # R: nx=dim(direction)[1]  ny=dim(direction)[2]
    nx = direction.shape[0]
    ny = direction.shape[1]

    # R: if(missing(mask)){mask=matrix(1, nrow=nx, ncol=ny)}
    if mask is None:
        mask = np.ones((nx, ny))

    # R: path.mask=matrix(0, nrow=nx, ncol=ny)
    path_mask = np.zeros((nx, ny))
    # R: path=NULL  (list of cells on the path in order)
    path = []

    # D4 neighbors - Rows: down, left top right. Columns (1)deltax, (2)deltay
    # R: kd=matrix(0, nrow=4, ncol=4)  kd[,1]=c(0,-1,0,1)  kd[,2]=c(-1,0,1,0)
    kd = np.zeros((4, 2))
    kd[:, 0] = [0, -1, 0, 1]
    kd[:, 1] = [-1, 0, 1, 0]

    # renumber the directions to 1=down, 2=left, 3=up, 4=right if a different numbering scheme was used
    # R: dir2=direction
    dir2 = direction.copy()
    # R: if(d4[1]!=1){dir2[which(direction==d4[1])]=1}  etc.
    if d4[0] != 1:
        dir2[direction == d4[0]] = 1
    if d4[1] != 2:
        dir2[direction == d4[1]] = 2
    if d4[2] != 3:
        dir2[direction == d4[2]] = 3
    if d4[3] != 4:
        dir2[direction == d4[3]] = 4

    # initializing things
    # R: indx=startpoint[1]  indy=startpoint[2]  (R 1-based row, col)
    startpoint = np.asarray(startpoint)
    if startpoint.ndim >= 2:
        indx = int(startpoint.flat[0])
        indy = int(startpoint.flat[1])
    else:
        indx = int(startpoint[0])
        indy = int(startpoint[1])
    # R: step=1
    step = 1
    # R: active=T
    active = True
    # R: output=NULL
    output = []

    # walking downstream
    # R: while(active==T){
    while active:
        # R: output=c(output,input[indx,indy])
        output.append(input[indx, indy])
        # R: path.mask[indx,indy]=step
        path_mask[indx, indy] = step
        # R: path=rbind(path, c(indx,indy))
        path.append([indx, indy])

        # look downstream
        # R: dirtemp=dir2[indx,indy]
        dirtemp = int(dir2[indx, indy])
        # R: downindx=indx+kd[dirtemp,1]  downindy=indy+kd[dirtemp,2]
        # R uses 1-based direction (1..4); kd rows are 1..4 so in Python kd[dirtemp-1, :]
        downindx = indx + int(kd[dirtemp - 1, 0])
        downindy = indy + int(kd[dirtemp - 1, 1])

        # If you have made it out of the domain then stop
        # R: if(downindx<1 | downindx>nx | downindy<1 | downindy>ny){ active=F }
        # R 1-based: valid 1..nx, 1..ny. Python 0-based: valid 0..nx-1, 0..ny-1
        if downindx < 0 or downindx >= nx or downindy < 0 or downindy >= ny:
            active = False
        else:
            # R: if(mask[downindx,downindy]==0){ active=F }
            if mask[downindx, downindy] == 0:
                active = False

        # R: indx=downindx  indy=downindy  step=step+1
        indx = downindx
        indy = downindy
        step = step + 1

    # R: output_list=list("data"=output, "path.mask"=path.mask, "path.list"=path)
    path_list = np.array(path, dtype=np.int64) if path else np.zeros((0, 2), dtype=np.int64)
    output_list = {
        "data": np.array(output, dtype=input.dtype),
        "path_mask": path_mask,
        "path_list": path_list,
    }
    return output_list
