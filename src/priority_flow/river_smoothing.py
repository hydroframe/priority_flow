"""
Apply smoothing to a DEM along a pre-defined stream network.

Line-by-line translation of River_Smoothing.R (RiverSmooth) from the R
PriorityFlow package. R uses 1-based indexing; we use 0-based. Summary
coordinates in river_summary are 0-based (row, col) for array indexing.
"""

import numpy as np
from typing import Dict, Optional, Tuple

from .fix_drainage import fix_drainage

####################################################################
# PriorityFlow - Topographic Processing Toolkit for Hydrologic Models
# Copyright (C) 2018  Laura Condon (lecondon@email.arizona.edu)
# Contributors - Reed Maxwell (rmaxwell@mines.edu)
####################################################################


def river_smooth(
    dem: np.ndarray,
    direction: np.ndarray,
    mask: Optional[np.ndarray] = None,
    river_summary: Optional[np.ndarray] = None,
    river_segments: Optional[np.ndarray] = None,
    bank_epsilon: float = 0.01,
    river_epsilon: float = 0.0,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    printflag: bool = False,
) -> Dict[str, np.ndarray]:
    # R: nx=dim(direction)[1]  ny=dim(direction)[2]  (set after dir2)
    nx = direction.shape[0]
    ny = direction.shape[1]

    # D4 neighbors - Rows: down, left top right. Columns: (1)deltax, (2)deltay
    # R: kd=matrix(0, nrow=4, ncol=4)  kd[,1]=c(0,-1,0,1)  kd[,2]=c(-1,0,1,0)
    kd = np.zeros((4, 2))
    kd[:, 0] = [0, -1, 0, 1]
    kd[:, 1] = [-1, 0, 1, 0]

    # R: if(missing(mask)){mask=matrix(1, nrow=nx, ncol=ny)}
    if mask is None:
        mask = np.ones((nx, ny))

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

    # setup a river list
    # R: nriver=nrow(river.summary)
    nriver = river_summary.shape[0]
    # R: marked.segments=rep(0,nriver)
    marked_segments = np.zeros(nriver)
    # R: marked.matrix=matrix(0, ncol=ny, nrow=nx)
    marked_matrix = np.zeros((nx, ny))

    # Setup a smoothing summary
    # R: riversmooth.summary=matrix(0, nrow=nriver, ncol=9)
    riversmooth_summary = np.zeros((nriver, 9))
    # R: riversmooth.summary[,1:5]=river.summary[,1:5]
    riversmooth_summary[:, 0:5] = river_summary[:, 0:5]

    # make a mask of the hillslope cells
    # R: hillmask=mask  hillmask[which(river.segments>0)]=0
    hillmask = mask.copy()
    hillmask[river_segments > 0] = 0

    # First make a list of all the terminal river reaches
    # R: queue=which(river.summary[,6]<=(0))
    queue = np.where(river_summary[:, 5] <= 0)[0]
    # R: if(length(queue)>0){active=TRUE}else{print("No terminal river segments...")}
    if queue.size > 0:
        active = True
    else:
        print("No terminal river segments provided, not adjusting DEM")
        active = False

    # R: dem2=dem
    dem2 = dem.copy()

    # get the length of every river segment
    # R: river.length=rep(0,max(river.summary[,1]))
    max_basin_id = int(np.max(river_summary[:, 0]))
    river_length = np.zeros(max_basin_id + 1)
    # R: for(i in 1:nx){ for(j in 1:ny){ rtemp=river.segments[i,j]; river.length[rtemp]=river.length[rtemp]+1 }}
    for i in range(nx):
        for j in range(ny):
            rtemp = int(river_segments[i, j])
            if rtemp > 0:
                river_length[rtemp] = river_length[rtemp] + 1

    # Loop over the river segments working upstream
    while active:
        # R: indr=queue[1]
        indr = int(queue[0])
        # R: r=river.summary[indr,1]
        r = int(river_summary[indr, 0])
        # R: rdown=river.summary[indr,6]
        rdown = river_summary[indr, 5]
        # R: length=river.length[r]
        seg_length = int(river_length[r])
        # R: riversmooth.summary[indr,6]=river.length[r]
        riversmooth_summary[indr, 5] = river_length[r]

        # find the top and bottom elevations of the current river segment
        # R: top=dem2[river.summary[indr,2], river.summary[indr,3]]
        top = dem2[
            int(river_summary[indr, 1]),
            int(river_summary[indr, 2]),
        ]
        # R: if(rdown<=0){ bottom=... length=length-1 } else{ bdir=... bottom=... }
        if rdown <= 0:
            # R: bottom=dem2[river.summary[indr,4], river.summary[indr,5]]
            bottom = dem2[
                int(river_summary[indr, 3]),
                int(river_summary[indr, 4]),
            ]
            seg_length = seg_length - 1
        else:
            # R: bdir=dir2[river.summary[indr,4], river.summary[indr,5]]
            bdir = int(
                dir2[
                    int(river_summary[indr, 3]),
                    int(river_summary[indr, 4]),
                ]
            )
            # R: bottom=dem2[(river.summary[indr,4]+kd[bdir,1]), (river.summary[indr,5]+kd[bdir,2])]
            # R uses 1-based dir (1..4); kd rows are 1..4 in R, so kd[bdir,] in Python is kd[bdir-1,]
            bottom = dem2[
                int(river_summary[indr, 3]) + int(kd[bdir - 1, 0]),
                int(river_summary[indr, 4]) + int(kd[bdir - 1, 1]),
            ]

        # R: topmin=bottom+river.epsilon*length
        topmin = bottom + river_epsilon * seg_length
        # R: if(top<topmin){
        if top < topmin:
            # R: top0=dem[river.summary[indr,2], river.summary[indr,3]]
            top0 = dem[
                int(river_summary[indr, 1]),
                int(river_summary[indr, 2]),
            ]
            if rdown > 0:
                bdir = int(
                    dir2[
                        int(river_summary[indr, 3]),
                        int(river_summary[indr, 4]),
                    ]
                )
                bottom0 = dem[
                    int(river_summary[indr, 3]) + int(kd[bdir - 1, 0]),
                    int(river_summary[indr, 4]) + int(kd[bdir - 1, 1]),
                ]
            else:
                bottom0 = dem[
                    int(river_summary[indr, 3]),
                    int(river_summary[indr, 4]),
                ]
            # R: delta=max((top0-bottom0)/(length),river.epsilon)
            delta = max((top0 - bottom0) / seg_length, river_epsilon)
            # R: top=bottom+delta*length
            top = bottom + delta * seg_length
            # R: dem2[river.summary[indr,2], river.summary[indr,3]]=top
            dem2[
                int(river_summary[indr, 1]),
                int(river_summary[indr, 2]),
            ] = top
            if printflag:
                print(
                    f"River top elevation<river bottom elevation for segment {r}"
                )
                print(
                    f"Original top {round(top0, 2)} and original bottom {round(bottom0, 2)}"
                )
                print(
                    f"Adjusting the top elevation from {round(top0, 2)} to {round(top, 2)}"
                )

        if printflag:
            print(f"River segment: {r}")
            print(
                f"Start: {river_summary[indr, 1]} {river_summary[indr, 2]} {round(top, 1)}"
            )
            print(
                f"End: {river_summary[indr, 3]} {river_summary[indr, 4]} {round(bottom, 1)}"
            )

        # walk from top to bottom smoothing out the river cells
        # R: indx=river.summary[indr,2]  indy=river.summary[indr,3]
        indx = int(river_summary[indr, 1])
        indy = int(river_summary[indr, 2])
        # R: marked.matrix[indx,indy]=marked.matrix[indx,indy]+1
        marked_matrix[indx, indy] = marked_matrix[indx, indy] + 1

        # R: if(length>1){delta=(top-bottom)/(length)}else{delta=0}
        if seg_length > 1:
            delta = (top - bottom) / seg_length
        else:
            delta = 0.0
        # R: if(delta<0){ print(...); delta=0 }
        if delta < 0:
            print(
                f"Warning: Calculated delta < 0, setting delta to 0 for segment {r}"
            )
            delta = 0.0
        # R: temp=top
        temp = top
        # R: riversmooth.summary[indr,7]=top  [8]=bottom  [9]=delta
        riversmooth_summary[indr, 6] = top
        riversmooth_summary[indr, 7] = bottom
        riversmooth_summary[indr, 8] = delta

        # R: if(length>1){
        if seg_length > 1:
            # R: for(i in 2:length){
            for i in range(1, seg_length):
                # R: temp=temp-delta
                temp = temp - delta
                # R: dirtemp=dir2[indx,indy]
                dirtemp = int(dir2[indx, indy])
                # R: downindx=indx+kd[dirtemp,1]  downindy=indy+kd[dirtemp,2]
                downindx = indx + int(kd[dirtemp - 1, 0])
                downindy = indy + int(kd[dirtemp - 1, 1])
                # R: if(river.segments[downindx,downindy]==r){
                if river_segments[downindx, downindy] == r:
                    # R: dem2[downindx,downindy]=temp
                    dem2[downindx, downindy] = temp
                    # R: marked.matrix[downindx,downindy]=...
                    marked_matrix[downindx, downindy] = (
                        marked_matrix[downindx, downindy] + 1
                    )
                    # R: drainfix=FixDrainage(dem=dem2, direction=dir2, mask=hillmask, bank.epsilon=bank.epsilon, startpoint=c(downindx,downindy))
                    # R: dem2=drainfix$dem.adj
                    drainfix = fix_drainage(
                        dem=dem2,
                        direction=dir2,
                        mask=hillmask,
                        bank_epsilon=bank_epsilon,
                        startpoint=(downindx, downindy),
                        d4=d4,
                    )
                    dem2 = drainfix["dem.adj"]
                else:
                    print(f"Warning: Check Segment for branches {r}")
                # R: indx=downindx  indy=downindy
                indx = downindx
                indy = downindy

        # R: marked.segments[indr]= marked.segments[indr]+1
        marked_segments[indr] = marked_segments[indr] + 1

        # Find all of the river segments that drain to this segment
        # R: uplist=which(river.summary[,6]==r)
        uplist = np.where(river_summary[:, 5] == r)[0]
        # R: if(length(uplist>0)){queue=c(uplist,queue[-1])} else{queue=queue[-1]}
        # (R has typo length(uplist>0) should be length(uplist)>0)
        if uplist.size > 0:
            queue = np.concatenate([uplist, queue[1:]])
        else:
            queue = queue[1:]
        # R: if(length(queue)==0){active=F}
        if queue.size == 0:
            active = False

    # R: output_list=list("dem.adj"=dem2, "processed"=marked.matrix, "summary"=riversmooth.summary)
    output_list = {
        "dem.adj": dem2,
        "processed": marked_matrix,
        "summary": riversmooth_summary,
    }
    return output_list
