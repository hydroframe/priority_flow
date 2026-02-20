"""
Calculated drainage area.

Line-by-line translation of drainage_area.R (drainageArea) from the R PriorityFlow package.
Calculates the number of cells draining to any cell given a flow direction file.
R uses 1-based indexing; we use 0-based.
"""

import numpy as np
from typing import Optional, Tuple


def drainage_area(
    direction: np.ndarray,
    mask: Optional[np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    printflag: bool = False,
) -> np.ndarray:
    """
    Calculate drainage area (number of cells draining to any cell) given a flow direction file.
    Translation of drainageArea() from R PriorityFlow.
    """
    # R: nx=nrow(direction)  ny=ncol(direction)
    nx = direction.shape[0]
    ny = direction.shape[1]

    # R: if(missing(mask)){mask=matrix(1, nrow=nx, ncol=ny)}
    if mask is None:
        mask = np.ones((nx, ny))

    # Setup the border
    # R: border=matrix(1, nrow=nx, ncol=ny)
    border = np.ones((nx, ny))
    # R: border[2:(nx-1), 2:(ny-1)]= mask[1:(nx-2), 2:(ny-1)] + mask[3:nx, 2:(ny-1)] + ...
    border[1 : (nx - 1), 1 : (ny - 1)] = (
        mask[0 : (nx - 2), 1 : (ny - 1)]
        + mask[2:nx, 1 : (ny - 1)]
        + mask[1 : (nx - 1), 0 : (ny - 2)]
        + mask[1 : (nx - 1), 2:ny]
    )
    # R: border=border*mask  border[which(border<4 & border!=0)]=1  border[border==4]=0
    border = border * mask
    border[(border < 4) & (border != 0)] = 1
    border[border == 4] = 0

    # initialize drainage area matrix
    # R: drainarea=matrix(1, nrow=nx, ncol=ny)
    drainarea = np.ones((nx, ny))

    # D4 neighbors  # R: kd=matrix(0, nrow=4, ncol=2)  ordered down, left top right
    kd = np.zeros((4, 2))
    kd[:, 0] = [0, -1, 0, 1]
    kd[:, 1] = [-1, 0, 1, 0]

    # make masks of which cells drain down, up, left right
    # R: down=up=left=right=matrix(0, nrow=nx, ncol=ny)
    down = np.zeros((nx, ny))
    up = np.zeros((nx, ny))
    left = np.zeros((nx, ny))
    right = np.zeros((nx, ny))
    # R: down[which(direction==d4[1])]=1  etc.
    down[direction == d4[0]] = 1
    left[direction == d4[1]] = 1
    up[direction == d4[2]] = 1
    right[direction == d4[3]] = 1

    # calculate the number of cells draining to any cell
    # R: draincount=matrix(0, nrow=nx, ncol=ny)
    draincount = np.zeros((nx, ny))
    # R: draincount[,1:(ny-1)]=draincount[,1:(ny-1)]+down[,2:ny]
    draincount[:, 0 : (ny - 1)] = draincount[:, 0 : (ny - 1)] + down[:, 1:ny]
    # R: draincount[,2:ny]=draincount[,2:ny]+up[,1:(ny-1)]
    draincount[:, 1:ny] = draincount[:, 1:ny] + up[:, 0 : (ny - 1)]
    # R: draincount[1:(nx-1),]=draincount[1:(nx-1),]+left[2:nx,]
    draincount[0 : (nx - 1), :] = draincount[0 : (nx - 1), :] + left[1:nx, :]
    # R: draincount[2:nx, ]=draincount[2:nx,]+right[1:(nx-1),]
    draincount[1:nx, :] = draincount[1:nx, :] + right[0 : (nx - 1), :]

    # give values outside the mask and on the border a negative count so they aren't processed
    # R: draincount[which(mask==0)]=(-99)
    draincount[mask == 0] = -99

    # initialize a queue with all the headwater cells (i.e. cells with zero upstream neighbors)
    # R: draintemp=draincount
    draintemp = draincount.copy()
    # R: queue=which(draintemp==0, arr.ind=T)  -> (row, col) for each
    queue_rows, queue_cols = np.where(draintemp == 0)
    queue = np.column_stack((queue_rows, queue_cols))
    # R: qlist=which(draintemp==0)  (linear indices - not used later in R, we can skip or keep)
    # R: blist=cbind(which(draintemp>0), which(draintemp>0, arr.ind=T))  -> cols: linear_idx, row, col
    # In R, which(draintemp>0) is column-major linear index. We need same for consistent indexing.
    # blist is used as: draintemp[blist[,1]] in R -> index by linear; and queue=blist[ilist,2:3].
    # So we need (linear_index, row, col). R linear index: column-major. For indexing we use
    # ilist=which(draintemp[blist[,1]]==0) - so we need to index draintemp by linear index.
    # In Python we can use (row,col) to index: draintemp[blist[:,1], blist[:,2]]. So blist cols: (dummy, row, col) or (row, col) only. But then queue=blist[ilist,2:3] means we need 3 cols. So blist has (linear, row, col). For draintemp[blist[ilist,1], blist[ilist,2]] we get values. So we don't need linear in the check - we need row,col. So blist = (row, col) for draintemp>0, 2 cols. Then ilist = where(draintemp[blist[:,0], blist[:,1]]==0). queue = blist[ilist]. But R has 3 cols so that length(blist)/3 is nrow. So R blist is (linear, row, col). So queue = blist[ilist, 2:3]. So we need 3 columns. Build: rows, cols = where(draintemp>0); linear = rows + cols*nx for column-major? No - R column-major: (i,j) -> (j-1)*nx + i in 1-based, so 0-based (i,j) -> j*nx + i. So linear = rows + cols * nx. Then blist = column_stack((linear, rows, cols)).
    linear_idx = np.where(draintemp.ravel(order="F") > 0)[0]
    rows_b = linear_idx % nx
    cols_b = linear_idx // nx
    blist = np.column_stack((linear_idx, rows_b, cols_b))
    # R: nqueue=nrow(queue)
    nqueue = queue.shape[0]

    ii = 1

    # R: while(nqueue>0){
    while nqueue > 0:
        if printflag:
            print("lap", ii, "ncell", nqueue)

        # loop through the queue
        # R: for(i in 1:nqueue){
        for i in range(nqueue):
            # look downstream add 1 to the area and subtract 1 from the drainage #
            # R: xtemp=queue[i,1]  ytemp=queue[i,2]
            xtemp = int(queue[i, 0])
            ytemp = int(queue[i, 1])

            # if it has a flow direction
            # R: if(is.na(direction[xtemp,ytemp])==F){
            if not np.isnan(direction[xtemp, ytemp]):
                # R: dirtemp=which(d4==direction[xtemp,ytemp])
                dirtemp = int(np.where(np.array(d4) == direction[xtemp, ytemp])[0][0])
                # R: xds=xtemp+kd[dirtemp,1]  yds=ytemp+kd[dirtemp,2]
                xds = int(xtemp + kd[dirtemp, 0])
                yds = int(ytemp + kd[dirtemp, 1])

                # add one to the area of the downstream cell as long as that cell is in the domain
                # R: if(xds<=nx & xds>=1 & yds<=ny & yds>=1){
                if 0 <= xds < nx and 0 <= yds < ny:
                    # R: drainarea[xds, yds]=drainarea[xds, yds]+drainarea[xtemp,ytemp]
                    drainarea[xds, yds] = (
                        drainarea[xds, yds] + drainarea[xtemp, ytemp]
                    )
                    # subtract one from the number of upstream cells from the downstream cell
                    # R: draintemp[xds,yds]= draintemp[xds,yds] - 1
                    draintemp[xds, yds] = draintemp[xds, yds] - 1
                # R: } #end if in the domain extent
            # R: } #end if not na

            # set the drain temp to -99 for current cell to indicate its been done
            # R: draintemp[xtemp,ytemp]=-99
            draintemp[xtemp, ytemp] = -99
        # R: } #end for i in 1:nqueue

        # make a new queue with the cells with zero upstream drains left
        # R: ilist=which(draintemp[blist[,1]]==0)
        # In R blist[,1] is column-major linear index. So draintemp[blist[,1]] are values at those cells.
        # In Python: values at (blist[:,1], blist[:,2])
        if blist.shape[0] > 0:
            vals = draintemp[blist[:, 1].astype(int), blist[:, 2].astype(int)]
            ilist = np.where(vals == 0)[0]
        else:
            ilist = np.array([], dtype=int)
        # R: queue=blist[ilist,2:3]
        if ilist.size > 0:
            queue = blist[ilist, 1:3].copy()
        else:
            queue = np.empty((0, 2))
        # R: if(length(ilist)!=length(blist)/3){ blist=blist[-ilist,] } else{ blist=NULL }
        if ilist.size != blist.shape[0]:
            blist = np.delete(blist, ilist, axis=0)
        else:
            blist = np.empty((0, 3))
            if printflag:
                print("blist empty")

        # R: nqueue=length(queue)/2
        nqueue = queue.shape[0]

        # R: if(nqueue==1){queue=matrix(queue, ncol=2, nrow=1)}
        if nqueue == 1:
            queue = queue.reshape(1, 2)
        # R: if(length(blist)/3==1){blist=matrix(blist, ncol=3, nrow=1)}
        if blist.shape[0] == 1:
            blist = blist.reshape(1, 3)

        ii = ii + 1
    # R: }

    # R: drainarea=drainarea*mask  return(drainarea)
    drainarea = drainarea * mask
    return drainarea