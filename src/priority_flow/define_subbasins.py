"""
Define subbasins and stream segments from flow direction and drainage area.

Line-by-line translation of Define_Subbasins.R (CalcSubbasins) from the R
PriorityFlow package. R uses 1-based indexing; we use 0-based for array
indexing; summary coordinate columns are 0-based.
"""

import numpy as np
from typing import Dict, Optional, Tuple

####################################################################
# PriorityFlow - Topographic Processing Toolkit for Hydrologic Models
# Copyright (C) 2018  Laura Condon (lecondon@email.arizona.edu)
# Contributors - Reed Maxwell (rmaxwell@mines.edu)
####################################################################


def calc_subbasins(
    direction: np.ndarray,
    area: np.ndarray,
    mask: Optional[np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
    riv_th: int = 50,
    printflag: bool = False,
    merge_th: int = 0,
) -> Dict[str, np.ndarray]:
    # R: nx=nrow(direction)
    # R: ny=ncol(direction)
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
    # R: border=border*mask
    border = border * mask
    # R: border[which(border<4 & border!=0)]=1
    border[(border < 4) & (border != 0)] = 1
    # R: border[border==4]=0
    border[border == 4] = 0

    # initilize drinage area matrix
    # R: subbasin=matrix(0, nrow=nx, ncol=ny)
    subbasin = np.zeros((nx, ny))
    # R: marked=matrix(0, nrow=nx, ncol=ny)
    marked = np.zeros((nx, ny))

    # D4 neighbors
    # R: kd=matrix(0, nrow=4, ncol=2) #ordered down, left top right
    kd = np.zeros((4, 2))
    # R: kd[,1]=c(0,-1,0,1)
    kd[:, 0] = [0, -1, 0, 1]
    # R: kd[,2]=c(-1,0,1,0)
    kd[:, 1] = [-1, 0, 1, 0]

    # make a river mask based on the drainage area threshold
    # R: rivers=area
    rivers = area.copy()
    # R: rivers[area<riv_th]=0
    rivers[area < riv_th] = 0
    # R: rivers[area>=riv_th]=1
    rivers[area >= riv_th] = 1
    # R: end=F
    end = False
    # R: if(sum(rivers)==0) { print(...); end=T }
    if np.sum(rivers) == 0:
        print(
            "Area Threshold too high.  No river cells found. Please select a lower riv_th value"
        )
        end = True

    if end == False:
        # make masks of which cells drain down, up, left right
        # R: down=up=left=right=matrix(0, nrow=nx, ncol=ny)
        down = np.zeros((nx, ny))
        up = np.zeros((nx, ny))
        left = np.zeros((nx, ny))
        right = np.zeros((nx, ny))
        # R: down[which(direction==d4[1])]=1
        down[direction == d4[0]] = 1
        # R: left[which(direction==d4[2])]=1
        left[direction == d4[1]] = 1
        # R: up[which(direction==d4[3])]=1
        up[direction == d4[2]] = 1
        # R: right[which(direction==d4[4])]=1
        right[direction == d4[3]] = 1

        # calculate the number of river cells draining to any cell
        # R: draincount=matrix(0, nrow=nx, ncol=ny)
        draincount = np.zeros((nx, ny))
        # R: draincount[,1:(ny-1)]=draincount[,1:(ny-1)]+down[,2:ny]*rivers[,2:ny]
        draincount[:, 0 : (ny - 1)] = (
            draincount[:, 0 : (ny - 1)] + down[:, 1:ny] * rivers[:, 1:ny]
        )
        # R: draincount[,2:ny]=draincount[,2:ny]+up[,1:(ny-1)]*rivers[,1:(ny-1)]
        draincount[:, 1:ny] = (
            draincount[:, 1:ny] + up[:, 0 : (ny - 1)] * rivers[:, 0 : (ny - 1)]
        )
        # R: draincount[1:(nx-1),]=draincount[1:(nx-1),]+left[2:nx,]*rivers[2:nx,]
        draincount[0 : (nx - 1), :] = (
            draincount[0 : (nx - 1), :] + left[1:nx, :] * rivers[1:nx, :]
        )
        # R: draincount[2:nx, ]=draincount[2:nx,]+right[1:(nx-1),]*rivers[1:(nx-1),]
        draincount[1:nx, :] = (
            draincount[1:nx, :] + right[0 : (nx - 1), :] * rivers[0 : (nx - 1), :]
        )

        # Identify all the headwater cells
        # R: headwater=matrix(0, nrow=nx, ncol=ny)
        headwater = np.zeros((nx, ny))
        # R: headwater[which(draincount==0 & rivers==1)]=1
        headwater[(draincount == 0) & (rivers == 1)] = 1

        # give values outside the mask and on the border a negative count so they aren't processed
        # R: marked[which(mask==0)]=1
        marked[mask == 0] = 1

        # start with all the headwater cells (i.e. cells with zero upstream neigbors)
        # R: blist=cbind(which(headwater==1), which(headwater==1, arr.ind=T))
        # R returns indices in column-major order; argwhere is row-major, so sort to match R
        blist = np.argwhere(headwater == 1)
        blist = blist[np.lexsort((blist[:, 0], blist[:, 1]))]  # sort by col then row (column-major)
        # R: nheadwater=nrow(blist)
        nheadwater = blist.shape[0]

        # R: ends=rivers
        ends = rivers.copy()
        # R: ends[blist[,1]]=2  -> set headwater cells to 2
        ends[headwater == 1] = 2

        # Get just the river areas to use for this
        # R: rivarea=area*rivers
        rivarea = area * rivers

        # R: index=0
        index = 0

        # R: subbasin=matrix(0, nrow=nx, ncol=ny)
        subbasin = np.zeros((nx, ny))
        # R: marked=matrix(0, nrow=nx, ncol=ny)
        marked = np.zeros((nx, ny))
        # R: first=T
        first = True

        ###1. walk down from every headwater marking stream segments
        # R: for(i in 1:nheadwater){
        for i in range(nheadwater):
            # R: xtemp=blist[i,2]  ytemp=blist[i,3]  (R 1-based row,col)
            xtemp = int(blist[i, 0])
            ytemp = int(blist[i, 1])
            # R: active=T
            active = True
            # R: index=index+1
            index = index + 1
            # R: subbasin[xtemp,ytemp]=index
            subbasin[xtemp, ytemp] = index
            # R: marked[xtemp,ytemp]=1
            marked[xtemp, ytemp] = 1
            # R: summarytemp=c(index, xtemp, ytemp, rep(0,4))
            summarytemp = [index, xtemp, ytemp, 0, 0, 0, 0]

            # R: while(active==T){
            while active:
                # get the direction and find downstream cell
                # R: dirtemp=which(d4==direction[xtemp,ytemp])
                dirtemp = int(np.where(np.array(d4) == direction[xtemp, ytemp])[0][0])
                # R: xds=xtemp+kd[dirtemp,1]
                xds = xtemp + int(kd[dirtemp, 0])
                # R: yds=ytemp+kd[dirtemp,2]
                yds = ytemp + int(kd[dirtemp, 1])

                # if the downstream neigbor hasn't already been procesed and its in the domain
                # R: if(xds*yds>0 & xds<=nx & yds<=ny){
                if xds >= 0 and yds >= 0 and xds < nx and yds < ny:
                    # R: if(marked[xds,yds]==0 & mask[xds,yds]==1){
                    if marked[xds, yds] == 0 and mask[xds, yds] == 1:
                        # Check the area difference
                        # R: accum=area[xds,yds]-area[xtemp, ytemp]
                        accum = area[xds, yds] - area[xtemp, ytemp]

                        # if there is a tributary coming in then start a new segment
                        # R: if(accum>riv_th){
                        if accum > riv_th:
                            # R: summarytemp[4:5]=c(xtemp,ytemp)
                            summarytemp[3] = xtemp
                            summarytemp[4] = ytemp
                            # R: summarytemp[6]=index+1
                            summarytemp[5] = index + 1
                            # R: index=index+1
                            index = index + 1
                            # R: ends[xtemp,ytemp]=3
                            ends[xtemp, ytemp] = 3
                            # R: ends[xds,yds]=2
                            ends[xds, yds] = 2
                            # R: if(first==T){ summary=summarytemp; first=F }else{ summary=rbind(summary, summarytemp) }
                            if first:
                                summary = np.array([summarytemp], dtype=np.float64)
                                first = False
                            else:
                                summary = np.vstack([summary, summarytemp])
                            # R: summarytemp=c(index, xds, yds, rep(0,4))
                            summarytemp = [index, xds, yds, 0, 0, 0, 0]

                        # assign subbasin number to the downstream cell and mark it off
                        # R: subbasin[xds,yds]=index
                        subbasin[xds, yds] = index
                        # R: marked[xds,yds]=1
                        marked[xds, yds] = 1
                        # R: xtemp=xds  ytemp=yds
                        xtemp = xds
                        ytemp = yds
                    else:
                        # if the downstream neighbor has been processed then move on to the next headwater cell
                        # R: active=FALSE
                        active = False
                        # R: ends[xtemp,ytemp]=3
                        ends[xtemp, ytemp] = 3
                        # R: summarytemp[4:5]=c(xtemp,ytemp)
                        summarytemp[3] = xtemp
                        summarytemp[4] = ytemp
                        # R: summarytemp[6]=subbasin[xds,yds]
                        summarytemp[5] = subbasin[xds, yds]
                else:
                    # R: active=FALSE  ends[xtemp,ytemp]=3  summarytemp[4:5]=c(xtemp,ytemp)  summarytemp[6]=-1
                    active = False
                    ends[xtemp, ytemp] = 3
                    summarytemp[3] = xtemp
                    summarytemp[4] = ytemp
                    summarytemp[5] = -1

            # R: if(first==T){ summary=matrix(summarytemp, ncol=7, byrow=T); first=F }else{ summary=rbind(summary, summarytemp) }
            if first:
                summary = np.array([summarytemp], dtype=np.float64)
                first = False
            else:
                summary = np.vstack([summary, summarytemp])

        # R: rownames(summary)=NULL  colnames(summary)=c(...)  (metadata only)

        ###2. Get the drainage basins for every segement
        # R: subbasinA=subbasin
        subbasinA = subbasin.copy()

        # start a queue with all the cells in the river
        # R: queue=which(subbasin>0, arr.ind=T)
        queue = np.argwhere(subbasin > 0)
        # R: nqueue=nrow(queue)
        nqueue = queue.shape[0]
        # R: ii=1
        ii = 1
        # R: while(nqueue>0){
        while nqueue > 0:
            # R: if(printflag){print(paste("lap", ii, "ncell", nqueue))}
            if printflag:
                print(f"lap {ii} ncell {nqueue}")
            # R: queue2=NULL
            queue2 = []

            # loop through the queue
            # R: for(i in 1:nqueue){
            for i in range(nqueue):
                # R: xtemp=queue[i,1]  ytemp=queue[i,2]
                xtemp = int(queue[i, 0])
                ytemp = int(queue[i, 1])
                # add one to the subbasin area for the summary
                # R: sbtemp=subbasinA[xtemp,ytemp]
                sbtemp = int(subbasinA[xtemp, ytemp])
                # R: summary[sbtemp,7]=summary[sbtemp,7]+1  (R: row index = basin ID)
                row_idx = np.where(summary[:, 0] == sbtemp)[0][0]
                summary[row_idx, 6] = summary[row_idx, 6] + 1

                # look for cells that drain to this cell
                # R: for(d in 1:4){
                for d in range(4):
                    # R: xus=xtemp-kd[d,1]  yus=ytemp-kd[d,2]
                    xus = xtemp - int(kd[d, 0])
                    yus = ytemp - int(kd[d, 1])
                    # R: if(xus*yus>0 & xus<=nx & yus<=ny){
                    if xus >= 0 and yus >= 0 and xus < nx and yus < ny:
                        # R: if(mask[xus,yus]==1 & subbasinA[xus,yus]==0){
                        if mask[xus, yus] == 1 and subbasinA[xus, yus] == 0:
                            # R: if(direction[xus,yus]==d4[d]){
                            if direction[xus, yus] == d4[d]:
                                # R: subbasinA[xus,yus]=subbasinA[xtemp,ytemp]
                                subbasinA[xus, yus] = subbasinA[xtemp, ytemp]
                                # R: queue2=rbind(queue2, c(xus,yus))
                                queue2.append([xus, yus])

            # R: if(length(queue2)>=2){ queue=queue2; nqueue=nrow(queue); ii=ii+1 } else{nqueue=0}
            # In R, length(queue2) is rows * cols; a single-row (1x2) matrix has length 2.
            # The condition length(queue2) >= 2 therefore means "at least one row".
            # In Python, len(queue2) is the number of rows, so we must check >= 1.
            if len(queue2) >= 1:
                queue = np.array(queue2)
                nqueue = queue.shape[0]
                ii = ii + 1
            else:
                nqueue = 0

        ###3. if merge_th >0 look for basins with areas less than the merge threshold...
        # R: delete=NULL
        delete = []
        # R: if(merge_th>0){
        if merge_th > 0:
            print(
                "WARNING: non-zero merge thresholds are not compatible with the RiverSmooth function"
            )
            # R: nsb=nrow(summary)
            nsb = summary.shape[0]
            # R: for(i in 1:nsb){
            for i in range(nsb):
                # check if area is less than the threshold & it does not drain externally
                # R: if(summary[i,7]<merge_th & summary[i,6]>0){
                if summary[i, 6] < merge_th and summary[i, 5] > 0:
                    # R: delete=c(delete,i)
                    delete.append(i)
                    # R: bas1=summary[i,1]  bas2=summary[i,6]
                    bas1 = int(summary[i, 0])
                    bas2 = int(summary[i, 5])

                    # replace numbers in the subbasin matrix
                    # R: ilist=which(subbasin==bas1)  subbasin[ilist]=bas2
                    subbasin[subbasin == bas1] = bas2

                    # replace numbers in the subbasin area  matrix
                    # R: ilistA=which(subbasinA==bas1)  subbasinA[ilistA]=bas2
                    subbasinA[subbasinA == bas1] = bas2

                    # adjust the summary matrix for the downstream basin
                    # R: summary[which(summary[,1]==bas2),7]=summary[which(summary[,1]==bas2),7] + summary[i,7]
                    ds_rows = np.where(summary[:, 0] == bas2)[0]
                    summary[ds_rows[0], 6] = summary[ds_rows[0], 6] + summary[i, 6]
                    # R: summary[which(summary[,1]==bas2),2]=summary[i,2]  summary[...,3]=summary[i,3]
                    summary[ds_rows[0], 1] = summary[i, 1]
                    summary[ds_rows[0], 2] = summary[i, 2]

                    # Change the downstream basin number for any upstream basins to downstream basin
                    # R: uplist=which(summary[,6]==bas1)  summary[uplist,6]=bas2
                    uplist = np.where(summary[:, 5] == bas1)[0]
                    summary[uplist, 5] = bas2

            # R: if(is.null(delete)==F){ summary=summary[-delete,] }
            if len(delete) > 0:
                summary = np.delete(summary, delete, axis=0)

    # R: output_list=list("segments"=subbasin, "subbasins"=subbasinA, "RiverMask"=rivers, "summary"=summary)
    # When end==True we never set subbasin, subbasinA, summary; return empty/default
    if end:
        subbasin = np.zeros((nx, ny))
        subbasinA = np.zeros((nx, ny))
        summary = np.zeros((0, 7))

    output_list = {
        "segments": subbasin,
        "subbasins": subbasinA,
        "RiverMask": rivers,
        "summary": summary,
    }
    # R: return(output_list)
    return output_list
