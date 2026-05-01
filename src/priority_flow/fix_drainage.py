"""
Walk upstream from a point ensuring DEM is increasing by a minimum epsilon.

Line-by-line translation of Fix_Drainage.R (FixDrainage) from the R PriorityFlow
package. R uses 1-based indexing; we use 0-based. startpoint is (row, col) 0-based.
"""

import numpy as np
from typing import Dict, Tuple, Union

####################################################################
# PriorityFlow - Topographic Processing Toolkit for Hydrologic Models
# Copyright (C) 2018  Laura Condon (lecondon@email.arizona.edu)
# Contributors - Reed Maxwell (rmaxwell@mines.edu)
####################################################################


def fix_drainage(
    dem: np.ndarray,
    direction: np.ndarray,
    mask: np.ndarray,
    bank_epsilon: float,
    startpoint: Union[Tuple[int, int], list, np.ndarray],
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
) -> Dict[str, np.ndarray]:
    """
    Walk upstream from a point ensuring DEM is increasing by a minum epsilon.

    This walks upstream from a point following a flow direction file and checks that the elevation upstream
    is greater than or equal to the elvation at the point + epsilon
    Once the function reaches a point where upstream cells pass it stops 
    
    NOTE: this is only processing the immediate neigborhood and does not recurse over the entire domain, 
    For example if you did the entire overal prioirty flow processing with an espilon of 0 then ran this
    functon starting at a river point with an epsilon of 0.1 this function will traverse upstream from the 
    river bottom checking that every cell conneting to the river cell is higher by at least this ammout but 
    once it reaches a point where every connected cell passes this test it will stop. Therefore there could
    still be locations higher up on the hillslope with the original epsilon of zero still.  This  is on purpose
    and this script is not intended to gobally ensure a given epsilon. 

    Parameters
    ----------
    dem : np.ndarray
        2D array of elevations for the domain (nx, ny).
    direction : np.ndarray
        2D array of D4 flow directions for each cell. Direction codes
        follow the ``d4`` numbering scheme.
    mask : np.ndarray
        2D mask with ones for cells that are allowed to be processed and
        zeros elsewhere. Only neighbors within the mask can be adjusted.
    bank_epsilon : float
        Minimum elevation difference that upstream cells must have
        relative to the current cell. If an upstream neighbor is lower
        than ``current + bank_epsilon``, its elevation is raised to
        ``current + bank_epsilon``.
    startpoint : tuple[int, int] or list or np.ndarray
        Starting grid cell given as ``(row, col)`` indices (0-based)
        from which to begin walking upstream.
    d4 : tuple of int, optional
        Direction numbering system for the D4 neighbors, given as
        ``(down, left, up, right)``. Defaults to ``(1, 2, 3, 4)`` to
        match the original R implementation.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:

        - ``\"dem.adj\"``: 2D elevation array after local adjustments.
        - ``\"processed\"``: 2D array marking cells that were adjusted
          (1) or visited during processing, and 0 elsewhere.
    """
    # D4 neighbors - Rows: down, left top right
    # Columns: (1)deltax, (2)deltay, direction number if you are walking upstream
    # R: ku=matrix(0, nrow=4, ncol=3)
    ku = np.zeros((4, 3))
    # R: ku[,1]=c(0,1,0,-1)
    ku[:, 0] = [0, 1, 0, -1]
    # R: ku[,2]=c(1,0,-1,0)
    ku[:, 1] = [1, 0, -1, 0]
    # R: ku[,3]=c(1, 2, 3, 4)
    ku[:, 2] = [1, 2, 3, 4]

    # renumber the directions to 1=down, 2=left, 3=up, 4=right if a different numbering scheme was used
    # R: dir2=direction
    dir2 = direction.copy()
    # R: nx=dim(direction)[1]  ny=dim(direction)[2]
    nx = direction.shape[0]
    ny = direction.shape[1]
    # R: if(d4[1]!=1){dir2[which(direction==d4[1])]=1}  etc.
    if d4[0] != 1:
        dir2[direction == d4[0]] = 1
    if d4[1] != 2:
        dir2[direction == d4[1]] = 2
    if d4[2] != 3:
        dir2[direction == d4[2]] = 3
    if d4[3] != 4:
        dir2[direction == d4[3]] = 4

    # initializing
    # R: marked=matrix(0, nrow=nx, ncol=ny)
    marked = np.zeros((nx, ny))
    # R: queue=cbind(startpoint[1],startpoint[2])
    queue = np.array([[startpoint[0], startpoint[1]]])
    # R: active=TRUE
    active = True
    # R: dem2=dem
    dem2 = dem.copy()

    # R: while(active==T){
    while active:
        # R: indx=queue[1,1]  indy=queue[1,2]
        indx = int(queue[0, 0])
        indy = int(queue[0, 1])
        # R: queuetemp=NULL
        queuetemp = None

        # Loop over four directions check for non-stream neighbors pointing to this cell
        # R: for(d in 1:4){
        for d in range(4):
            # R: tempx=indx+ku[d,1]  tempy=indy+ku[d,2]
            tempx = indx + int(ku[d, 0])
            tempy = indy + int(ku[d, 1])
            # if it points to the cell, is within the mask of cells to be processed, and has epsilon < the threshold
            # R: if(tempx*tempy>0 & tempx<nx & tempy<ny){
            # R 1-based: valid 1..nx, 1..ny; R has tempx<nx so 1..nx-1 - translating to 0-based: 0..nx-1, 0..ny-1
            if tempx >= 0 and tempy >= 0 and tempx < nx and tempy < ny:
                # R: if((d-dir2[tempx,tempy])==0 & mask[tempx,tempy]==1){
                # R d is 1..4; in Python d is 0..3 so direction at neighbor should be d+1
                if (d + 1 - dir2[tempx, tempy]) == 0 and mask[tempx, tempy] == 1:
                    # R: if((dem2[tempx,tempy]-dem2[indx,indy])<bank.epsilon){
                    if (dem2[tempx, tempy] - dem2[indx, indy]) < bank_epsilon:
                        # R: dem2[tempx,tempy]=dem2[indx,indy]+bank.epsilon
                        dem2[tempx, tempy] = dem2[indx, indy] + bank_epsilon
                        # R: marked[tempx,tempy]=1
                        marked[tempx, tempy] = 1
                        # R: queuetemp=rbind(c(tempx,tempy),queuetemp)
                        if queuetemp is None:
                            queuetemp = np.array([[tempx, tempy]])
                        else:
                            queuetemp = np.vstack(
                                [np.array([[tempx, tempy]]), queuetemp]
                            )

        # if cells were adjusted then add to the top of the queue replacing the cell that was just done
        # R: if(length(queuetemp>0)){  (typo in R: should be length(queuetemp)>0)
        if queuetemp is not None and queuetemp.size > 0:
            # R: queue=rbind(queuetemp,queue[-1,])
            queue = np.vstack([queuetemp, queue[1:]])
        else:
            # if no cells were adjusted remove this cell from the queue and if its the last one you are done
            # R: if(nrow(queue)>1){
            if queue.shape[0] > 1:
                # R: queue=queue[-1,]
                queue = queue[1:]
                # R: if(length(queue)==2){ queue=matrix(queue, ncol=2, byrow=T) }
                if queue.size == 2:
                    queue = queue.reshape(1, 2)
            else:
                # R: active=F
                active = False

    # R: output_list=list("dem.adj"=dem2, "processed"=marked)
    output_list = {"dem.adj": dem2, "processed": marked}
    return output_list
