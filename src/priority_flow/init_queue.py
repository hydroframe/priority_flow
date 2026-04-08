"""
Initialize queue for topographic processing.

Line-by-line translation of Init_Queue.R from the R PriorityFlow package.
R uses 1-based indexing and column-major (Fortran) order; we use 0-based
and NumPy Fortran order so queue and behaviour match R.
"""

import numpy as np
from typing import Dict, Optional, Tuple

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


def init_queue(
    dem: np.ndarray,
    initmask: Optional[np.ndarray] = None,
    domainmask: Optional[np.ndarray] = None,
    border: Optional[np.ndarray] = None,
    d4: Tuple[int, int, int, int] = (1, 2, 3, 4),
) -> Dict[str, np.ndarray]:
    """
    Initialize queue for topographic processing.

    Sets up a queue and initializes marked and step matrices for DEM processing

    This function is a port of the R function ``InitQueue`` from the
    PriorityFlow package. It sets up the initial queue of cells, along
    with ``marked``, ``step``, ``basin`` and ``direction`` matrices used
    during DEM processing.

    Parameters
    ----------
    dem : np.ndarray
        2D array of elevations for the domain in HydroFrame orientation (nx by ny).
    initmask : np.ndarray, optional
        Mask with the same shape as ``dem`` indicating the subset of
        cells to be considered for the initial queue (for example, only
        river cells). If ``None``, all border cells in the domain are
        eligible to be added to the queue.
    domainmask : np.ndarray, optional
        Mask defining the domain extent to be considered. If ``None``,
        the entire rectangular extent of ``dem`` is used as the domain.
    border : np.ndarray, optional
        Optional pre-computed border mask. If ``None``, the border is
        derived from ``domainmask`` (cells on the outer edge of the
        domain and internal boundaries).
    d4 : tuple of int, optional
        Direction numbering system for the D4 neighbors, given as
        ``(down, left, up, right)``. Defaults to ``(1, 2, 3, 4)`` to
        match the original R implementation.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:

        - ``\"mask\"``: the effective initialization mask used to define
          candidate outlet cells (``initmask`` after any defaults).
        - ``\"queue\"``: 2D array of outlet cells with columns
          ``(row, col, elevation)`` in column-major order, suitable for
          use with ``d4_traverse_b``.
        - ``\"marked\"``: 2D array indicating which cells were added to
          the initial queue (1 for queue cells, 0 otherwise).
        - ``\"basins\"``: 2D array assigning a unique basin number to
          each outlet cell (queue entry).
        - ``\"direction\"``: 2D array of flow directions at outlet
          points, pointing out of the domain, using the ``d4`` numbering
          scheme.
    """
    # HydroFrame layout -> internal R-style layout (transpose)
    dem = dem.T.copy()
    if initmask is not None:
        initmask = initmask.T.copy()
    if domainmask is not None:
        domainmask = domainmask.T.copy()
    if border is not None:
        border = border.T.copy()

    # initialize queue and matrices
    # R: ny=ncol(dem)  nx=nrow(dem)
    ny = dem.shape[1]
    nx = dem.shape[0]
    queue = None
    # R: marked=matrix(0, nrow=nx, ncol=ny)
    marked = np.zeros((nx, ny))
    # R: step=matrix(0, nrow=nx, ncol=ny)
    step = np.zeros((nx, ny))

    # setup flow directions
    # D4 neighbors  # R: kd=matrix(0, nrow=4, ncol=3)  ordered down, left top right
    kd = np.zeros((4, 3))
    # R: kd[,1]=c(0,-1,0,1)
    kd[:, 0] = [0, -1, 0, 1]
    # R: kd[,2]=c(-1,0,1,0)
    kd[:, 1] = [-1, 0, 1, 0]
    # R: kd[,3]=c(d4[1], d4[2], d4[3], d4[4])
    kd[:, 2] = [d4[0], d4[1], d4[2], d4[3]]

    # R: if(missing(initmask)){
    if initmask is None:
        # R: print("No init mask provided all border cells will be added to queue")
        print("No init mask provided all border cells will be added to queue")
        # R: initmask=matrix(1, nrow=nx, ncol=ny)
        initmask = np.ones((nx, ny))
    # R: }

    # R: if(missing(domainmask)){
    if domainmask is None:
        # R: print("No domain mask provided using entire domain")
        print("No domain mask provided using entire domain")
        # R: domainmask=matrix(1, nrow=nx, ncol=ny)
        domainmask = np.ones((nx, ny))
    # R: }

    # Setup the border
    # R: if(missing(border)){
    if border is None:
        # R: print("No border provided, setting border using domain mask")
        print("No border provided, setting border using domain mask")
        # R: border=matrix(1, nrow=nx, ncol=ny)
        border = np.ones((nx, ny))
        # R: border[2:(nx-1), 2:(ny-1)]= domainmask[1:(nx-2), 2:(ny-1)] + ...
        #    1-based R: rows 2..nx-1, cols 2..ny-1  -> Python 0-based: rows 1..nx-2, cols 1..ny-2
        border[1 : (nx - 1), 1 : (ny - 1)] = (
            domainmask[0 : (nx - 2), 1 : (ny - 1)]
            + domainmask[2:nx, 1 : (ny - 1)]
            + domainmask[1 : (nx - 1), 0 : (ny - 2)]
            + domainmask[1 : (nx - 1), 2:ny]
        )
        # R: border=border*domainmask
        border = border * domainmask
        # R: border[which(border<4 & border!=0)]=1
        border[(border < 4) & (border != 0)] = 1
        # R: border[border==4]=0
        border[border == 4] = 0
    # R: }

    # R: basin=matrix(0, nrow=nx, ncol=ny)
    basin = np.zeros((nx, ny))
    # R: maskbound=initmask*border
    maskbound = initmask * border
    # R: blist=which(maskbound>0)  # array indices (1-based, column-major in R)
    # In Python: use column-major (Fortran) so order matches R
    dem_flat_f = dem.ravel(order="F")
    maskbound_flat_f = maskbound.ravel(order="F")
    blist = np.where(maskbound_flat_f > 0)[0]  # 0-based flat indices, column-major order
    # R: binlist=which(maskbound>0, arr.ind=T)  # xy indices (row,col in R)
    # column-major: flat index k -> row = k % nx, col = k // nx (0-based)
    binlist_rows = blist % nx
    binlist_cols = blist // nx
    # R: queue=cbind(binlist, dem[blist])  -> columns: row, col, elevation
    queue = np.column_stack((binlist_rows, binlist_cols, dem_flat_f[blist]))
    # R: marked[blist]=1  (linear index in column-major)
    marked[binlist_rows, binlist_cols] = 1
    # R: basin[blist]=1:length(blist)
    basin[binlist_rows, binlist_cols] = np.arange(1, len(blist) + 1)

    # assign flow direction to point out of the domain
    # R: direction=matrix(NA, nrow=nx, ncol=ny)
    direction = np.full((nx, ny), np.nan)
    # R: for(i in 1:nrow(queue)){
    for i in range(queue.shape[0]):
        # R: xtemp=queue[i,1]   (row in R, 1-based)
        xtemp = int(queue[i, 0])  # row, 0-based
        # R: ytemp=queue[i,2]   (col in R, 1-based)
        ytemp = int(queue[i, 1])  # col, 0-based
        # R: temp=rep(0,4)
        temp = np.zeros(4)
        # R: for(d in 1:4){
        for d in range(4):
            # R: xtest=xtemp+kd[d,1]
            xtest = xtemp + kd[d, 0]
            # R: ytest=ytemp+kd[d,2]
            ytest = ytemp + kd[d, 1]

            # R: if(xtest*ytest==0 | xtest>nx | ytest>ny){
            # In R valid is 1..nx, 1..ny. So outside: xtest<1 or ytest<1 or xtest>nx or ytest>ny
            # Python 0-based: outside xtest<0 or ytest<0 or xtest>=nx or ytest>=ny
            if xtest < 0 or ytest < 0 or xtest >= nx or ytest >= ny:
                # R: temp[d]=0
                temp[d] = 0
            else:
                # R: temp[d]=domainmask[xtest, ytest]
                temp[d] = domainmask[int(xtest), int(ytest)]
        # R: dtemp=which.min(temp)  (1-based index)
        dtemp = np.argmin(temp)  # 0-based index
        # R: direction[xtemp,ytemp]=kd[dtemp,3]
        direction[xtemp, ytemp] = kd[dtemp, 2]
    # R: }

    # Internal layout -> HydroFrame layout (transpose 2D arrays; swap queue row/col)
    queue_hf = np.column_stack((queue[:, 1], queue[:, 0], queue[:, 2]))
    # R: output_list=list("mask"=initmask,"queue" = queue, "marked"=marked, "basins"=basin, "direction"=direction)
    output_list = {
        "mask": initmask.T,
        "queue": queue_hf,
        "marked": marked.T,
        "basins": basin.T,
        "direction": direction.T,
    }
    # R: return(output_list)
    return output_list
