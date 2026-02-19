"""
Calculate Stream Order functions for PriorityFlow.

This module provides functions for calculating Strahler stream orders using
stream segments delineated using the CalcSubbasins function.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


def calc_stream_order(
    basin_id: np.ndarray,
    ds_id: np.ndarray,
    segments: Optional[np.ndarray] = None
) -> Dict[str, Union[np.ndarray, np.ndarray]]:
    """
    Calculate Strahler stream orders using stream segments delineated using the CalcSubbasins function.
    
    Parameters
    ----------
    basin_id : np.ndarray
        An array of basin IDs (can be obtained from the CalcSubbasins function 
        Summary output column 1 "Basin_ID")
    ds_id : np.ndarray
        Downstream basin IDs, the downstream ID of each basin, should be corresponding 
        to the basin ID (can be obtained from the CalcSubbasins function Summary 
        output Column 6 "Downstream_Basin_ID")
    segments : np.ndarray, optional
        A channel mask with segments assigned with basin IDs 
        (can use the CalcSubbasins 'segments' output for this)
    
    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary containing:
        - 'summary': A summary table with a row for every basin with three columns 
          "Basin_ID", "Downstream_ID", "StreamOrder_number"
        - 'order_mask': (optional) Only available if segments is an input, this will 
          output a spatial raster with the channel orders
    
    Notes
    -----
    This function implements the Strahler stream ordering system where:
    - First-order streams are those without any tributaries
    - When two streams of the same order join, the resulting stream is of the next higher order
    - When streams of different orders join, the resulting stream maintains the higher order
    """
    # Initialize order numbers array
    order_no = np.zeros((len(basin_id), 1))
    
    # Find 1st order streams - streams without any basin draining into this basin
    ds_all = np.unique(ds_id)
    headwater = basin_id[~np.isin(basin_id, ds_all)]
    
    for i in range(len(headwater)):
        blist = np.where(basin_id == headwater[i])[0]
        order_no[blist] = 1
    
    # Process each headwater basin to calculate stream orders
    for i in range(len(headwater)):
        active = True
        btemp = headwater[i]
        
        while active:
            # Find downstream basin
            blist = np.where(basin_id == btemp)[0]
            dstemp = ds_id[blist][0]  # Get the downstream basin ID
            dlist = np.where(basin_id == dstemp)[0]
            
            if dstemp == 0:  # Stop when the basin drains outside the domain
                active = False
            
            # Find all basins draining to this downstream basin
            ulist = np.where(ds_id == dstemp)[0]
            
            if len(ulist) != 1:  # If more than one basin
                urest = ulist[~np.isin(ulist, blist)]  # Remove the basin in process
                ordertemp = order_no[urest].flatten()
                ordertemp[np.isnan(ordertemp)] = 0
                
                if np.prod(ordertemp) != 0:  # Check if there is any upstream not been processed
                    umax = np.max(ordertemp)
                    
                    if umax == order_no[blist][0]:  # Compare the orders of all tributaries
                        order_no[dlist] = umax + 1
                    else:
                        order_no[dlist] = max(umax, order_no[blist][0])
                    
                    btemp = dstemp
                else:
                    active = False
            else:  # If only one is draining, assign the same order
                order_no[dlist] = order_no[blist]
                btemp = dstemp
    
    # Create summary table
    summary = np.column_stack([basin_id, ds_id, order_no])
    # TODO NOTE: this should be a dataframe. Check other places where it is called.
    # Prepare output
    if segments is None:
        output_dict = {"summary": summary}
        # TODO NOTE: or perhaps the arrays should go to the dict with keys?
    else:
        segments2 = segments.copy()
        for i in range(len(basin_id)):
            btemp = basin_id[i]
            blist2 = np.where(segments2 == btemp)
            segments[blist2] = order_no[i, 0]  # Scalar value for order
        output_dict = {"summary": summary, "order_mask": segments}
    
    return output_dict 