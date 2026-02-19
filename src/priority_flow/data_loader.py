"""
Data loading functions for PriorityFlow package.

This module provides functions to load the test data files that were converted
from the R package's TestDomain_Inputs directory.
"""

import numpy as np
import os
from pathlib import Path

# Get the directory where this module is located
_MODULE_DIR = Path(__file__).parent
_DATA_DIR = _MODULE_DIR / "data"

def load_dem():
    """
    Load the Digital Elevation Model (DEM) test data.
    
    This is a small elevation dataset (215km by 172km at 1km spatial resolution)
    converted from the R package's TestDomain_Inputs.
    
    Returns
    -------
    numpy.ndarray
        A 2D array of elevation values with shape (215, 172) representing
        the domain dimensions (nrow=215, ncol=172).
        
    Examples
    --------
    >>> import priority_flow.data_loader as dl
    >>> dem = dl.load_dem()
    >>> print(f"DEM shape: {dem.shape}")
    >>> print(f"Elevation range: {dem.min():.2f} to {dem.max():.2f}")
    """
    data_path = _DATA_DIR / "DEM.npy"
    if not data_path.exists():
        raise FileNotFoundError(f"DEM data file not found at {data_path}")
    
    return np.load(data_path)

def load_watershed_mask():
    """
    Load the watershed mask test data.
    
    A mask showing the watershed drainage area for the test domain.
    
    Returns
    -------
    numpy.ndarray
        A 2D array of 0's and 1's showing the watershed extent 
        (1=inside the watershed, 0=outside the watershed) with shape (215, 172).
        
    Examples
    --------
    >>> import priority_flow.data_loader as dl
    >>> mask = dl.load_watershed_mask()
    >>> print(f"Watershed mask shape: {mask.shape}")
    >>> print(f"Watershed cells: {np.sum(mask)}")
    """
    data_path = _DATA_DIR / "watershed_mask.npy"
    if not data_path.exists():
        raise FileNotFoundError(f"Watershed mask data file not found at {data_path}")
    
    return np.load(data_path)

def load_river_mask():
    """
    Load the river mask test data.
    
    A mask showing an example river network for the test domain.
    
    Returns
    -------
    numpy.ndarray
        A 2D array of 0's and 1's showing the location of river cells 
        (1=river, 0=non-river) with shape (215, 172).
        
    Examples
    --------
    >>> import priority_flow.data_loader as dl
    >>> river_mask = dl.load_river_mask()
    >>> print(f"River mask shape: {river_mask.shape}")
    >>> print(f"River cells: {np.sum(river_mask)}")
    """
    data_path = _DATA_DIR / "river_mask.npy"
    if not data_path.exists():
        raise FileNotFoundError(f"River mask data file not found at {data_path}")
    
    return np.load(data_path)

def load_all_test_data():
    """
    Load all test data files at once.
    
    Returns
    -------
    dict
        A dictionary containing all test data arrays with keys:
        - 'dem': Digital Elevation Model
        - 'watershed_mask': Watershed drainage area mask
        - 'river_mask': River network mask
        
    Examples
    --------
    >>> import priority_flow.data_loader as dl
    >>> data = dl.load_all_test_data()
    >>> print(f"Available data: {list(data.keys())}")
    >>> print(f"DEM shape: {data['dem'].shape}")
    """
    return {
        'dem': load_dem(),
        'watershed_mask': load_watershed_mask(),
        'river_mask': load_river_mask()
    }

def get_data_info():
    """
    Get information about the available test data files.
    
    Returns
    -------
    dict
        A dictionary containing metadata about each data file.
    """
    info = {}
    
    for name, load_func in [('DEM', load_dem), 
                           ('Watershed Mask', load_watershed_mask),
                           ('River Mask', load_river_mask)]:
        try:
            data = load_func()
            info[name] = {
                'shape': data.shape,
                'dtype': str(data.dtype),
                'min': float(data.min()),
                'max': float(data.max()),
                'size': data.size
            }
        except FileNotFoundError as e:
            info[name] = {'error': str(e)}
    
    return info


