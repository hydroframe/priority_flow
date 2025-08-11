"""
Write Raster functions for PriorityFlow.

This module provides functions to write raster data to ASCII format
compatible with GIS software like ArcGIS, QGIS, and others.
"""

import numpy as np
from typing import Union


def write_raster(
    data: np.ndarray,
    fout: str,
    xllcorner: float = 0.0,
    yllcorner: float = 0.0,
    dx: float = 1000.0,
    naval: Union[int, float] = -999
) -> None:
    """
    Write raster text file for GIS software.
    
    This function writes out a matrix with the six header rows needed for GIS rasters.
    The output format is compatible with ArcGIS, QGIS, GRASS GIS, and other GIS software.
    
    Parameters
    ----------
    data : np.ndarray
        Matrix of values for the raster (2D numpy array)
    fout : str
        File name for output (full path or filename)
    xllcorner : float, optional
        X-coordinate of the lower left corner of the domain. Defaults to 0.0
    yllcorner : float, optional
        Y-coordinate of the lower left corner of the domain. Defaults to 0.0
    dx : float, optional
        Grid cell lateral dimension/resolution. Defaults to 1000.0
    naval : Union[int, float], optional
        Value assigned to NaN/None values in the raster. Defaults to -999
    
    Returns
    -------
    None
        The function writes to file and does not return a value.
    
    Notes
    -----
    This function creates an ASCII raster file with the following header structure:
    
    ncols      [number of columns]
    nrows      [number of rows]
    xllcorner      [x coordinate of lower left corner]
    yllcorner      [y coordinate of lower left corner]
    cellsize      [grid cell size]
    NODATA_value      [no data value]
    
    The header is followed by the data matrix, with each row of the matrix
    written as a separate line in the output file.
    
    The function handles:
    - Automatic header generation based on data dimensions
    - Customizable spatial reference parameters
    - Missing data value specification
    - GIS-compatible formatting
    
    Examples
    --------
    Basic usage with default parameters:
    
    >>> import numpy as np
    >>> from priority_flow.write_raster import write_raster
    >>> data = np.random.rand(100, 200)
    >>> write_raster(data, "output.asc")
    
    Custom parameters for UTM coordinates:
    
    >>> write_raster(data, "output_utm.asc", 
    ...              xllcorner=500000, yllcorner=4000000, dx=30)
    
    Custom missing data value:
    
    >>> write_raster(data, "output_custom.asc", naval=-32768)
    
    Notes on GIS Compatibility:
    - ArcGIS: Use "ASCII to Raster" tool for import
    - QGIS: Native support for ASCII raster format
    - GRASS GIS: Compatible with r.in.ascii module
    - SAGA GIS: Supports ASCII raster import
    
    Coordinate System Considerations:
    - The function does not include projection information
    - Users must manually specify the coordinate system in GIS software
    - Supports any coordinate system (UTM, geographic, projected, etc.)
    - Assumes square grid cells (dx = dy)
    
    File Format Details:
    - ASCII text format (human readable)
    - Space-separated values
    - No row or column headers
    - Standard GIS raster format
    
    Performance Considerations:
    - Suitable for small to medium-sized rasters
    - For very large datasets, consider binary formats (GeoTIFF, NetCDF)
    - ASCII format provides maximum compatibility but larger file sizes
    """
    
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
    
    # Input validation
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy array")
    
    if data.ndim != 2:
        raise ValueError("data must be a 2D array")
    
    if not isinstance(fout, str):
        raise TypeError("fout must be a string")
    
    if not isinstance(xllcorner, (int, float)):
        raise TypeError("xllcorner must be a number")
    
    if not isinstance(yllcorner, (int, float)):
        raise TypeError("yllcorner must be a number")
    
    if not isinstance(dx, (int, float)) or dx <= 0:
        raise TypeError("dx must be a positive number")
    
    if not isinstance(naval, (int, float)):
        raise TypeError("naval must be a number")
    
    # Get dimensions
    nrows, ncols = data.shape
    
    # Create header lines
    header_lines = [
        f"ncols      {ncols}",
        f"nrows      {nrows}",
        f"xllcorner      {xllcorner}",
        f"yllcorner      {yllcorner}",
        f"cellsize      {dx}",
        f"NODATA_value      {naval}"
    ]
    
    # Handle NaN/None values in data
    data_export = data.copy()
    if np.any(np.isnan(data_export)):
        data_export = np.where(np.isnan(data_export), naval, data_export)
    
    # Write header and data to file
    try:
        with open(fout, 'w') as f:
            # Write header
            for header_line in header_lines:
                f.write(header_line + '\n')
            
            # Write data
            for i in range(nrows):
                row_data = data_export[i, :]
                # Convert to string with space separation, no scientific notation for small numbers
                row_str = ' '.join([f"{val:.6g}" if isinstance(val, (int, float)) else str(val) 
                                   for val in row_data])
                f.write(row_str + '\n')
                
    except IOError as e:
        raise IOError(f"Failed to write raster file '{fout}': {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error writing raster file: {str(e)}") 