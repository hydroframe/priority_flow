"""
Calculate Overland Flow functions for PriorityFlow.

This module contains functions for calculating overland flow from ParFlow pressure file outputs.
It supports both OverlandFlow and OverlandKinematic formulations.
"""

import numpy as np
from typing import List, Union, Optional
import os
from parflow.tools.io import read_pfb, write_pfb


def calc_flow(
    file_path: str,
    run_name: str,
    file_nums: List[int],
    slopex_file: str,
    slopey_file: str,
    overland_type: str,
    mannings: Union[float, np.ndarray],
    epsilon: float = 1e-5,
    dx: float = 1.0,
    dy: float = 1.0,
    mask: Optional[np.ndarray] = None
) -> None:
    """
    Calculate overland flow from ParFlow pressure file outputs.
    
    This function will write three pfb outputs for every timestep:
    (1) outflow - volumetric outflow from each grid cell [l3/t]
    (2) q_east - volumetric flow across the east face of each cell [nx+1, ny]
    (3) q_north - volumetric flow across the north face of each cell [nx, ny+1]
    
    Parameters
    ----------
    file_path : str
        Directory path where pressure and slope files are located, also where flow files will be written
    run_name : str
        ParFlow run name used to read pressure files (standard naming: runname.out.press.00000.pfb)
    file_nums : List[int]
        List of file numbers to be processed
    slopex_file : str
        Name of the x slopes file (must be a pfb file)
    slopey_file : str
        Name of the y slopes file (must be a pfb file)
    overland_type : str
        Type of overland flow used for ParFlow simulation.
        Choices: "OverlandFlow" or "OverlandKinematic"
    mannings : Union[float, np.ndarray]
        Manning roughness coefficient, can be either a value or a matrix
    epsilon : float, optional
        Epsilon used only in OverlandKinematic formulation.
        Set with Solver.OverlandKinematic.Epsilon key in ParFlow. Defaults to 1e-5.
    dx : float, optional
        Grid size in x direction [l]. Defaults to 1.0.
    dy : float, optional
        Grid size in y direction [l]. Defaults to 1.0.
    mask : np.ndarray, optional
        Mask with ones for cells to be processed and zeros for everything else.
        Defaults to mask of all 1's. Should be [nx, ny] matrix where [0,0] is lower left corner.
    
    Notes
    -----
    This is a Python port of the R function CalcFlow from PriorityFlow.
    The function calculates overland flow using either OverlandFlow or OverlandKinematic formulations.
    
    File naming convention expected:
    - Pressure files: {run_name}.out.press.{file_num:05d}.pfb
    - Output files: {run_name}.out.{outflow/q_east/q_north}.{file_num:05d}.pfb
    """
    
    # Get the range of file numbers to be read in
    nfile = len(file_nums)
    
    # Read in the slope files
    fin_x = os.path.join(file_path, slopex_file)
    slopex = read_pfb(fin_x, verbose=False)
    fin_y = os.path.join(file_path, slopey_file)
    slopey = read_pfb(fin_y, verbose=False)
    
    # Extract first layer if 3D
    if slopex.ndim == 3:
        slopex = slopex[:, :, 0]
    if slopey.ndim == 3:
        slopey = slopey[:, :, 0]
    
    nx, ny = slopex.shape
    
    # Make a zeros matrix for pmax calculations later
    zeros = np.zeros((nx, ny))
    
    # Handle missing mask
    if mask is None:
        mask = np.ones((nx, ny), dtype=int)
    
    # Test that mask dimensions match slope dimensions
    if mask.shape != (nx, ny):
        raise ValueError("ERROR: X Y dimensions of slope and mask do not match")
    
    for f in range(nfile):
        fn = file_nums[f]
        
        # Read in the pressure file and get grid dimensions
        press_file = os.path.join(file_path, f"{run_name}.out.press.{fn:05d}.pfb")
        press = read_pfb(press_file, verbose=False)
        nz = press.shape[2]
        
        # Test that pressure files and slope files are the same size
        nxtest = press.shape[0]
        nytest = press.shape[1]
        if abs(nxtest - nx) + abs(nytest - ny) != 0:
            raise ValueError("ERROR: X Y dimensions of slope and pressure files do not match")
        
        # Get positive pressures in the top layer for overland flow
        ptop = press[:, :, nz - 1]
        ptop[ptop < 0] = 0
        
        ################################################
        ## Overland Flow
        ################################################
        if overland_type == "OverlandFlow":
            #####
            # Calculate fluxes across east and north faces
            # First the x direction
            qx = -np.sign(slopex) * np.abs(slopex) ** 0.5 / mannings * ptop ** (5/3) * dy
            
            # Upwinding to get flux across the east face of cells
            # Based on qx[i] if positive and qx[i+1] if negative
            qeast = np.maximum(qx[0:nx-1, :], zeros[0:nx-1, :]) - np.maximum(-qx[1:nx, :], zeros[1:nx, :])
            
            # Adding the left boundary - pressures outside domain are 0
            # Flux across this boundary only occurs when qx[0] is negative
            qeast = np.vstack([-np.maximum(-qx[0, :], 0), qeast])
            
            # Adding the right boundary - pressures outside domain are 0
            # Flux across this boundary only occurs when qx[nx-1] is positive
            qeast = np.vstack([qeast, np.maximum(qx[nx-1, :], 0)])
            
            #####
            # Next the y direction
            qy = -np.sign(slopey) * np.abs(slopey) ** 0.5 / mannings * ptop ** (5/3) * dx
            
            # Upwinding to get flux across the north face of cells
            # Based on qy[j] if positive and qy[j+1] if negative
            qnorth = np.maximum(qy[:, 0:ny-1], zeros[:, 0:ny-1]) - np.maximum(-qy[:, 1:ny], zeros[:, 1:ny])
            
            # Adding the bottom boundary - pressures outside domain are 0
            # Flux across this boundary only occurs when qy[:, 0] is negative
            qnorth = np.column_stack([-np.maximum(-qy[:, 0], 0), qnorth])
            
            # Adding the top boundary - pressures outside domain are 0
            # Flux across this boundary only occurs when qy[:, ny-1] is positive
            qnorth = np.column_stack([qnorth, np.maximum(qy[:, ny-1], 0)])
        
        ################################################
        ## Overland Kinematic
        ################################################
        elif overland_type == "OverlandKinematic":
            ####
            # Repeat the slopes on the lower and left boundaries that are inside the domain but outside the mask
            # Find indices of all cells that are off the mask but have a neighbor to their right that is on the mask
            mask_padded = np.vstack([mask[1:nx, :], np.zeros(ny)])
            fill_left = np.where((mask_padded - mask[0:nx, :]) == 1)
            if len(fill_left[0]) > 0:
                # Get the indices of their neighbors to the right
                fill_left2 = (fill_left[0] + 1, fill_left[1])
                # Pad the slopes to the left with their neighboring cells in the mask
                slopex[fill_left] = slopex[fill_left2]
            
            # Find indices of all cells that are off the mask but have a neighbor above them that is on the mask
            mask_padded = np.column_stack([mask[:, 1:ny], np.zeros(nx)])
            fill_down = np.where((mask_padded - mask[:, 0:ny]) == 1)
            if len(fill_down[0]) > 0:
                # Get the indices of their neighbors above
                fill_down2 = (fill_down[0], fill_down[1] + 1)
                # Pad the slopes below with their neighboring cells in the mask
                slopey[fill_down] = slopey[fill_down2]
            
            ####
            # Calculate the slope magnitude
            sfmag = np.maximum((slopex ** 2 + slopey ** 2) ** 0.5, epsilon)
            
            ###
            # For OverlandKinematic slopes are face centered and calculated across the upper and right boundaries
            # (i.e., Z[i+1]-Z[i])
            # For cells on the lower and left boundaries it's assumed that the slopes repeat
            # (i.e., repeating the upper and right face boundary for the lower and left for these border cells)
            slopex_pad = np.vstack([slopex[0, :], slopex])
            slopey_pad = np.column_stack([slopey[:, 0], slopey])
            
            ####
            # Upwind the pressure - Note this is for the north and east face of all cells
            # The slopes are calculated across these boundaries so the upper boundary is included in these
            # calculations and the lower and right boundary of the domain will be added later
            pupwindx = (np.maximum(np.sign(slopex) * np.vstack([ptop[1:nx, :], np.zeros(ny)]), 0) +
                       np.maximum(-np.sign(slopex) * ptop[0:nx, :], 0))
            pupwindy = (np.maximum(np.sign(slopey) * np.column_stack([ptop[:, 1:ny], np.zeros(nx)]), 0) +
                       np.maximum(-np.sign(slopey) * ptop[:, 0:ny], 0))
            
            ###
            # Calculate fluxes across east and north faces
            # First the x direction
            qeast = -slopex / (sfmag ** 0.5 * mannings) * pupwindx ** (5/3) * dy
            qnorth = -slopey / (sfmag ** 0.5 * mannings) * pupwindy ** (5/3) * dx
            
            ###
            # Fix the lower x boundary
            # Use the slopes of the first column with the pressures for cell i
            qleft = -slopex[0, :] / (sfmag[0, :] ** 0.5 * mannings) * (np.maximum(np.sign(slopex[0, :]) * ptop[0, :], 0)) ** (5/3) * dy
            qeast = np.vstack([qleft, qeast])
            
            ###
            # Fix the lower y boundary
            # Use the slopes of the bottom row with the pressures for cell j
            qbottom = -slopey[:, 0] / (sfmag[:, 0] ** 0.5 * mannings) * (np.maximum(np.sign(slopey[:, 0]) * ptop[:, 0], 0)) ** (5/3) * dx
            qnorth = np.column_stack([qbottom, qnorth])
        
        else:
            raise ValueError("Invalid overland_type: You must select either 'OverlandFlow' or 'OverlandKinematic'")
        
        # Calculate total outflow
        # Outflow is a positive qeast[i,j] or qnorth[i,j] or a negative qeast[i-1,j], qnorth[i,j-1]
        outflow = (np.maximum(qeast[1:nx+1, :], zeros) + np.maximum(-qeast[0:nx, :], zeros) +
                  np.maximum(qnorth[:, 1:ny+1], zeros) + np.maximum(-qnorth[:, 0:ny], zeros))
        
        # Write the outputs
        outflow_file = os.path.join(file_path, f"{run_name}.out.outflow.{fn:05d}.pfb")
        write_pfb(outflow, outflow_file, dx, dy, 1)
        
        qeast_file = os.path.join(file_path, f"{run_name}.out.q_east.{fn:05d}.pfb")
        write_pfb(qeast, qeast_file, dx, dy, 1)
        
        qnorth_file = os.path.join(file_path, f"{run_name}.out.q_north.{fn:05d}.pfb")
        write_pfb(qnorth, qnorth_file, dx, dy, 1)


 