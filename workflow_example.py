#!/usr/bin/env python3
"""
Priority Flow Workflow Example

This script walks through a typical workflow for processing a DEM to ensure a fully
connected hydrologic drainage network. For more details on the Priority Flow tool
refer to Condon and Maxwell (2019): https://doi.org/10.1016/j.cageo.2019.01.020

This is a Python translation of the R vignette Workflow_Example.Rmd from the
PriorityFlow R package (https://github.com/lecondon/PriorityFlow).

The example uses the sample watershed from Condon and Maxwell (2019). The DEM and
mask files are provided with the PriorityFlow library.
"""

import sys
from pathlib import Path

# Ensure package is on path when run from project root
_project_root = Path(__file__).resolve().parent
_src = _project_root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import matplotlib.pyplot as plt

# Import from priority_flow package (all at package level via __init__.py)
from priority_flow import (
    init_queue,
    d4_traverse_b,
    load_dem,
    load_watershed_mask,
    load_river_mask,
    drainage_area,
    calc_subbasins,
    calc_stream_order,
    river_smooth,
    path_extract,
    slope_calc_standard,
    riv_slope,
)


# =============================================================================
# Background Information and Setup
# =============================================================================
# Three inputs for domain processing:
# 1. The unprocessed DEM (Digital Elevation Model)
# 2. A mask of the watershed we are interested in
# 3. A mask of our desired drainage network
#
# NOTE: The only required input is a DEM; the other two are optional depending
# on how you would like to process things.

# Load test data
DEM = load_dem()
watershed_mask = load_watershed_mask()
river_mask = load_river_mask()

nx, ny = DEM.shape
print(f"Domain dimensions: nx={nx}, ny={ny}")
print(f"DEM elevation range: {DEM.min():.2f} to {DEM.max():.2f}")

# Plot inputs (optional - requires matplotlib)
def _plot_inputs():
    """Plot the three input datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(DEM, origin="lower")
    axes[0].set_title("Elevation")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(watershed_mask, origin="lower")
    axes[1].set_title("Watershed Mask")
    plt.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(river_mask, origin="lower")
    axes[2].set_title("River Network")
    plt.colorbar(im2, ax=axes[2])
    plt.tight_layout()
    plt.savefig("workflow_inputs.png", dpi=150)
    plt.close()


# =============================================================================
# Step 1: Processing the DEM
# =============================================================================
# The priority flow algorithm traverses the raw DEM and ensures that every grid
# cell has a pathway to exit along D4 drainage pathways (no unintended internal
# sinks).
#
# Outcomes:
# 1. A processed DEM with elevations adjusted to ensure drainage
# 2. A map of flow directions (1=down, 2=left, 3=up, 4=right)
#
# Option 2: Process DEM within a pre-defined watershed mask (used here)
# The mask file should have values of 1 for cells inside the domain and 0 elsewhere.

# Setup target points
init = init_queue(DEM, domain_mask=watershed_mask)

# Process the DEM
trav_hs = d4_traverse_b(
    DEM,
    init["queue"],
    init["marked"],
    mask=watershed_mask,
    basins=init["basins"],
    epsilon=0,
    n_chunk=10,
)

# Calculations for plotting
dem_diff = trav_hs["dem"] - DEM
dem_diff = np.where(dem_diff == 0, np.nan, dem_diff)
targets = np.where(init["marked"] == 0, np.nan, init["marked"])

# Plotting Step 1 results (optional)
def _plot_step1():
    """Plot DEM processing results."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(np.where(np.isnan(targets), 0, 1), origin="lower")
    axes[0, 0].set_title("Target Points")
    im1 = axes[0, 1].imshow(trav_hs["dem"], origin="lower")
    axes[0, 1].set_title("Processed DEM")
    plt.colorbar(im1, ax=axes[0, 1])
    im2 = axes[1, 0].imshow(dem_diff, origin="lower")
    axes[1, 0].set_title("DEM differences")
    plt.colorbar(im2, ax=axes[1, 0])
    im3 = axes[1, 1].imshow(trav_hs["direction"], origin="lower")
    axes[1, 1].set_title("Flow Direction")
    plt.colorbar(im3, ax=axes[1, 1])
    plt.tight_layout()
    plt.savefig("workflow_step1.png", dpi=150)
    plt.close()


# =============================================================================
# Step 2: Smoothing along the drainage network
# =============================================================================

# Step 2.1: Calculate drainage areas
area = drainage_area(
    trav_hs["direction"],
    mask=watershed_mask,
    printflag=False,
)

# Step 2.2: Define river network using drainage area threshold
# riv_th: cells with >= riv_th cells draining to it count as rivers
subbasin = calc_subbasins(
    trav_hs["direction"],
    area=area,
    mask=watershed_mask,
    riv_th=60,
    merge_th=0,
)

# Calculate stream order (optional)
stream_order = calc_stream_order(
    subbasin["summary"][:, 0],
    subbasin["summary"][:, 5],
    subbasin["segments"].copy(),
)

# Step 2.3: Smooth the DEM along river segments
riv_smooth_result = river_smooth(
    dem=trav_hs["dem"],
    direction=trav_hs["direction"],
    mask=watershed_mask,
    river_summary=subbasin["summary"],
    river_segments=subbasin["segments"],
    bank_epsilon=1,
)

# Plot elevation differences from river smoothing
def _plot_step2():
    """Plot river smoothing results."""
    dif = riv_smooth_result["dem.adj"] - trav_hs["dem"]
    riv_mask = np.where(subbasin["segments"] > 0, 1, 0)
    hill_mask = 1 - riv_mask
    dif_hill = dif * hill_mask
    dif_riv = dif * riv_mask

    dif_plot = np.where(dif == 0, np.nan, dif)
    dif_riv_plot = np.where(dif_riv == 0, np.nan, dif_riv)
    dif_hill_plot = np.where(dif_hill == 0, np.nan, dif_hill)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(dif_plot, origin="lower")
    axes[0].set_title("All Elev. Diffs")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(dif_riv_plot, origin="lower")
    axes[1].set_title("Stream Cell Diffs")
    plt.colorbar(im1, ax=axes[1])
    if np.any(~np.isnan(dif_hill_plot)):
        im2 = axes[2].imshow(dif_hill_plot, origin="lower")
        axes[2].set_title("Non-Stream Cell Diffs")
        plt.colorbar(im2, ax=axes[2])
    plt.tight_layout()
    plt.savefig("workflow_step2_smoothing.png", dpi=150)
    plt.close()


# PathExtract example: walk downstream from a stream segment start
segment = 30  # Pick a stream segment (0-indexed, change to see different segment)
start = np.array([[subbasin["summary"][segment, 1], subbasin["summary"][segment, 2]]])

streamline_old = path_extract(
    trav_hs["dem"],
    trav_hs["direction"],
    mask=watershed_mask,
    startpoint=start,
)
streamline_new = path_extract(
    riv_smooth_result["dem.adj"],
    trav_hs["direction"],
    mask=watershed_mask,
    startpoint=start,
)
streamline_riv = path_extract(
    subbasin["segments"],
    trav_hs["direction"],
    mask=watershed_mask,
    startpoint=start,
)

transect_old = streamline_old["data"]
transect_new = streamline_new["data"]
transect_riv = streamline_riv["data"]


# =============================================================================
# Step 3: Calculate the slopes
# =============================================================================
# Using SlopeCalcStan for OverlandKinematic or OverlandDiffusive boundary
# conditions in ParFlow. For OverlandFlow, use SlopeCalcUP instead.

# 3.1: Calculate slopes for the entire domain
slopes_calc = slope_calc_standard(
    dem=riv_smooth_result["dem.adj"],
    direction=trav_hs["direction"],
    mask=watershed_mask,
    minslope=1e-5,
    maxslope=1,
    dx=1000,
    dy=1000,
    secondary_th=-1,
)

# 3.2: Adjust slopes along river cells
river_mask_slope = np.where(subbasin["segments"] > 0, 1, 0)
slopes_calc2 = riv_slope(
    direction=trav_hs["direction"],
    slopex=slopes_calc["slopex"],
    slopey=slopes_calc["slopey"],
    minslope=1e-4,
    river_mask=river_mask_slope,
    remove_sec=True,
)

slopex = slopes_calc2["slopex"]
slopey = slopes_calc2["slopey"]


# =============================================================================
# Step 4: Write slope files in ParFlow ASCII format
# =============================================================================
# These ASCII files can be converted to pfb or silo using standard pftools.

def write_parflow_ascii(data: np.ndarray, filepath: str) -> None:
    """
    Write a 2D array in ParFlow ASCII format.
    Format: header line with nx, ny, 1 followed by flattened data.
    """
    nx, ny = data.shape
    flat = np.zeros(nx * ny)
    jj = 0
    for j in range(ny):
        for i in range(nx):
            flat[jj] = data[i, j]
            jj += 1
    with open(filepath, "w") as f:
        f.write(f"{nx} {ny} 1\n")
        for val in flat:
            f.write(f"{val}\n")


# Uncomment to write output files
write_parflow_ascii(slopex, "SlopeX.sa")
write_parflow_ascii(slopey, "SlopeY.sa")
write_parflow_ascii(riv_smooth_result["dem.adj"], "DEM.Processed.sa")
write_parflow_ascii(trav_hs["direction"], "Flow.Direction.sa")


# =============================================================================
# Main: run workflow and optionally generate plots
# =============================================================================
if __name__ == "__main__":
    print("Priority Flow Workflow Example")
    print("=" * 40)
    print("Step 1: DEM processing complete")
    print(f"  Processed DEM shape: {trav_hs['dem'].shape}")
    print("Step 2: River smoothing complete")
    print(f"  Number of subbasins: {len(subbasin['summary'])}")
    print("Step 3: Slope calculation complete")
    print(f"  Slopex range: {slopex.min():.6f} to {slopex.max():.6f}")

    try:
        _plot_inputs()
        _plot_step1()
        _plot_step2()
        print("\nPlots saved: workflow_inputs.png, workflow_step1.png, workflow_step2_smoothing.png")
    except Exception as e:
        print(f"\nSkipping plots: {e}")
