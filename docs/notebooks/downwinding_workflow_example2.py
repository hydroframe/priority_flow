#!/usr/bin/env python3
"""
Downwinding Workflow Example 2 (Python)

Translation of `Downwinding_Workflow_Example2.R` from the PriorityFlow R package.

Scenario:
- Irregular watershed boundary (mask) and no river mask provided a priori
- DEM processing with PriorityFlow limited to the watershed mask
- Slope calculation using upwind/downwinding approach consistent with ParFlow's
  OverlandFlow boundary condition (SlopeCalcUP in R, slope_calc_upwind in Python)

Inputs:
- A DEM (`load_dem`)
- A watershed mask (`load_watershed_mask`)
"""

import numpy as np

from priority_flow import (
    init_queue,
    d4_traverse_b,
    load_dem,
    load_watershed_mask,
    drainage_area,
    calc_subbasins,
    slope_calc_upwind,
)


def write_parflow_ascii(data: np.ndarray, filepath: str) -> None:
    """
    Write a 2D array in ParFlow ASCII (.sa) format.

    Format:
        First line: "nx ny 1"
        Then nx*ny values, one per line, flattened in column-major order
        (j varies slowest, i varies fastest).
    """
    nx, ny = data.shape
    flat = np.zeros(nx * ny, dtype=float)
    jj = 0
    for j in range(ny):
        for i in range(nx):
            flat[jj] = data[i, j]
            jj += 1

    with open(filepath, "w") as f:
        f.write(f"{nx} {ny} 1\n")
        for val in flat:
            f.write(f"{val}\n")


def main() -> None:
    # -------------------------------------------------------------------------
    # Settings (mirroring the R script)
    # -------------------------------------------------------------------------
    # DEM processing
    ep = 0.01  # epsilon for D4TraverseB (PriorityFlow processing)

    # Slope scaling
    maxslope = 0.5  # maximum slope; set to -1 to disable
    minslope = 1e-5  # minimum slope; set to -1 to disable
    scale = 0.1  # max ratio of |secondary| / |primary| (secondaryTH)

    # River and subbasin size for slope calculations
    sub_th = 100  # area threshold (cells) for subbasin delineation
    riv_th = sub_th  # optional: threshold for river mask for slope processing
    riv_method = 3  # 0: none, 1: scale river secondary, 2: basin mean, 3: reach mean
    mrg_th = 10  # merge threshold for small subbasins

    # Grid dimensions for slopes
    dx = 1000.0
    dy = 1000.0

    # Run name for outputs
    runname = "Test"

    # -------------------------------------------------------------------------
    # Load DEM and watershed mask
    # -------------------------------------------------------------------------
    DEM = load_dem()
    watershed_mask = load_watershed_mask()
    nx, ny = DEM.shape
    print(f"Domain size: nx={nx}, ny={ny}")

    # -------------------------------------------------------------------------
    # Process the DEM: initialize queue and traverse within watershed mask
    # -------------------------------------------------------------------------
    # 1. Initialize queue using the irregular watershed mask
    init = init_queue(DEM, domainmask=watershed_mask)

    # 2. Process DEM so that all cells inside the mask drain to the watershed boundary
    trav_hs = d4_traverse_b(
        DEM,
        init["queue"].copy(),
        init["marked"].copy(),
        basins=init["basins"].copy(),
        epsilon=ep,
        mask=watershed_mask,
    )

    print("DEM processing complete (irregular watershed mask).")

    # -------------------------------------------------------------------------
    # Option 1: Slopes for entire masked domain (no special river treatment)
    # -------------------------------------------------------------------------
    # Upwind slope calculation consistent with OverlandFlow BC in ParFlow,
    # applied only within the watershed mask.
    slopes_uw = slope_calc_upwind(
        dem=trav_hs["dem"].copy(),
        direction=trav_hs["direction"].copy(),
        dx=dx,
        dy=dy,
        mask=watershed_mask.copy(),
        minslope=minslope,
        maxslope=maxslope,
        secondary_th=scale,
    )

    print("Option 1 slopes (masked domain, no river distinction) computed.")

    # -------------------------------------------------------------------------
    # Option 2: Treat river cells differently using subbasins
    # -------------------------------------------------------------------------
    # Calculate drainage area from processed directions (no mask here, as in R)
    area = drainage_area(trav_hs["direction"], printflag=False)

    # Define subbasins for calculating river reach slopes
    # R: CalcSubbasins(travHS$direction, mask=watershed.mask, area, riv_th=sub_th, merge_th=mrg_th)
    subbasin = calc_subbasins(
        trav_hs["direction"].copy(),
        area=area,
        mask=watershed_mask.copy(),
        riv_th=sub_th,
        merge_th=mrg_th,
    )

    print("Subbasins and river mask computed for slope option 2.")

    # Upwind slopes with river-specific method, using subbasin RiverMask
    slopes_uw = slope_calc_upwind(
        dem=trav_hs["dem"].copy(),
        direction=trav_hs["direction"].copy(),
        dx=dx,
        dy=dy,
        mask=watershed_mask.copy(),
        minslope=minslope,
        maxslope=maxslope,
        secondary_th=scale,
        river_method=riv_method,
        rivermask=subbasin["RiverMask"].copy(),
        subbasins=subbasin["subbasins"].copy(),
    )

    print("Option 2 slopes (river-specific within watershed) computed.")

    # -------------------------------------------------------------------------
    # Option 2b: Alternate river mask separate from subbasin river mask
    # -------------------------------------------------------------------------
    # Create a river mask based on riv_th, possibly different from sub_th
    rivers = np.zeros_like(area)
    rivers[area < riv_th] = 0
    rivers[area >= riv_th] = 1

    # Upwind slopes using alternate river mask but same subbasins
    slopes_uw = slope_calc_upwind(
        dem=trav_hs["dem"].copy(),
        direction=trav_hs["direction"].copy(),
        dx=dx,
        dy=dy,
        mask=watershed_mask.copy(),
        minslope=minslope,
        maxslope=maxslope,
        secondary_th=scale,
        river_method=riv_method,
        rivermask=rivers,
        subbasins=subbasin["subbasins"].copy(),
    )

    print("Option 2b slopes (alternate river mask within watershed) computed.")

    # -------------------------------------------------------------------------
    # (Re)calculate drainage area if needed using final directions (masked)
    # -------------------------------------------------------------------------
    area = drainage_area(trav_hs["direction"], mask=watershed_mask, printflag=False)
    print("Drainage area within watershed mask computed.")

    # -------------------------------------------------------------------------
    # Write slopes in ParFlow ASCII format
    # -------------------------------------------------------------------------
    slopex = slopes_uw["slopex"]
    slopey = slopes_uw["slopey"]

    sx_file = f"{runname}.slopex.sa"
    sy_file = f"{runname}.slopey.sa"
    write_parflow_ascii(slopex, sx_file)
    write_parflow_ascii(slopey, sy_file)
    print(f"Wrote slopex to {sx_file}")
    print(f"Wrote slopey to {sy_file}")

    # -------------------------------------------------------------------------
    # Optional: write additional matrices as plain text (similar to R examples)
    # -------------------------------------------------------------------------
    np.savetxt(
        f"{runname}.direction.out.txt",
        trav_hs["direction"],
        fmt="%.8g",
    )
    np.savetxt(f"{runname}.dem.out.txt", trav_hs["dem"], fmt="%.8g")
    np.savetxt(f"{runname}.area.out.txt", area, fmt="%.8g")
    np.savetxt(f"{runname}.subbasins.out.txt", subbasin["subbasins"], fmt="%.8g")
    np.savetxt(f"{runname}.subbasin_streams.out.txt", subbasin["segments"], fmt="%.8g")

    print("Additional outputs written (direction, DEM, area, subbasins, streams).")


if __name__ == "__main__":
    main()

