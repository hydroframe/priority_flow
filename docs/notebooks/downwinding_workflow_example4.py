#!/usr/bin/env python3
"""
Downwinding Workflow Example 4 (Python)

Translation of `Downwinding_Workflow_Example4.R` from the PriorityFlow R package.

Scenario:
- Irregular domain (watershed mask) and a pre-defined river network used for DEM processing.
- Calculating slopes with a downwinding approach consistent with ParFlow's OverlandFlow
  boundary condition (SlopeCalcUP in R, slope_calc_upwind in Python).
- Requires three inputs: (1) DEM, (2) river mask, (3) watershed mask.

This example uses the test domain from Condon and Maxwell (2019);
datasets are provided with the PriorityFlow package. See help(DEM), help(watershed.mask),
and help(river.mask) in the R package for more information.
"""

import numpy as np

from priority_flow import (
    init_queue,
    d4_traverse_b,
    load_dem,
    load_river_mask,
    load_watershed_mask,
    get_border,
    drainage_area,
    calc_subbasins,
    slope_calc_upwind,
    stream_traverse,
    find_orphan,
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
    ep = 0.01  # epsilon value applied to flat cells

    # Slope scaling
    maxslope = 0.5  # maximum slope; set to -1 to disable
    minslope = 1e-5  # minimum slope; set to -1 to disable
    scale = 0.1  # max ratio of secondary to primary flow direction (secondaryTH)

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
    # Load DEM, river mask, and watershed mask
    # -------------------------------------------------------------------------
    DEM = load_dem()
    river_mask = load_river_mask()
    watershed_mask = load_watershed_mask()
    nx, ny = DEM.shape
    print(f"Domain size: nx={nx}, ny={ny}")

    # -------------------------------------------------------------------------
    # Process the DEM:
    # 1. Initialize the queue with river cells that fall on the border
    # 2. Traverse the stream network (first pass)
    # 3. Find orphan branches and continue until all connected
    # 4. Initialize queue with processed river boundary + watershed border
    # 5. Process hillslopes from there
    # -------------------------------------------------------------------------

    # 1. Initialize the queue with river cells on the border (irregular domain)
    init = init_queue(
        DEM,
        initmask=river_mask,
        domainmask=watershed_mask,
    )

    # 2. First pass: traverse the streams within the river mask
    trav1 = stream_traverse(
        dem=DEM,
        mask=river_mask,
        queue=init["queue"].copy(),
        marked=init["marked"].copy(),
        basins=init["basins"].copy(),
        printstep=False,
        epsilon=ep,
    )
    first_pass_pct = (
        np.sum(trav1["marked"] * river_mask) / np.sum(river_mask) * 100.0
        if np.sum(river_mask) > 0
        else 0.0
    )
    print(f"First Pass: {first_pass_pct:.1f} % cells processed")

    # 3. Look for orphaned branches and continue traversing until all connected
    norphan = 1
    lap = 1
    while norphan > 0:
        orphan = find_orphan(
            dem=trav1["dem"],
            mask=river_mask,
            marked=trav1["marked"],
        )
        norphan = int(orphan["norphan"])
        print(f"lap {lap}: {norphan} orphans found")

        if norphan > 0:
            trav2 = stream_traverse(
                dem=trav1["dem"],
                mask=river_mask,
                queue=orphan["queue"],
                marked=trav1["marked"],
                step=trav1["step"],
                direction=trav1["direction"],
                basins=trav1["basins"],
                printstep=False,
                epsilon=ep,
            )
            trav1 = trav2
            lap += 1
        else:
            print("Done! No orphan branches found")

    final_pass_pct = (
        np.sum(trav1["marked"] * river_mask) / np.sum(river_mask) * 100.0
        if np.sum(river_mask) > 0
        else 0.0
    )
    print(f"Final pass: {final_pass_pct:.1f} % cells processed")

    # 4. Initialize the queue with every cell on the processed river boundary
    #    (marked rivers from last step plus watershed border cells)
    border_t = get_border(watershed_mask)
    riv_border = border_t + trav1["marked"]
    riv_border[riv_border > 1] = 1

    init2 = init_queue(trav1["dem"], border=riv_border)

    # 5. Process all cells off the river using the river as the boundary (irregular domain)
    trav_hs = d4_traverse_b(
        dem=trav1["dem"],
        queue=init2["queue"].copy(),
        marked=init2["marked"].copy(),
        direction=trav1["direction"].copy(),
        basins=trav1["basins"].copy(),
        step=trav1["step"].copy(),
        epsilon=ep,
        mask=watershed_mask,
    )

    # -------------------------------------------------------------------------
    # Calculate the slopes
    # -------------------------------------------------------------------------

    # Option 1: Slopes for entire domain with no distinction between river and hillslope cells
    slopes_uw = slope_calc_upwind(
        dem=trav_hs["dem"].copy(),
        mask=watershed_mask.copy(),
        direction=trav_hs["direction"].copy(),
        dx=dx,
        dy=dy,
        secondary_th=scale,
        maxslope=maxslope,
        minslope=minslope,
    )

    # Option 2: Handle river cells differently using subbasins
    area = drainage_area(
        trav_hs["direction"].copy(),
        printflag=False,
    )

    subbasin = calc_subbasins(
        trav_hs["direction"].copy(),
        area=area,
        mask=watershed_mask.copy(),
        riv_th=sub_th,
        merge_th=mrg_th,
    )

    slopes_uw = slope_calc_upwind(
        dem=trav_hs["dem"].copy(),
        mask=watershed_mask.copy(),
        direction=trav_hs["direction"].copy(),
        dx=dx,
        dy=dy,
        secondary_th=scale,
        maxslope=maxslope,
        minslope=minslope,
        river_method=riv_method,
        rivermask=subbasin["RiverMask"].copy(),
        subbasins=subbasin["subbasins"].copy(),
    )

    # Option 2b: River mask from drainage area threshold (riv_th)
    rivers = np.zeros_like(area)
    rivers[area < riv_th] = 0
    rivers[area >= riv_th] = 1

    slopes_uw = slope_calc_upwind(
        dem=trav_hs["dem"].copy(),
        mask=watershed_mask.copy(),
        direction=trav_hs["direction"].copy(),
        dx=dx,
        dy=dy,
        secondary_th=0.1,
        maxslope=maxslope,
        minslope=minslope,
        river_method=riv_method,
        rivermask=rivers,
        subbasins=subbasin["subbasins"].copy(),
    )

    # -------------------------------------------------------------------------
    # Recalculate drainage area from final directions
    # -------------------------------------------------------------------------
    area = drainage_area(
        slopes_uw["direction"],
        mask=watershed_mask,
        printflag=False,
    )

    # -------------------------------------------------------------------------
    # Write slopes in ParFlow ASCII format
    # -------------------------------------------------------------------------
    slopex = slopes_uw["slopex"]
    slopey = slopes_uw["slopey"]
    write_parflow_ascii(slopex, f"{runname}.slopex.sa")
    write_parflow_ascii(slopey, f"{runname}.slopey.sa")

    # -------------------------------------------------------------------------
    # Write other variables as matrices (R: t(...[,ny:1]) = flip column order)
    # -------------------------------------------------------------------------
    write_parflow_ascii(slopes_uw["slopex"], f"{runname}.slopex.sa")
    write_parflow_ascii(slopes_uw["slopey"], f"{runname}.slopey.sa")
    write_parflow_ascii(np.flip(slopes_uw["direction"], axis=1).T, f"{runname}.direction.sa")
    write_parflow_ascii(np.flip(trav_hs["dem"], axis=1).T, f"{runname}.dem.sa")
    write_parflow_ascii(np.flip(area, axis=1).T, f"{runname}.area.sa")
    write_parflow_ascii(np.flip(subbasin["subbasins"], axis=1).T, f"{runname}.subbasins.sa")
    write_parflow_ascii(np.flip(subbasin["segments"], axis=1).T, f"{runname}.subbasin_streams.sa")
    write_parflow_ascii(subbasin["summary"], f"{runname}.Subbasin_Summary.sa")

    print("Downwinding workflow example 4 complete.")


if __name__ == "__main__":
    main()
