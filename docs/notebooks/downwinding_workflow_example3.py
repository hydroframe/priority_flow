#!/usr/bin/env python3
"""
Downwinding Workflow Example 3 (Python)

Translation of `Downwinding_Workflow_Example3.R` from the PriorityFlow R package.

Scenario:
- Rectangular domain and a pre-defined river network (river mask) used explicitly
  in the DEM processing.
- DEM processing uses stream traversal along the provided river network, with
  orphan-branch detection and reconnection.
- Slope calculation uses an upwind/downwinding approach consistent with
  ParFlow's OverlandFlow boundary condition (SlopeCalcUP in R,
  slope_calc_upwind in Python).

Inputs:
- A DEM (`load_dem`)
- A river mask (`load_river_mask`)
"""

import numpy as np

from priority_flow import (
    init_queue,
    d4_traverse_b,
    load_dem,
    load_river_mask,
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
    ep = 0.01  # epsilon for stream/DEM processing

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
    # Load DEM and river mask
    # -------------------------------------------------------------------------
    DEM = load_dem()
    river_mask = load_river_mask()
    nx, ny = DEM.shape
    print(f"Domain size: nx={nx}, ny={ny}")

    # -------------------------------------------------------------------------
    # Step 1: Process the DEM along the river network
    # -------------------------------------------------------------------------
    # 1. Initialize the queue with river cells that fall on the border
    init = init_queue(DEM, initmask=river_mask)

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
    print(f"First Pass: {first_pass_pct:.1f} % river cells processed")

    # 3. Look for orphaned branches and continue traversing until they are all connected
    # Orphaned branches are portions of the river network connected only diagonally.
    norphan = 1
    lap = 1
    while norphan > 0:
        orphan = find_orphan(
            dem=trav1["dem"],
            mask=river_mask,
            marked=trav1["marked"],
        )
        norphan = int(orphan["norphan"])
        print(f"Lap {lap}: {norphan} orphans found")

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
    print(f"Final pass: {final_pass_pct:.1f} % river cells processed")

    # -------------------------------------------------------------------------
    # Step 2: Process hillslopes using the processed river as boundary
    # -------------------------------------------------------------------------
    # 4. Initialize the queue with every cell on the processed river boundary
    #    using the marked rivers from the last step plus the rectangular edges.
    inittemp = init_queue(DEM)  # rectangular boundary mask
    riv_border = inittemp["marked"] + trav1["marked"]
    riv_border[riv_border > 1] = 1

    init2 = init_queue(trav1["dem"], border=riv_border)

    # 5. Process all hillslope cells off the river using the river as the boundary
    trav_hs = d4_traverse_b(
        dem=trav1["dem"],
        queue=init2["queue"].copy(),
        marked=init2["marked"].copy(),
        direction=trav1["direction"].copy(),
        basins=trav1["basins"].copy(),
        step=trav1["step"].copy(),
        epsilon=ep,
    )

    print("DEM processing complete (streams + hillslopes).")

    # -------------------------------------------------------------------------
    # Step 3: Calculate the slopes
    # -------------------------------------------------------------------------
    # Option 1: Slopes for the entire domain with no special river treatment
    slopes_uw = slope_calc_upwind(
        dem=trav_hs["dem"].copy(),
        direction=trav_hs["direction"].copy(),
        dx=dx,
        dy=dy,
        minslope=minslope,
        maxslope=maxslope,
        secondary_th=scale,
    )
    print("Option 1 slopes (no river distinction) computed.")

    # Option 2: Treat river cells differently using subbasins (river cells from drainage area)
    area = drainage_area(trav_hs["direction"], printflag=False)

    subbasin = calc_subbasins(
        trav_hs["direction"].copy(),
        area=area,
        mask=None,
        riv_th=sub_th,
        merge_th=mrg_th,
    )
    print("Subbasins and river mask computed for slope option 2.")

    slopes_uw = slope_calc_upwind(
        dem=trav_hs["dem"].copy(),
        direction=trav_hs["direction"].copy(),
        dx=dx,
        dy=dy,
        minslope=minslope,
        maxslope=maxslope,
        secondary_th=scale,
        river_method=riv_method,
        rivermask=subbasin["RiverMask"].copy(),
        subbasins=subbasin["subbasins"].copy(),
    )
    print("Option 2 slopes (river-specific, drainage-area-based) computed.")

    # Option 2b: Alternate river mask based on drainage area threshold riv_th
    rivers = np.zeros_like(area)
    rivers[area < riv_th] = 0
    rivers[area >= riv_th] = 1

    slopes_uw = slope_calc_upwind(
        dem=trav_hs["dem"].copy(),
        direction=trav_hs["direction"].copy(),
        dx=dx,
        dy=dy,
        minslope=minslope,
        maxslope=maxslope,
        secondary_th=scale,
        river_method=riv_method,
        rivermask=rivers,
        subbasins=subbasin["subbasins"].copy(),
    )
    print("Option 2b slopes (alternate drainage-area river mask) computed.")

    # -------------------------------------------------------------------------
    # Step 4: Recalculate drainage area from final directions
    # -------------------------------------------------------------------------
    area = drainage_area(slopes_uw["direction"], printflag=False)
    print("Drainage area recomputed from final directions.")

    # -------------------------------------------------------------------------
    # Step 5: Write slope files in ParFlow ASCII format
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
    np.savetxt(f"{runname}.Subbasin_Summary.txt", subbasin["summary"], fmt="%.8g")

    print("Additional outputs written (direction, DEM, area, subbasins, streams, summary).")


if __name__ == "__main__":
    main()

