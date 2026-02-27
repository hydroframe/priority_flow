import os
import numpy as np
from conftest import CORRECT_OUTPUT_DIR

CORRECT_OUTPUT_DIR = CORRECT_OUTPUT_DIR / "downwinding_4"

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

def test_downwinding_4():
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

    init = init_queue(
        DEM,
        initmask=river_mask,
        domainmask=watershed_mask,
    )

    for key in init.keys():
        if key == 'direction' or key == 'queue':
            continue
        R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_4_init_{key}.txt'
        R_data = np.loadtxt(R_file)
        python_data = init[key]
        assert np.array_equal(python_data, R_data)
    R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_4_init_queue.txt'
    r_queue = np.loadtxt(R_file)
    for row in r_queue:
        row[0] -= 1
        row[1] -= 1
    assert np.array_equal(r_queue, init['queue'])
    with open(f'{CORRECT_OUTPUT_DIR}/downwinding_4_init_direction.txt') as f:
        content = f.read().replace("NA", "nan")
    R_data = np.loadtxt(content.splitlines(), delimiter=" ")
    python_data = init['direction']
    assert np.array_equal(python_data, R_data, equal_nan=True)

    trav1 = stream_traverse(
        dem=DEM,
        mask=river_mask,
        queue=init["queue"].copy(),
        marked=init["marked"].copy(),
        basins=init["basins"].copy(),
        printstep=False,
        epsilon=ep,
    )
    for key in trav1.keys():
        if key == 'dem' or key == 'direction':
            continue
        R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_4_trav1_{key}.txt'
        R_data = np.loadtxt(R_file)
        python_data = trav1[key]
        assert np.array_equal(python_data, R_data)
    R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_4_trav1_dem.txt'
    R_data = np.loadtxt(R_file)
    python_data = trav1['dem']
    assert np.allclose(python_data, R_data)

    first_pass_pct = (
        np.sum(trav1["marked"] * river_mask) / np.sum(river_mask) * 100.0
        if np.sum(river_mask) > 0
        else 0.0
    )
    print(f"First Pass: {first_pass_pct:.1f} % cells processed")

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

    border_t = get_border(watershed_mask)
    riv_border = border_t + trav1["marked"]
    riv_border[riv_border > 1] = 1
    R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_4_borderT.txt'
    R_data = np.loadtxt(R_file)
    python_data = border_t
    assert np.array_equal(python_data, R_data)

    riv_border = border_t + trav1["marked"]
    riv_border[riv_border > 1] = 1
    init2 = init_queue(trav1["dem"], border=riv_border)

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
    for key in trav_hs.keys():
        if key == 'direction' or key == 'dem':
            continue
        R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_4_travHS_{key}.txt'
        R_data = np.loadtxt(R_file)
        python_data = trav_hs[key]
        assert np.array_equal(python_data, R_data)
    R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_4_travHS_dem.txt'
    R_data = np.loadtxt(R_file)
    python_data = trav_hs['dem']
    assert np.allclose(python_data, R_data)
    with open(f'{CORRECT_OUTPUT_DIR}/downwinding_4_travHS_direction.txt') as f:
        content = f.read().replace("NA", "nan")

    R_data = np.loadtxt(content.splitlines(), delimiter=" ")
    python_data = trav_hs['direction']
    assert np.array_equal(python_data, R_data, equal_nan=True)

    # Option 1
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
    for key in slopes_uw.keys():
        if key == 'direction':
            continue
        R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_4_slopesUW_option1_{key}.txt'
        R_data = np.loadtxt(R_file)
        python_data = slopes_uw[key]
        assert np.allclose(python_data, R_data)
    with open('/home/ga6/downwinding_4_slopesUW_option1_direction.txt') as f:
        content = f.read().replace("NA", "nan")

    R_data = np.loadtxt(content.splitlines(), delimiter=" ")
    python_data = slopes_uw['direction']
    assert np.array_equal(python_data, R_data, equal_nan=True)

    # Option 2
    area = drainage_area(
        trav_hs["direction"].copy(),
        printflag=False,
    )
    R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_4_area_from_travHS_direction.txt'
    R_data = np.loadtxt(R_file)
    python_data = area
    assert np.array_equal(python_data, R_data, equal_nan=True)

    subbasin = calc_subbasins(
        trav_hs["direction"].copy(),
        area=area,
        mask=watershed_mask.copy(),
        riv_th=sub_th,
        merge_th=mrg_th,
    )
    for key in subbasin.keys():
        if key == 'summary':
            continue
        R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_4_subbasin_{key}.txt'
        R_data = np.loadtxt(R_file)
        python_data = subbasin[key]
        assert np.array_equal(python_data, R_data)
    R_data = np.loadtxt('/home/ga6/downwinding_4_subbasin_summary.txt')
    for row in R_data:
        for j in range(1, 5):
            row[j] -= 1
    assert np.array_equal(R_data, subbasin['summary'])

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
    for key in slopes_uw.keys():
        if key == 'direction':
            continue
        R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_4_slopesUW_option2_{key}.txt'
        R_data = np.loadtxt(R_file)
        python_data = slopes_uw[key]
        assert np.allclose(python_data, R_data)
    with open('/home/ga6/downwinding_4_slopesUW_option2_direction.txt') as f:
        content = f.read().replace("NA", "nan")
    R_data = np.loadtxt(content.splitlines(), delimiter=" ")
    python_data = slopes_uw['direction']
    assert np.array_equal(python_data, R_data, equal_nan=True)

    # Option 2b
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
    for key in slopes_uw.keys():
        if key == 'direction':
            continue
        R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_4_slopesUW_option2b_{key}.txt'
        R_data = np.loadtxt(R_file)
        python_data = slopes_uw[key]
        assert np.allclose(python_data, R_data)
    with open(f'{CORRECT_OUTPUT_DIR}/downwinding_4_slopesUW_option2b_direction.txt') as f:
        content = f.read().replace("NA", "nan")

    R_data = np.loadtxt(content.splitlines(), delimiter=" ")
    python_data = slopes_uw['direction']
    assert np.array_equal(python_data, R_data, equal_nan=True)

    area = drainage_area(
        slopes_uw["direction"],
        mask=watershed_mask,
        printflag=False,
    )
    R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_4_area_from_final_direction.txt'
    R_data = np.loadtxt(R_file)
    python_data = area
    assert np.array_equal(python_data, R_data, equal_nan=True)