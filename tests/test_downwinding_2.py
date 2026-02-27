import os
import numpy as np
from conftest import CORRECT_OUTPUT_DIR

CORRECT_OUTPUT_DIR = CORRECT_OUTPUT_DIR / "downwinding_2"

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


def test_downwinding_2():
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

    init = init_queue(DEM, domainmask=watershed_mask)
    for key in init.keys():
        if key == 'direction':
            continue
        R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_2_init_{key}.txt'
        R_data = np.loadtxt(R_file)
        python_data = init[key]
        print(key, np.array_equal(python_data, R_data))
    R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_2_init_queue.txt'
    r_queue = np.loadtxt(R_file)
    for row in r_queue:
        row[0] -= 1
        row[1] -= 1
    assert np.array_equal(r_queue, init['queue'])
    with open(os.path.join(CORRECT_OUTPUT_DIR, "downwinding_2_init_direction.txt")) as f:
        content = f.read().replace("NA", "nan")
    R_data = np.loadtxt(content.splitlines(), delimiter=" ")
    python_data = init["direction"]
    assert np.array_equal(python_data, R_data, equal_nan=True)
    
    trav_hs = d4_traverse_b(
        DEM,
        init["queue"].copy(),
        init["marked"].copy(),
        basins=init["basins"].copy(),
        epsilon=ep,
        mask=watershed_mask,
    )
    for key in trav_hs.keys():
        if key == 'direction' or key == 'dem':
            continue
        R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_2_travHS_{key}.txt'
        R_data = np.loadtxt(R_file)
        python_data = trav_hs[key]
        assert np.array_equal(python_data, R_data)
    R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_2_travHS_dem.txt'
    R_data = np.loadtxt(R_file)
    python_data = trav_hs['dem']
    assert np.allclose(python_data, R_data)
    with open(os.path.join(CORRECT_OUTPUT_DIR, "downwinding_2_travHS_direction.txt")) as f:
        content = f.read().replace("NA", "nan")
    R_data = np.loadtxt(content.splitlines(), delimiter=" ")
    python_data = trav_hs['direction']
    assert np.array_equal(python_data, R_data, equal_nan=True)

    # Option 1
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
    for key in slopes_uw.keys():
        if key == 'direction':
            continue
        R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_2_slopesUW_option1_{key}.txt'
        R_data = np.loadtxt(R_file)
        python_data = slopes_uw[key]
        assert np.allclose(python_data, R_data)

    with open(os.path.join(CORRECT_OUTPUT_DIR, "downwinding_2_slopesUW_option1_direction.txt")) as f:
        content = f.read().replace("NA", "nan")
    R_data = np.loadtxt(content.splitlines(), delimiter=" ")
    python_data = slopes_uw['direction']
    assert np.array_equal(python_data, R_data, equal_nan=True)

    # Option 2
    area = drainage_area(trav_hs["direction"], printflag=False)
    R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_2_area_from_travHS_direction.txt'
    R_data = np.loadtxt(R_file)
    python_data = area
    assert np.array_equal(python_data, R_data)
    
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
        R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_2_subbasin_{key}.txt'
        R_data = np.loadtxt(R_file)
        python_data = subbasin[key]
        assert np.array_equal(python_data, R_data)
    R_file = f'{CORRECT_OUTPUT_DIR}/downwinding_2_subbasin_summary.txt'
    R_data = np.loadtxt(R_file)
    for row in R_data:
        for j in range(1, 5):
            row[j] -= 1
    python_data = subbasin['summary']
    assert np.array_equal(python_data, R_data)
    