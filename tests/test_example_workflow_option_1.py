# Note: This test throws an error at the streamorder step:
# IndexError: index 0 is out of bounds for axis 0 with size 0
# The R workflow throws the same error.

import os
import numpy as np
from conftest import CORRECT_OUTPUT_DIR

CORRECT_OUTPUT_DIR = CORRECT_OUTPUT_DIR / "test_example_workflow_option_1"

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

def test_example_workflow_option_1():
    DEM = load_dem()
    watershed_mask = load_watershed_mask()
    river_mask = load_river_mask()

    nx, ny = DEM.shape

    init = init_queue(DEM)
    for key in ['mask', 'marked', 'basins']:
        R_file = os.path.join(CORRECT_OUTPUT_DIR, f"init_{key}.txt")
        R_data = np.loadtxt(R_file)
        python_data = init[key]
        assert np.array_equal(python_data, R_data), f"init_{key}.txt is not equal to the correct output"
    R_file = os.path.join(CORRECT_OUTPUT_DIR, "init_queue.txt")
    R_data = np.loadtxt(R_file)
    for row in R_data:
        row[0] -= 1
        row[1] -= 1
    python_data = init['queue']
    assert np.array_equal(python_data, R_data), "init_queue.txt is not equal to the correct output"
    with open(os.path.join(CORRECT_OUTPUT_DIR, "init_direction.txt")) as f:
        content = f.read().replace("NA", "nan")
    R_data = np.loadtxt(content.splitlines(), delimiter=" ")
    python_data = init["direction"]
    assert np.array_equal(python_data, R_data, equal_nan=True), "init_direction.txt is not equal to the correct output"

    trav_hs = d4_traverse_b(
        DEM,
        init["queue"].copy(),
        init["marked"].copy(),
        basins=init["basins"].copy(),
        epsilon=0,
        n_chunk=10,
    )

    for key in trav_hs.keys():
        if key == "direction":
            continue
        R_file = os.path.join(CORRECT_OUTPUT_DIR, f"travHS_{key}.txt")
        R_data = np.loadtxt(R_file)
        python_data = trav_hs[key]
        assert np.array_equal(python_data, R_data)

    with open(os.path.join(CORRECT_OUTPUT_DIR, "travHS_direction.txt")) as f:
        content = f.read().replace("NA", "nan")
    r_data = np.loadtxt(content.splitlines(), delimiter=" ")
    python_data = trav_hs["direction"]
    assert np.array_equal(python_data, r_data, equal_nan=True)

    area = drainage_area(
        trav_hs["direction"].copy(),
        mask=watershed_mask,
        printflag=False,
    )
    R_file = os.path.join(CORRECT_OUTPUT_DIR, "drainage_area.txt")
    R_data = np.loadtxt(R_file)
    assert np.array_equal(area, R_data)

    subbasin = calc_subbasins(
        trav_hs["direction"].copy(),
        area=area.copy(),
        mask=watershed_mask.copy(),
        riv_th=60,
        merge_th=0,
    )
    for key in subbasin.keys():
        if key == "summary":
            continue
        R_file = os.path.join(CORRECT_OUTPUT_DIR, f"subbasin_{key}.txt")
        R_data = np.loadtxt(R_file)
        python_data = subbasin[key]
        assert np.array_equal(python_data, R_data)
    r_data = np.loadtxt(os.path.join(CORRECT_OUTPUT_DIR, "subbasin_summary.txt"))
    for row in r_data:
        for j in range(1, 5):
            row[j] -= 1
    python_data = subbasin["summary"]
    assert np.array_equal(python_data, r_data)

    stream_order = calc_stream_order(
        subbasin["summary"][:, 0],
        subbasin["summary"][:, 5],
        subbasin["segments"].copy(),
    )
    for key in stream_order.keys():
        R_file = os.path.join(CORRECT_OUTPUT_DIR, f"streamorder_{key}.txt")
        R_data = np.loadtxt(R_file)
        python_data = stream_order[key]
        assert np.array_equal(python_data, R_data)

    riv_smooth_result = river_smooth(
        dem=trav_hs["dem"].copy(),
        direction=trav_hs["direction"].copy(),
        mask=watershed_mask.copy(),
        river_summary=subbasin["summary"].copy(),
        river_segments=subbasin["segments"].copy(),
        bank_epsilon=1,
    )
    key = "dem.adj"
    R_file = os.path.join(CORRECT_OUTPUT_DIR, f"RivSmooth_{key}.txt")
    R_data = np.loadtxt(R_file)
    python_data = riv_smooth_result[key]
    assert np.allclose(python_data, R_data)

    key = "processed"
    R_file = os.path.join(CORRECT_OUTPUT_DIR, f"RivSmooth_{key}.txt")
    R_data = np.loadtxt(R_file)
    python_data = riv_smooth_result[key]
    assert np.array_equal(python_data, R_data)

    key = "summary"
    R_file = os.path.join(CORRECT_OUTPUT_DIR, f"RivSmooth_{key}.txt")
    R_data = np.loadtxt(R_file)
    for row in R_data:
        for j in range(1, 5):
            row[j] -= 1
    python_data = riv_smooth_result[key]
    assert np.allclose(python_data, R_data, atol=1e-6)

    segment = 29
    start = np.array(
        [[subbasin["summary"][segment, 1], subbasin["summary"][segment, 2]]]
    )
    streamline_old = path_extract(
        trav_hs["dem"].copy(),
        trav_hs["direction"].copy(),
        mask=watershed_mask.copy(),
        startpoint=start,
    )
    for key in streamline_old.keys():
        if key == "path_list":
            continue
        R_file = os.path.join(CORRECT_OUTPUT_DIR, f"streamline_old_{key}.txt")
        R_data = np.loadtxt(R_file)
        python_data = streamline_old[key]
        assert np.allclose(python_data, R_data)
    R_file = os.path.join(CORRECT_OUTPUT_DIR, "streamline_old_path_list.txt")
    R_data = np.loadtxt(R_file)
    for row in R_data:
        row[0] -= 1
        row[1] -= 1
    python_data = streamline_old["path_list"]
    assert np.array_equal(python_data, R_data)

    streamline_new = path_extract(
        riv_smooth_result["dem.adj"].copy(),
        trav_hs["direction"].copy(),
        mask=watershed_mask.copy(),
        startpoint=start,
    )
    for key in streamline_new.keys():
        if key == "path_list":
            continue
        R_file = os.path.join(CORRECT_OUTPUT_DIR, f"streamline_new_{key}.txt")
        R_data = np.loadtxt(R_file)
        python_data = streamline_new[key]
        assert np.allclose(python_data, R_data)
    R_file = os.path.join(CORRECT_OUTPUT_DIR, "streamline_new_path_list.txt")
    R_data = np.loadtxt(R_file)
    for row in R_data:
        row[0] -= 1
        row[1] -= 1
    python_data = streamline_new["path_list"]
    assert np.array_equal(python_data, R_data)

    streamline_riv = path_extract(
        subbasin["segments"].copy(),
        trav_hs["direction"].copy(),
        mask=watershed_mask.copy(),
        startpoint=start,
    )
    for key in streamline_riv.keys():
        if key == "path_list":
            continue
        R_file = os.path.join(CORRECT_OUTPUT_DIR, f"streamline_riv_{key}.txt")
        R_data = np.loadtxt(R_file)
        python_data = streamline_riv[key]
        assert np.allclose(python_data, R_data)
    R_file = os.path.join(CORRECT_OUTPUT_DIR, "streamline_riv_path_list.txt")
    R_data = np.loadtxt(R_file)
    for row in R_data:
        row[0] -= 1
        row[1] -= 1
    python_data = streamline_riv["path_list"]
    assert np.array_equal(python_data, R_data)

    slopes_calc = slope_calc_standard(
        dem=riv_smooth_result["dem.adj"].copy(),
        direction=trav_hs["direction"].copy(),
        mask=watershed_mask.copy(),
        minslope=1e-5,
        maxslope=1,
        dx=1000,
        dy=1000,
        secondary_th=-1,
    )
    for key in slopes_calc.keys():
        if key == "direction":
            continue
        R_file = os.path.join(CORRECT_OUTPUT_DIR, f"slopesCalc_{key}.txt")
        R_data = np.loadtxt(R_file)
        python_data = slopes_calc[key]
        assert np.allclose(python_data, R_data)
    with open(os.path.join(CORRECT_OUTPUT_DIR, "slopesCalc_direction.txt")) as f:
        content = f.read().replace("NA", "nan")
    R_data = np.loadtxt(content.splitlines(), delimiter=" ")
    python_data = slopes_calc["direction"]
    assert np.array_equal(python_data, R_data, equal_nan=True)

    slopes_calc2 = riv_slope(
        direction=trav_hs["direction"].copy(),
        slopex=slopes_calc["slopex"].copy(),
        slopey=slopes_calc["slopey"].copy(),
        minslope=1e-4,
        river_mask=watershed_mask.copy(),
        remove_sec=True,
    )
    for key in slopes_calc2.keys():
        R_file = os.path.join(CORRECT_OUTPUT_DIR, f"slopesCalc2_{key}.txt")
        R_data = np.loadtxt(R_file)
        python_data = slopes_calc2[key]
        if key == "slopex" or key == "slopey":
            assert np.allclose(python_data, R_data)
        else:
            assert np.array_equal(python_data, R_data)