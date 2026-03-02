# priority_flow

*priority_flow* is a toolkit for topographic processing for hydrologic models. This repo contains an python package and a set of workflow examples (see instructions below). This is the Python version of the R PriorityFlow package.

## Development Team
+ Laura Condon (lecondon@email.arizona.edu)
+ Reed Maxwell (reedmaxwell@princeton.edu)
+ Georgios Artavanis (ga6@princeton.edu)

## Citation
For more details on the model and if you use priority_flow in published work please cite the following reference:  
   *Condon, LE and RM Maxwell (2019). Modified priority flood and global slope enforcement algorithm for topographic processing in physically based hydrologic modeling applications. Computers & Geosciences, [doi:10.1016/j.cageo.2019.01.20](https://doi.org/10.1016/j.cageo.2019.01.020).*

## Installation

```bash
$ pip install priority_flow
```

## Getting  started

The best way to get started with this toolset would be to walk through the `Workflow_Example_Notebook.ipynb` Jupyter notebook in the 'Notebooks' section of this documentation. In addition, you can walk through the four Downwinding workflows. You can also refer to the documentation for each individual function provided in this package.

## DEM Processing

The DEM processing code is a modified version of the 'Priority Flood' algorithm which is a depression filling algorithm (Wang and Liu 2006; Barnes et al., 2014; Zhou, 2016).  This is an optimal approach that processes the DEM to ensure every cell drains to the border.

As implemented here there are options to ensure drainage to the edges of a regular rectangular domain or the user can provide a mask for an irregular domain. NOTE: if you are providing a mask for an irregular domain boundary you must ensure that your mask is D4 contiguous. See the mask tip below for one approach to achieve this using grass.

Additionally, a second processing option is provided if there is an river network that you would like to enforce. In this case, the river network is provided first as a mask to the processing algorithm to ensure that every identified river cell drains to the boundary of the domain (regular or irregular). This step will also ensure a D4 connected river network (i.e. stair stepping around any diagonal river neighbors). Next the remaining cells are processed using the river network as the boundary, ensuring that every other cell drains either to a river or to a boundary. For examples of this approach refer to `Downwinding_Workflow_Example3.ipynb` and `Downwinding_Workflow_Example4.ipynb`.

Slope Calculations
--------------------
There are two slope calculations functions in this repo:

`slope_calc_upwind`  calculates slopes in the x and y direction down-winded to be consistent with the ParFlow OverlandFLow boundary condition.

`slope_calc_standard` calculates slopes in the x and y direction using indexing to be consistent with the ParFlow OverlandKinematic and OverlandDiffusive boundary conditions. This is the approach that is used in the main workflow example.

## Workflow Scripts
1. `Workflow_Example_Notebook.ipynb`: This is the most updated  workflow example and  the  one  I recommend starting from.

The next four examples show the older slope calculation function with downwinding
1. `Downwinding_Workflow_Example1.ipynb`: Rectangular domain with no river network
2. `Downwinding_Workflow_Example2.ipynb`: Irregular domain with no river network
3. `Downwinding_Workflow_Example3.ipynb`: Rectangular domain with river network
4. `Downwinding_Workflow_Example4.ipynb`: Irregular domain with river network

## Tips

If you want to process your DEM within a pre-defined watershed mask and you need help creating that mask. An example workflow using QGIS and GRASS to ensure a D4 connected mask (i.e. one where you don't have any cells that are only connected to the rest of the domain diagonally):
1. create the mask in QGIS
2. Using GRASS, clump it: r.clump
3. change no data value to 0: r.null
4. identify the ID of the big contiguous block (clump) you want to keep
5. re-mask:`("NullRaster@1"=65)*1+("NullRaster@1" != 65)*0`

## References

+ Barnes, R., C. Lehman, and D. Mulla, Priority-flood: An optimal depression-filling and watershed-labeling algorithm for digital elevation models. Computers & Geosciences, 2014. 62: p. 117-127.
+ Wang, L. and H. Liu, An efficient method for identifying and filling surface depressions in digital elevation models for hydrologic analysis and modelling. International Journal of Geographical Information Science, 2006. 20(2): p. 193-213.
+ Zhou, G., Z. Sun, and S. Fu, An efficient variant of the Priority-Flood algorithm for filling depressions in raster digital elevation models. Computers & Geosciences, 2016. 90: p. 87-96.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`priority_flow` was created by Laura E. Condon and Reed M. Maxwell. It is licensed under the terms of the GNU General Public License v3.0 license.

## Credits

`priority_flow` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
