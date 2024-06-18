# lidar-processing
Using Python to process/filter/interpolate LiDAR point clouds. Uses classified LiDAR `.las`/`.laz` files to generate 3D meshes.


## Setup
 1. Clone this repository: `git clone https://github.com/stanley-wei/lidar-processing.git`
 2. Install the required packages (*requires Python3*): `cd lidar-processing && pip install -r lidar-processing/requirements.txt`


## Usage
The script `lidar-processing/interpolate_and_mesh.py` takes as input a *classified* (i.e. split into buildings, ground, etc.) LiDAR `.las`/`.laz` file, and will output a 3D mesh. Supports modification of mesh output via image masking.

(*LiDAR classification codes as specified [here](https://desktop.arcgis.com/en/arcmap/latest/manage-data/las-dataset/lidar-point-classification.htm)*)

[*Note: This repository does not currently support classification. Many GIS applications support classification of LiDAR data; I personally used [CloudCompare](https://www.cloudcompare.org/) and [LASTools](https://rapidlasso.de/product-overview/) for interfacing with LiDAR files.*]

**Example:**
<br><code>python lidar-processing/interpolate_and_mesh.py input.las output.stl --mask my_mask.png</code>

**General Options**:

 - `-r`/`--resolution [RESOLUTION]`: Specifies the resolution (meters/cell length) used when discretizing the point cloud. [*Default: 4*]
	 - *Note: Low resolutions (<1.0, e.g.) may result in odd-looking meshes.*
 - `-b`/`--base [BASE]`: Specifies the base height (i.e. height of the lowest point) in the output mesh. Can be either positive or negative. [*Default: 0*]
 - `--disable-discretize`: By default, the program uses a discretized grid representation of the LiDAR point cloud (rather than the raw point cloud itself) during meshing. This option causes the program to use the raw point cloud instead.
	 - *Note: May result in messier or otherwise less friendly meshes.*
 - `--include-unclassified`: By default, the program will throw out all points not labeled as being either ground or building. This option causes the program to keep all points not labeled as "tree". (To keep tree points, use the `--tree-mask` option.)

**Masking Options**:
- `--mask [MASK_NAME]`: This option takes a Boolean image mask [file name] that is applied to the output mesh. Points [pixels] with value `>0` will be retained; points/pixels with value `=0` will be ignored.
- `--generate-mask`: If `--mask` is enabled, this option will cause the program to create an image representation of the point cloud and prompt the user for a Boolean mask before continuining
- `--tree-mask [MASK_NAME]`: By default, the program will exclude all points classified as "tree" from the output mesh. This option allows the user to use an image mask to manually specify which tree-classified points are kept.
- `--generate-tree-mask`: Similar to `--generate-mask`

(*See `python lidar-processing/interpolate_and_mesh.py -h` for more details*)
