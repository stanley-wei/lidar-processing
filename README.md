
# lidar-processing
Using Python to process, filter, and classify LiDAR point clouds. Also supports mesh generation for 3D modeling.


## Setup
 1. Clone this repository: `git clone https://github.com/stanley-wei/lidar-processing.git`
 2. Install the required packages (*requires Python3*): `cd lidar-processing && pip install -r lidar-processing/requirements.txt`


## Usage

*All LiDAR classification codes are as specified [here](https://desktop.arcgis.com/en/arcmap/latest/manage-data/las-dataset/lidar-point-classification.htm).*

### Ground & Feature Extraction
    python3 -m lidar-processing.classification.ground_extraction <LiDAR_FILE>
    python3 -m lidar-processing.classification.feature_extraction <LiDAR_PATH>

The script `lidar-processing/classification/ground_extraction` takes as input a LiDAR `.las`/`.laz` file and outputs a `.las`/`.laz` with accompanying `ground`/`non-ground` class annotations.

The script `lidar-processing/classification/feature_extraction` takes as input a LiDAR file (or directory of LiDAR files) **with classified ground points** and outputs `.csv` file(s) containing a set of extracted features for every point. (One `.csv` for each input file.)

(*Use `python3 -m lidar-processing.classification.ground_extraction --help` and `python3 -m lidar-processing.classification.feature_extraction --help` to view additional options.*)

### Classification
*Training*:

    python3 -m lidar-processing.classification.train <DATASET_PATH>

This takes as input a directory of **classified** LiDAR `.las`/`.laz` files and trains a model to classify point types. 

(Has classifier options; see `python3 -m lidar-processing.classification.train -h` for more details.)

*Testing*:

    python3 -m lidar-processing.classification.test <DATASET_PATH> <CLASSIFIER_PATH>

This takes as input: (1) a directory of **classified** LiDAR `.las`/`.laz` files and (2) a `joblib`-pickled classifier with a function `.predict()`, then evaluates the performance of the classifier over the dataset. 

### 3D Modeling
    python3 -m lidar-processing.scripts.interpolate_and_mesh <LiDAR_FILE>

The script `lidar-processing/scripts/interpolate_and_mesh.py` takes as input a *classified* (i.e. split into buildings, ground, etc.) LiDAR `.las`/`.laz` file, and will output a 3D mesh. Supports modification of mesh output via image masking.

[*Note: Many GIS applications support classification of LiDAR data; I personally used [CloudCompare](https://www.cloudcompare.org/) and [LASTools](https://rapidlasso.de/product-overview/) for interfacing with LiDAR files.*]

(*See `python -m lidar-processing.scripts.interpolate_and_mesh.py -h` for more details*)
