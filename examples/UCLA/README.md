For the larger UCLA campus: LiDAR point clouds (full & subsampled), image masks obtainable from `https://www.dropbox.com/scl/fo/y559ocqydvgt1oc3o82g1/h?rlkey=cjrhadl3v68xmrch2q88hwn0m&st=31mgkdjp&dl=0`
 - All masks created using 5-meter resolution

To run: `python lidar-processing/interpolate_and_mesh.py ucla.laz output.stl -r 5.0 --mask ucla_mask.png -b -550.0`

(The last parameter, `-b`/`--base`, may be varied based on personal preference.)
