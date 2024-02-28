For the larger UCLA campus: LiDAR point clouds (full & subsampled), image masks obtainable from `https://www.dropbox.com/scl/fi/4wd7tvhdzx5gj2mlii2gd/ucla_mask.png?rlkey=l1vetwttdj6daycwt6p2k9ift&dl=0`
 - All masks created using 5-meter resolution

To run: `python lidar-processing/interpolate_and_mesh.py ucla.laz output.stl -r 5.0 --mask ucla_mask.png -b -550.0`

(The last parameter, `-b`/`--base`, may be varied based on personal preference.)
