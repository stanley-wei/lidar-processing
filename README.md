# las-processing (title TBD)
Using Python to process/filter 3D LiDAR point clouds

## Usage
All functionality is currently contained in ```src/interpolate_surface.py```. Given a *classified* (i.e. split into ground, trees, buildings, etc.) .las/.laz LiDAR point cloud, ```src/interpolate_surface.py``` will output a mesh. Output can be modified with CLI options; notably, a boolean image mask may be applied to the final mesh.

*Example:*
<br><code>python src/interpolate_surface.py --file Input.las --generate-mask --mask Input_Mask.png --output Output.stl</code>
