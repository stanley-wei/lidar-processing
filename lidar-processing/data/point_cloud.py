import cv2
import math
import numpy as np
import pyvista as pv
from scipy import interpolate, ndimage
import time
from tqdm import tqdm

from . import utils


class PointCloud:

	def __init__(self, point_cloud, classification=None, bounds=None):
		self.point_cloud = point_cloud
		self.classification = classification
		self._bounds = bounds # Stored as a dictionary; see bounds() function


	def __getitem__(self, index):
		return self.point_cloud[index]


	@property
	def shape(self):
		return self.point_cloud.shape
	

	@property
	def bounds(self):
		if not self._bounds:
			minimums = np.min(self.point_cloud, axis = 0)
			maximums = np.max(self.point_cloud, axis = 0)

			self._bounds = {"max_x": maximums[0], "max_y": maximums[1], "max_z": maximums[2],
				"min_x": minimums[0], "min_y": minimums[1], "min_z": minimums[2]}

		return self._bounds			


	def combine(cloud_1, cloud_2):
		'''
		Combines two PointCloud objects into a new PointCloud containing
		points from both previous PointClouds, updating bounds to match.
		'''
		if not cloud_1:
			return cloud_2
		elif not cloud_2:
			return cloud_1

		return PointCloud(np.concatenate((cloud_1.point_cloud, cloud_2.point_cloud), axis=0),
			bounds=utils.combine_bounds(cloud_1.bounds, cloud_2.bounds))


	def remove_outliers(self, z_mean = None, z_std = None, num_stdev = 2):
		'''
		Removes any outlying points that are [num_stdev] standard deviations above
		the mean. Mean/standard deviation passed as function parameters (if specific 
		values for mean/stdev are desired) or determined from point cloud.
		'''
		mean_z = np.mean(self.point_cloud, axis=0)[2] if z_mean is None else z_mean
		std_z = np.std(self.point_cloud, axis=0)[2] if z_std is None else z_std

		to_delete = np.where(abs(self.point_cloud[2] - mean_z) > num_stdev * std_z)
		self.point_cloud = np.delete(self.point_cloud, to_delete, axis = 0)

		return mean_z, std_z


	def apply_mask(self, mask_image, resolution, start_index=0):
		'''
		Given a mask image, outputs list of all points marked as
		excluded in the mask.
		'''
		excluded_points = []
		for point_i in tqdm(range(self.point_cloud.shape[0])):
			grid_y, grid_x = utils.point_cloud_coord_to_grid(self.point_cloud[point_i][0], 
				self.point_cloud[point_i][1], resolution, bounds=self.bounds)

			if not grid_y >= mask_image.shape[0] and not grid_x >= mask_image.shape[1] and mask_image[grid_y][grid_x] == 0:
				excluded_points.append(point_i + start_index)

		return excluded_points


	def write_to_file(self, file_name, edit_type_ = 'w'):
		file_name = utils.truncate_file_name(file_name)

		file = open(file_name, edit_type_);
		for i in range(self.shape[0]):
			file.write("%f %f %f\n" % (self.point_cloud[i][0], self.point_cloud[i][1], self.point_cloud[i][2]))
	    
		file.close();
		return file_name;


	def generate_mesh(self, excluded_points = [], base_height = 0, bounds = []):
		'''
		Uses PyVista library to generate a 3D mesh from PointCloud object,
		using Delaunay triangulation to create a surface and extruding to 
		a plane for volume. Height of extrusion determined by base_height parameter.
		'''
		if not bounds:
			bounds = self.bounds

		pv_cloud = pv.PolyData(self.point_cloud)
		surface = pv_cloud.delaunay_2d()

		if excluded_points:
			surface, indices = surface.remove_points(excluded_points)

		plane = pv.Plane(
			center = (surface.center[0], surface.center[1], pv_cloud.bounds[4] - base_height),
			direction = (0, 0, -1.0),
			i_size = bounds['max_x'] - bounds['min_x'],
			j_size = bounds['max_y'] - bounds['min_y'])

		extruded_mesh = surface.extrude_trim((0, 0, -1.0), plane)
		return extruded_mesh

