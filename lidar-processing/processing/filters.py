import math
import numpy as np
import pyvista as pv
from scipy import spatial, stats
from timebudget import timebudget

from ..data import PointCloud


class VoxelFilter:
	def __init__(self, resolution=5.0):
		self.resolution = resolution

	def filter(self, point_cloud):
		min_bounds = np.min(point_cloud.point_cloud, axis=0)
		max_bounds = np.max(point_cloud.point_cloud, axis=0)

		num_x_cells = math.ceil((max_bounds[0] - min_bounds[0]) / self.resolution)
		num_y_cells = math.ceil((max_bounds[1] - min_bounds[1]) / self.resolution)

		point_index_pairs = np.column_stack((np.arange(0, point_cloud.point_cloud.shape[0]), 
				np.zeros((point_cloud.shape[0]), dtype=int)))

		point_index_pairs[:, 1] = np.ceil((point_cloud.point_cloud[:, 0] - min_bounds[0]) / self.resolution) \
			   + np.ceil((point_cloud.point_cloud[:, 1] - min_bounds[1]) / self.resolution) * num_x_cells \
			   + np.ceil((point_cloud.point_cloud[:, 1] - min_bounds[1]) / self.resolution) * num_x_cells * num_y_cells

		values, counts = np.unique(point_index_pairs[:, 1], return_counts=True)

		sorted_point_indices = point_index_pairs[point_index_pairs[:, 1].argsort(), 0]

		downsampled_cloud = np.zeros((values.shape[0], 3), dtype=point_cloud.point_cloud.dtype)
		if point_cloud.classification is not None:
			downsampled_classes = np.zeros((values.shape[0]), dtype=int)

		value_indices = np.concatenate((np.zeros(1, dtype=int), np.cumsum(counts)))
		for i in range(0, len(values)):
			downsampled_cloud[i] = np.mean(point_cloud.point_cloud[sorted_point_indices[value_indices[i]:value_indices[i+1]]], axis=0)
			if downsampled_classes is not None:
				downsampled_classes[i] = stats.mode(point_cloud.classification[sorted_point_indices[value_indices[i]:value_indices[i+1]]]).mode

		return PointCloud(downsampled_cloud, downsampled_classes)


class VerticalOutlierFilter:

	def __init__(self, num_stdev=2.0):
		self.num_stdev = num_stdev

	def filter(point_cloud):
		'''
		Removes any outlying points that are [num_stdev] standard deviations above
		the mean. Mean/standard deviation passed as function parameters (if specific 
		values for mean/stdev are desired) or determined from point cloud.
		'''
		mean_z = np.mean(point_cloud[:, 2])
		std_z = np.std(point_cloud[:, 2])

		indices_mask = np.ones((point_cloud.shape[0]), dtype=bool)
		for i in range(point_cloud.shape[0]):
			if abs(point_cloud[i][2] - mean_z) > self.num_stdev * std_z:
				indices_mask[i] = False

		return point_cloud[indices_mask]


class StatisticalOutlierFilter:
	def __init__(self, num_neighbors=20, num_stdev=2.0):
		self.num_neighbors = num_neighbors
		self.num_stdev = num_stdev

	def filter(self, point_cloud):
		points_kdtree = spatial.KDTree(data=point_cloud.point_cloud)
		query = points_kdtree.query(point_cloud.point_cloud, k=self.num_neighbors)

		mean_neighbor_distance = np.mean(np.asarray(query[0]), axis=1)
		stdev = np.std(mean_neighbor_distance)

		threshold = np.mean(mean_neighbor_distance) + self.num_stdev * stdev
		mask = np.nonzero(mean_neighbor_distance < threshold)[0]

		if point_cloud.classification is not None:
			return PointCloud(point_cloud.point_cloud[mask, :], point_cloud.classification[mask])
		else:
			return PointCloud(point_cloud.point_cloud[mask, :])


class RadiusOutlierFilter:
	def __init__(self, num_neighbors=5, radius=10.0):
		self.num_neighbors = num_neighbors
		self.radius = radius

	def filter(self, point_cloud):
		points_kdtree = spatial.KDTree(data=point_cloud)

		query = points_kdtree.query_ball_tree(points_kdtree, r=self.radius)

		indices = np.asarray(list(map(len, query)), dtype=int) >= self.num_neighbors

		return point_cloud[indices, :]
