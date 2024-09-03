import math
import numpy as np
import pyvista as pv
from scipy import spatial
from timebudget import timebudget


@timebudget
def voxel_filter(point_cloud, resolution=5.0):
	min_bounds = np.min(point_cloud, axis=0)
	max_bounds = np.max(point_cloud, axis=0)

	num_x_cells = math.ceil((max_bounds[0] - min_bounds[0]) / resolution)
	num_y_cells = math.ceil((max_bounds[1] - min_bounds[1]) / resolution)

	point_index_pairs = np.column_stack((np.arange(0, point_cloud.shape[0]), 
			np.zeros((point_cloud.shape[0]), dtype=int)))

	point_index_pairs[:, 1] = np.ceil((point_cloud[:, 0] - min_bounds[0]) / resolution) \
		   + np.ceil((point_cloud[:, 1] - min_bounds[1]) / resolution) * num_x_cells \
		   + np.ceil((point_cloud[:, 1] - min_bounds[1]) / resolution) * num_x_cells * num_y_cells

	values, counts = np.unique(point_index_pairs[:, 1], return_counts=True)

	sorted_point_indices = point_index_pairs[point_index_pairs[:, 1].argsort(), 0]

	downsampled_cloud = np.zeros((values.shape[0], 3), dtype=point_cloud.dtype)

	value_indices = np.concatenate((np.zeros(1, dtype=int), np.cumsum(counts)))
	for i in range(0, len(values)):
		downsampled_cloud[i] = np.mean(point_cloud[sorted_point_indices[value_indices[i]:value_indices[i+1]]], axis=0)

	return downsampled_cloud


@timebudget
def remove_vertical_outliers(point_cloud, num_stdev = 2.0):
	'''
	Removes any outlying points that are [num_stdev] standard deviations above
	the mean. Mean/standard deviation passed as function parameters (if specific 
	values for mean/stdev are desired) or determined from point cloud.
	'''
	mean_z = np.mean(point_cloud[:, 2])
	std_z = np.std(point_cloud[:, 2])

	indices_mask = np.ones((point_cloud.shape[0]), dtype=bool)
	for i in range(point_cloud.shape[0]):
		if abs(point_cloud[i][2] - mean_z) > num_stdev * std_z:
			indices_mask[i] = False

	return point_cloud[indices_mask]


@timebudget
def remove_statistical_outliers(point_cloud, num_neighbors=20, num_stdev=2.0):
	points_kdtree = spatial.KDTree(data=point_cloud)

	query = points_kdtree.query(point_cloud, k=num_neighbors)

	mean_neighbor_distance = np.mean(np.asarray(query[0]), axis=1)

	stdev = np.std(mean_neighbor_distance)

	threshold = np.mean(mean_neighbor_distance) + num_stdev * stdev

	return point_cloud[mean_neighbor_distance < threshold, :]


@timebudget
def remove_radius_outliers(point_cloud, num_neighbors=5, radius=10.0):
	points_kdtree = spatial.KDTree(data=point_cloud)

	query = points_kdtree.query_ball_tree(points_kdtree, r=radius)

	indices = np.asarray(list(map(len, query)), dtype=int) >= num_neighbors

	return point_cloud[indices, :]
