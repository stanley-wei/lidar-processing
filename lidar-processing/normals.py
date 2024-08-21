import laspy
import math
import numpy as np
import pyvista as pv
from scipy import spatial
from sklearn.decomposition import PCA
import sys
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
	for i in range(1, len(values)):
		downsampled_cloud[i] = np.mean(point_cloud[sorted_point_indices[value_indices[i-1]:value_indices[i]]], axis=0)

	return downsampled_cloud


def retrieve_point_neighborhoods(point_cloud, radius, k, type='spherical'):
	'''
	type: 'spherical', 'knn', 'cylindrical'
	'''

	if type not in ['spherical', 'knn', 'cylindrical']:
		raise TypeError(f"Parameter 'type' must be one of: ['spherical', 'knn', 'cylindrical'].")

	if type == 'spherical':
		points_kdtree = spatial.KDTree(data=point_cloud)
		query = points_kdtree.query_ball_tree(points_kdtree, r=radius)

		return query, None

	elif type == 'cylindrical':
		points_kdtree = spatial.KDTree(data=point_cloud[:, 0:2])
		query = points_kdtree.query_ball_tree(points_kdtree, r=radius)

		return query, None

	elif type == 'knn':
		points_kdtree = spatial.KDTree(data=point_cloud)
		query = points_kdtree.query(point_cloud, k=k)

		return query[1], query[0]

@timebudget
def classify_points(point_cloud, radius, k, planarity, type='spherical'):
	query = retrieve_point_neighborhoods(point_cloud, type, radius, k)

	evrs = np.ones((point_cloud.shape[0]), dtype=float)
	pca = PCA(n_components=3)
	for i in range(point_cloud.shape[0]):
		neighbors = point_cloud[query[i]]
		if neighbors.shape[0] > 3:
			pca.fit(neighbors)

			linearity = (pca.explained_variance_[1] - pca.explained_variance_[2]) / pca.explained_variance_[1]
			planarity = (pca.explained_variance_[2] - pca.explained_variance_[3]) / pca.explained_variance_[1]
			sphericity = pca.explained_variance_[3] / pca.explained_variance_[1]
			evrs[i] = pca.explained_variance_[2]

	evrs = evrs / np.max(evrs)

	return np.where(evrs < planarity, 2, 5)


@timebudget
def remove_outliers(point_cloud, num_stdev = 2.0):
	'''
	Removes any outlying points that are [num_stdev] standard deviations above
	the mean. Mean/standard deviation passed as function parameters (if specific 
	values for mean/stdev are desired) or determined from point cloud.
	'''
	mean_z = np.mean(point_cloud, axis=0)[2]
	std_z = np.std(point_cloud, axis=0)[2]

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

	indices = np.zeros(len(query), dtype=bool)
	for i in range(len(query)):
		indices[i] = True if len(query[i]) >= num_neighbors else False

	return point_cloud[indices, :]


def estimate_normals(point_cloud, num_neighbors=30):
	points_kdtree = spatial.KDTree(data=point_cloud)

	query = points_kdtree.query_ball_tree(points_kdtree, r=search_area)

	normals = np.zeros((point_cloud.shape[0], 3), dtype=float)
	pca = PCA(n_components=3)
	for i in range(point_cloud.shape[0]):
		neighbors = point_cloud[query[i]]
		if neighbors.shape[0] > 3:
			pca.fit(neighbors)

			normals[i] = pca.components_

	return normals

def color_by_classification(point_cloud, classifications):
	cloud = pv.PolyData(point_cloud)

	cloud['point_color'] = classifications

	pv.plot(cloud, scalars='point_color')


if __name__ == "__main__":
	las_data = laspy.open(sys.argv[1]).read()
	points = np.asarray(las_data.xyz)

	new_points = voxel_filter(points, resolution=float(2.0))
	cleaned_points = remove_statistical_outliers(new_points)
	classifications = classify_points(cleaned_points, search_area=6, planarity=0.15)

	# Write to LiDAR file
	classified_header = laspy.LasHeader(version="1.4", point_format=6)
	classified_las = laspy.LasData(classified_header)
	classified_las.xyz = cleaned_points
	classified_las.classification = classifications
	classified_las.write("classified.laz")
