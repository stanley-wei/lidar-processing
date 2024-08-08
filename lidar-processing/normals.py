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
	for i in range(point_cloud.shape[0]):
		index = math.ceil((point_cloud[i][0] - min_bounds[0]) / resolution) \
				+ math.ceil((point_cloud[i][1] - min_bounds[1]) / resolution) * num_x_cells \
				+ math.ceil((point_cloud[i][1] - min_bounds[1]) / resolution) * num_x_cells * num_y_cells
		point_index_pairs[i][1] = index

	sorted_point_indices = point_index_pairs[point_index_pairs[:, 1].argsort()]

	downsampled_cloud = np.zeros((point_cloud.shape[0], 3), dtype=point_cloud.dtype)

	idx = 0
	num_added_points = 0
	while idx < point_index_pairs.shape[0]:
		start_idx = idx

		idx += 1
		while (idx < point_index_pairs.shape[0] and 
			   point_index_pairs[idx, 1] == point_index_pairs[idx-1, 1]):
			idx += 1

		downsampled_cloud[num_added_points, :] = np.mean(point_cloud[start_idx:idx], axis=0)
		num_added_points += 1

	return downsampled_cloud[:num_added_points]


@timebudget
def classify_points(point_cloud, search_area, planarity):
	points_kdtree = spatial.KDTree(data=point_cloud)

	query = points_kdtree.query_ball_tree(points_kdtree, r=search_area)

	evrs = np.ones((point_cloud.shape[0]), dtype=float)
	pca = PCA(n_components=3)
	for i in range(point_cloud.shape[0]):
		neighbors = point_cloud[query[i]]
		if neighbors.shape[0] > 3:
			pca.fit(neighbors)
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

if __name__ == "__main__":
	las_data = laspy.open(sys.argv[1]).read()
	points = np.asarray(las_data.xyz)

	new_points = voxel_filter(points, resolution=float(5))
	cleaned_points = remove_outliers(new_points)
	classifications = classify_points(cleaned_points, search_area=6, planarity=0.15)

	# Write to LiDAR file
	classified_header = laspy.LasHeader(version="1.4", point_format=6)
	classified_las = laspy.LasData(classified_header)
	classified_las.xyz = cleaned_points
	classified_las.classification = classifications
	classified_las.write("classified.laz")
