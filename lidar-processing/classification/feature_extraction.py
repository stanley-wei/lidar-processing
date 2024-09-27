import argparse
import glob
from joblib import Parallel, delayed
import laspy
import math
import numpy as np
import os
import pyvista as pv
from scipy import interpolate, spatial
from sklearn.decomposition import PCA
import sys
from timebudget import timebudget
from tqdm import tqdm

from . import ground_extraction
from . import utils
from ..config import classes
from ..data import PointCloud
from ..processing import VoxelFilter, StatisticalOutlierFilter


def retrieve_point_neighborhoods(point_cloud, radius, k, type='spherical', 
								 start=None, end=None):
	'''
	type: 'spherical', 'knn', 'cylindrical'
	'''

	if type not in ['spherical', 'knn', 'cylindrical']:
		raise TypeError(f"Parameter 'type' must be one of: ['spherical', 'knn', 'cylindrical'].")

	if type == 'spherical':
		points_kdtree = spatial.KDTree(data=point_cloud)
		if not start or not end:
			query = points_kdtree.query_ball_tree(points_kdtree, r=radius)
		else:
			search_kdtree = spatial.KDTree(data=point_cloud[start:end])
			query = points_kdtree.query_ball_tree(search_kdtree, r=radius)
		return query, None

	elif type == 'cylindrical':
		points_kdtree = spatial.KDTree(data=point_cloud[:, 0:2])
		if not start or not end:
			query = points_kdtree.query_ball_tree(points_kdtree, r=radius)
		else:
			search_kdtree = spatial.KDTree(data=point_cloud[start:end, 0:2])
			query = points_kdtree.query_ball_tree(search_kdtree, r=radius)
		return query, None

	elif type == 'knn':
		points_kdtree = spatial.KDTree(data=point_cloud)
		if not start or not end:
			query = points_kdtree.query(point_cloud, k=k)
		else:
			query = points_kdtree.query(point_cloud[start:end], k=k)
		return query[1], query[0]


def extract_features(point_cloud, radius, k, type, start, end):
	query_points, query_distances = retrieve_point_neighborhoods(point_cloud, radius, k, type, 
																 start, end)

	num_features = 24
	features = np.zeros((end-start, num_features), dtype=float)

	pca = PCA(n_components=3)
	pca_2d = PCA(n_components=2)
	for i in range(start, end):
		neighbors = point_cloud[query_points[i-start]]

		if neighbors.shape[0] < 3:
			continue

		elevation = point_cloud[i, 2]

		num_points = neighbors.shape[0]
		neighborhood_radius = np.max(np.linalg.norm(point_cloud[i] - neighbors[:]))
		max_height_diff = np.max(neighbors[:, 2]) - np.min(neighbors[:, 2])
		height_std = np.std(neighbors[:, 2])
		neighborhood_density = (num_points + 1) / ((4.0/3.0) * np.pi * np.power(neighborhood_radius, 3))

		radius_2d = np.max(np.linalg.norm(point_cloud[i, 0:2] - neighbors[:, 0:2]))
		density_2d = (num_points + 1) / (np.pi * np.power(neighborhood_radius, 2))

		pca.fit(neighbors)

		sum_of_eigenvalues = np.sum(pca.explained_variance_)

		linearity = (pca.explained_variance_[0] - pca.explained_variance_[1]) / pca.explained_variance_[0]
		planarity = (pca.explained_variance_[1] - pca.explained_variance_[2]) / pca.explained_variance_[0]
		sphericity = pca.explained_variance_[2] / pca.explained_variance_[0]
		change_of_curvature = pca.explained_variance_[2] / (sum_of_eigenvalues)

		anisotropy = (pca.explained_variance_[0] - pca.explained_variance_[2]) / pca.explained_variance_[0]

		omnivariance = math.pow(np.prod(pca.explained_variance_), 1.0/3.0)
		if np.min(pca.explained_variance_ratio_) != 0.0:
			eigentropy = -np.sum(pca.explained_variance_ratio_ * np.log(pca.explained_variance_ratio_))
		else:
			eigentropy = np.finfo(np.float32).max

		verticality = np.arccos(np.dot(pca.components_[2], [0,0,1]) / np.linalg.norm(pca.components_[2]))

		first_moment = np.sum(np.inner(neighbors-point_cloud[i], pca.components_))/float(neighbors.shape[0])
		second_moment = np.sum(np.power(np.inner(neighbors-point_cloud[i], pca.components_), 2))/float(neighbors.shape[0])

		first_moment_abs = np.abs(np.sum(np.inner(neighbors-point_cloud[i], pca.components_)))/float(neighbors.shape[0])
		second_moment_abs = np.abs(np.sum(np.power(np.inner(neighbors-point_cloud[i], pca.components_), 2)))/float(neighbors.shape[0])

		pca_2d.fit(neighbors[:, 0:2])
		sum_of_eigenvalues_2d = np.sum(pca_2d.explained_variance_)
		ratio_of_eigenvalues_2d = pca_2d.explained_variance_ratio_

		features[i-start] = np.array([elevation,						# 0
									 num_points,						# 1
									 neighborhood_radius,				# 2
									 max_height_diff,					# 3
									 height_std,						# 4
									 neighborhood_density,				# 5
									 sum_of_eigenvalues,				# 6
									 linearity,							# 7
									 planarity,							# 8
									 sphericity,						# 9
									 change_of_curvature,				# 10
									 anisotropy,						# 11
									 omnivariance,						# 12
									 eigentropy,						# 13
									 verticality,						# 14
									 first_moment,						# 15
									 second_moment,						# 16
									 first_moment_abs,					# 17
									 second_moment_abs,					# 18
									 sum_of_eigenvalues_2d,				# 19
									 ratio_of_eigenvalues_2d[0],		# 20
									 ratio_of_eigenvalues_2d[1],		# 21
									 radius_2d,							# 22
									 density_2d])						# 23

	return features


def compute_features(point_cloud, radius, k, type='spherical'):
	num_threads = 4
	splits = [int(point_cloud.shape[0] * i / num_threads) for i in range(num_threads + 1)]

	results = Parallel(n_jobs=num_threads)(delayed(extract_features)(point_cloud, radius, k, type, splits[i-1], splits[i]) for i in range(1, len(splits)))
	features = np.concatenate(results, axis=0)

	return features


def compute_elevation(ground, non_ground, cell_size=2.0):
	max_xyz = np.max([np.max(ground, axis=0), np.max(non_ground, axis=0)], axis=0)
	min_xyz = np.min([np.min(ground, axis=0), np.min(non_ground, axis=0)], axis=0)

	point_grid = np.zeros((math.floor((max_xyz[1]-min_xyz[1])/cell_size) + 1,
						   math.floor((max_xyz[0]-min_xyz[0])/cell_size) + 1), dtype=ground.dtype)
	point_grid.fill(-1)

	for i in range(ground.shape[0]):
		y_index = math.floor((ground[i][1] - min_xyz[1])/cell_size)
		x_index = math.floor((ground[i][0] - min_xyz[0])/cell_size)

		if point_grid[y_index][x_index] == -1 or point_grid[y_index, x_index] > ground[i, 2]:
			point_grid[y_index][x_index] = ground[i, 2]

	unfilled_cells = np.nonzero(point_grid == -1)
	filled_cells = np.nonzero(point_grid != -1)

	# Populate empty cells via nearest-neighbor interpolation
	interpolated_values = interpolate.griddata(np.stack(filled_cells, axis=-1), point_grid[filled_cells],
		np.stack(unfilled_cells, axis=-1), method='nearest')

	point_grid[unfilled_cells] = interpolated_values

	elevation = np.zeros(non_ground.shape[0], dtype=non_ground.dtype)

	for i in range(non_ground.shape[0]):
		y_index = math.floor((non_ground[i, 1] - min_xyz[1])/cell_size)
		x_index = math.floor((non_ground[i, 0] - min_xyz[0])/cell_size)
		elevation[i] = non_ground[i, 2] - point_grid[y_index][x_index]

	return elevation


def extract_dataset_features(dataset_path, output_path):
	dataset_files = os.listdir(dataset_path)
	for dataset_file in tqdm(dataset_files):

		output_filename = dataset_file.split('.')[0] + '.csv'
		if output_filename in os.listdir(output_path):
			continue

		las_data = laspy.open(os.path.join(dataset_path, dataset_file)).read()
		points = np.asarray(las_data.xyz)
		classifications = utils.remap_classes(las_data.classification, classes.DALES_CLASSES)

		points = PointCloud(points, classifications)
		filters = [VoxelFilter(resolution=0.5),
			   StatisticalOutlierFilter()]
		points = utils.apply_filters(points, filters)

		ground = points.point_cloud[points.classification == classes.GROUND]
		non_ground = points.point_cloud[points.classification != classes.GROUND]

		adjusted_points = np.array(non_ground)
		adjusted_points[:, 2] = compute_elevation(ground, non_ground)

		features = compute_features(adjusted_points, k=None, radius=2.0, type='spherical')
		features = np.concatenate((features, np.expand_dims(points.classification[points.classification != classes.GROUND], axis=1)), axis=1)
		np.savetxt(os.path.join(output_path, output_filename), features, delimiter=',')


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Given a directory of LiDAR .las/.laz \
		files with classified ground points, for each file: \
		computes a set of per-point neighborhood features.")
	parser.add_argument('dataset_path', help="Location of input dataset")
	parser.add_argument('features_path', help="Location to save output extracted features")
	args = parser.parse_args();

	DATASET_PATH = args.dataset_path
	FEATURES_PATH = args.features_path
	extract_dataset_features(DATASET_PATH, FEATURES_PATH)
