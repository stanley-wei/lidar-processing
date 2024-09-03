import glob
from joblib import Parallel, delayed
import laspy
import math
import multiprocessing
import numpy as np
import os
import pyvista as pv
from scipy import interpolate, spatial
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sys
from timebudget import timebudget
from tqdm import tqdm

import config
import filters
import ground_extraction


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


def extract_features(point_cloud, query_points, start, end):
	num_features = 24
	features = np.zeros((end-start, num_features), dtype=float)

	pca = PCA(n_components=3)
	pca_2d = PCA(n_components=2)
	for i in range(start, end):
		neighbors = point_cloud[query_points[i]]

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
		eigentropy = -np.sum(pca.explained_variance_ratio_ * np.log(pca.explained_variance_ratio_))

		verticality = np.arccos(np.dot(pca.components_[2], [0,0,1]) / np.linalg.norm(pca.components_[2]))

		first_moment = np.sum(np.inner(neighbors-point_cloud[i], pca.components_))/float(neighbors.shape[0])
		second_moment = np.sum(np.power(np.inner(neighbors-point_cloud[i], pca.components_), 2))/float(neighbors.shape[0])

		first_moment_abs = np.abs(np.sum(np.inner(neighbors-point_cloud[i], pca.components_)))/float(neighbors.shape[0])
		second_moment_abs = np.abs(np.sum(np.power(np.inner(neighbors-point_cloud[i], pca.components_), 2)))/float(neighbors.shape[0])

		pca_2d.fit(neighbors[:, 0:2])
		sum_of_eigenvalues_2d = np.sum(pca_2d.explained_variance_)
		ratio_of_eigenvalues_2d = pca_2d.explained_variance_ratio_

		features[i-start, 0] = elevation
		features[i-start, 1] = num_points
		features[i-start, 2] = neighborhood_radius
		features[i-start, 3] = max_height_diff
		features[i-start, 4] = height_std
		features[i-start, 5] = neighborhood_density
		features[i-start, 6] = sum_of_eigenvalues
		features[i-start, 7] = linearity
		features[i-start, 8] = planarity
		features[i-start, 9] = sphericity
		features[i-start, 10] = change_of_curvature
		features[i-start, 11] = anisotropy
		features[i-start, 12] = omnivariance
		features[i-start, 13] = eigentropy
		features[i-start, 14] = verticality
		features[i-start, 15] = first_moment
		features[i-start, 16] = second_moment
		features[i-start, 17] = first_moment_abs
		features[i-start, 18] = second_moment_abs
		features[i-start, 19] = sum_of_eigenvalues_2d
		features[i-start, 20] = ratio_of_eigenvalues_2d[0]
		features[i-start, 21] = ratio_of_eigenvalues_2d[1]
		features[i-start, 22] = radius_2d
		features[i-start, 23] = density_2d

	return features


@timebudget
def compute_features(point_cloud, radius, k, type='spherical'):
	query_points, query_distances = retrieve_point_neighborhoods(point_cloud, radius, k, type)

	num_threads = 4
	splits = [int(point_cloud.shape[0] * i / num_threads) for i in range(num_threads + 1)]

	results = Parallel(n_jobs=num_threads)(delayed(extract_features)(point_cloud, query_points, splits[i-1], splits[i]) for i in range(1, len(splits)))

	features = np.concatenate(results, axis=0)

	return features


def remap_classes(classifications, class_dict):
	remapped = np.zeros(classifications.shape, dtype=classifications.dtype)

	for key in class_dict.keys():
		remapped[np.nonzero(classifications == key)] = class_dict[key]

	return remapped


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


def extract_dataset_features(path, dataset):

	for dataset_file in tqdm(dataset):
		output_path = dataset_file.split('.')[0] + '.csv'

		if output_path in os.listdir("./"):
			continue

		las_data = laspy.open(os.path.join(path, dataset_file)).read()
		points = np.asarray(las_data.xyz)
		classifications = remap_classes(las_data.classification, config.DALES_CLASSES)

		ground = points[classifications == config.GROUND]
		non_ground = points[classifications != config.GROUND]

		elevation = compute_elevation(ground, non_ground)

		adjusted_points = np.array(non_ground)
		adjusted_points[:, 2] = elevation

		features = compute_features(adjusted_points, k=20, radius=None, type='knn')

		print(features.shape)
		print(classifications[classifications != config.GROUND].shape)

		features = np.concatenate((features, np.expand_dims(classifications[classifications != config.GROUND], axis=1)), axis=1)

		np.savetxt(f'{output_path}', features, delimiter=',')


def color_by_classification(point_cloud, classifications):
	cloud = pv.PolyData(point_cloud)

	cloud['point_color'] = classifications

	pv.plot(cloud, scalars='point_color')


if __name__ == "__main__":
	DATASET_PATH = sys.argv[1]

	TRAIN_PATH = os.path.join(DATASET_PATH, "train")
	TEST_PATH = os.path.join(DATASET_PATH, "test")

	train_dataset = os.listdir(TRAIN_PATH)
	test_dataset = os.listdir(TEST_PATH)

	extract_dataset_features(TRAIN_PATH, train_dataset)