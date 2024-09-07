import glob
import joblib
import laspy
import math
import numpy as np
import os
import pyvista as pv
from scipy import interpolate, spatial
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
import sys
from timebudget import timebudget
from tqdm import tqdm

import config
import extract_features
import filters
import ground_extraction
import lidar_data


def classify_building(point_cloud, clf):
	points = filters.voxel_filter(point_cloud, resolution=2.0)
	points = filters.remove_statistical_outliers(points)

	classifications = ground_extraction.progressive_morphological_filter(points)

	ground = points[classifications == config.GROUND]
	non_ground = points[classifications != config.GROUND]

	elevation = classify_buildings.compute_elevation(ground, non_ground)

	adjusted_points = np.array(non_ground)
	adjusted_points[:, 2] = elevation

	features = classify_buildings.compute_features(adjusted_points, k=20, radius=None, type='knn')

	pred = clf.predict(features)

	classified_header = laspy.LasHeader(version="1.4", point_format=6)
	classified_las = laspy.LasData(classified_header)
	classified_las.xyz = np.concatenate((ground, non_ground), axis=0)
	classified_las.classification = np.concatenate((classifications[classifications == config.GROUND], pred), axis=0)
	classified_las.write("classified.laz")


if __name__ == "__main__":
	DATASET_PATH = sys.argv[1]
	TRAIN_PATH = os.path.join(DATASET_PATH, "train")
	TEST_PATH = os.path.join(DATASET_PATH, "test")

	TRAIN_BATCH = False

	train_dataset = os.listdir(TRAIN_PATH)
	test_dataset = os.listdir(TEST_PATH)
	extracted_features = glob.glob(os.path.join("./", '*.csv'))

	if "rf.joblib" in os.listdir():
		clf = joblib.load("rf.joblib")
	else:
		print("Train")
		if TRAIN_BATCH:
			clf = RandomForestClassifier(warm_start=True, n_estimators=10)
			for i in range(0, len(extracted_features)):
				print(extracted_features[i])
				features = np.loadtxt(extracted_features[i], delimiter=',')
				clf.n_estimators += 10
				clf.fit(features[:, 0:-1], features[:, -1])
		else:
			# clf = RandomForestClassifier(n_estimators=100)
			clf = make_pipeline(StandardScaler(),
                    LinearSVC(random_state=0, tol=1e-5))
			features = np.loadtxt(extracted_features[0], delimiter=',')
			for i in range(1, len(extracted_features)):
				features = np.concatenate((
					features, np.loadtxt(extracted_features[i], delimiter=',')), axis=0)
			clf.fit(features[:, 0:-1], features[:, -1])

		joblib.dump(clf, "rf.joblib")

	print("Test")
	for file in test_dataset:
		print(f"\n{file}")

		las_data = laspy.open(os.path.join(TEST_PATH, file)).read()
		points = np.asarray(las_data.xyz)
		classes = extract_features.remap_classes(las_data.classification, config.DALES_CLASSES)

		points = lidar_data.PointCloud(points, classes)
		points = filters.voxel_filter(points, resolution=2.0)
		points = filters.remove_statistical_outliers(points)
		classes = points.classification

		# points.classification = ground_extraction.progressive_morphological_filter(points.point_cloud)
		ground = points.point_cloud[points.classification == config.GROUND]
		non_ground = points.point_cloud[points.classification != config.GROUND]

		adjusted_points = np.array(non_ground)
		adjusted_points[:, 2] = extract_features.compute_elevation(ground, non_ground)

		features = extract_features.compute_features(adjusted_points, k=None, radius=2.0, type='spherical')

		pred = np.asarray(clf.predict(features))

		classes = classes[classes != config.GROUND]

		f1_score = metrics.f1_score(classes, pred, average='micro')
		accuracy_score = metrics.accuracy_score(classes, pred)
		precision_score = metrics.precision_score(classes, pred, average='micro')
		recall_score = metrics.recall_score(classes, pred, average='micro')
		# roc_auc_score

		print(f"F1: {f1_score}")
		print(f"Accuracy: {accuracy_score}")
		print(f"Precision: {precision_score}")
		print(f"Recall: {recall_score}")
		# print(f"Importances: {clf.feature_importances_}")
