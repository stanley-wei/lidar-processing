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
from sklearn.svm import SVC
import sys
from timebudget import timebudget
from tqdm import tqdm

import config
import extract_features
import filters
import ground_extraction


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
	train_dataset = os.listdir(TRAIN_PATH)
	test_dataset = os.listdir(TEST_PATH)

	extracted_features = glob.glob(os.path.join("./", '*.csv'))
	data = np.loadtxt(extracted_features[0], delimiter=',')

	classifications = data[:, -1]
	data = data[:, 0:-1]

	clf = RandomForestClassifier(random_state=0)
	if "rf.joblib" in os.listdir():
		clf = joblib.load("rf.joblib")
	else:
		clf.fit(data, classifications)
		joblib.dump(clf, "rf.joblib")

	data = np.loadtxt(extracted_features[1], delimiter=',')
	classifications = data[:, -1]
	data = data[:, 0:-1]

	pred = np.asarray(clf.predict(data))

	f1_score = metrics.f1_score(classifications, pred, average='micro')
	accuracy_score = metrics.accuracy_score(classifications, pred)
	precision_score = metrics.precision_score(classifications, pred, average='micro')
	recall_score = metrics.recall_score(classifications, pred, average='micro')
	# roc_auc_score

	print(f"F1: {f1_score}")
	print(f"Accuracy: {accuracy_score}")
	print(f"Precision: {precision_score}")
	print(f"Recall: {recall_score}")
	print(f"Importances: {clf.feature_importances_}")
