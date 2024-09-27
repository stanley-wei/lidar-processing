import argparse
import glob
import joblib
import laspy
import numpy as np
import os
import sklearn.metrics as metrics

from . import feature_extraction
from . import ground_extraction
from . import utils
from ..config import classes
from ..data import PointCloud
from ..processing import VoxelFilter, StatisticalOutlierFilter


def classify_building(points, clf, filters):
	points = utils.apply_filters(points, filters)

	# classifications = ground_extraction.progressive_morphological_filter(points)

	ground = points.point_cloud[points.classification == classes.GROUND]
	non_ground = points.point_cloud[points.classification != classes.GROUND]

	elevation = classify_buildings.compute_elevation(ground, non_ground)

	adjusted_points = np.array(non_ground)
	adjusted_points[:, 2] = elevation

	features = classify_buildings.compute_features(adjusted_points, k=20, radius=None, type='knn')

	pred = clf.predict(features)

	classified_header = laspy.LasHeader(version="1.4", point_format=6)
	classified_las = laspy.LasData(classified_header)
	classified_las.xyz = np.concatenate((ground, non_ground), axis=0)
	classified_las.classification = np.concatenate((points.classifications[points.classifications == classes.GROUND], pred), axis=0)
	classified_las.write("classified.laz")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Given a test dataset of classified \
		LiDAR .las/.laz files and a classifier, evaluates the performance of the classifier over the dataset.")
	parser.add_argument('dataset_path', help="Location of input test dataset")
	parser.add_argument('model_path', help="File name of trained point classifier")
	args = parser.parse_args();

	DATASET_PATH = args.dataset_path
	TEST_PATH = os.path.join(DATASET_PATH, "test")
	test_dataset = os.listdir(TEST_PATH)

	clf = joblib.load(args.model_path)
	filters = [VoxelFilter(resolution=0.5),
			   StatisticalOutlierFilter()]

	accuracy_scores = []
	f1_scores = []
	for file in test_dataset:
		las_data = laspy.open(os.path.join(TEST_PATH, file)).read()
		points = np.asarray(las_data.xyz)
		classifications = utils.remap_classes(las_data.classification, classes.DALES_CLASSES)

		points = util.apply_filters(PointCloud(points, classifications), filters)
		classifications = points.classification

		# points.classification = ground_extraction.progressive_morphological_filter(points.point_cloud)
		ground = points.point_cloud[points.classification == classes.GROUND]
		non_ground = points.point_cloud[points.classification != classes.GROUND]

		adjusted_points = np.array(non_ground)
		adjusted_points[:, 2] = feature_extraction.compute_elevation(ground, non_ground)

		features = feature_extraction.compute_features(adjusted_points, k=None, radius=2.0, type='spherical')

		pred = np.asarray(clf.predict(features))

		classifications = classifications[classifications != classes.GROUND]

		f1_score = metrics.f1_score(classifications, pred, average='micro')
		accuracy_score = metrics.accuracy_score(classifications, pred)
		precision_score = metrics.precision_score(classifications, pred, average='micro')
		recall_score = metrics.recall_score(classifications, pred, average='micro')

		print(f"F1: {f1_score}")
		print(f"Accuracy: {accuracy_score}")
		print(f"Precision: {precision_score}")
		print(f"Recall: {recall_score}")
		# print(f"Importances: {clf.feature_importances_}")

		accuracy_scores.append(accuracy_score)
		f1_scores.append(f1_score)

	print("Overall Results")
	print(f"Accuracy mean: {np.mean(accuracy_scores)}")
	print(f"Accuracy std: {np.std(accuracy_scores)}")
	print(f"Accuracy min: {np.min(accuracy_scores)}")
	print(f"Accuracy max: {np.max(accuracy_scores)}")
	
	print(f"F1 mean: {np.mean(f1_scores)}")
	print(f"F1 std: {np.std(f1_scores)}")
	print(f"F1 min: {np.min(f1_scores)}")
	print(f"F1 max: {np.max(f1_scores)}")
