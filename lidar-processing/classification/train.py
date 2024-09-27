import argparse
import glob
import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from tqdm import tqdm

from . import feature_extraction
from . import ground_extraction
from ..data import PointCloud


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Given a train dataset of classified \
		LiDAR .las/.laz files, trains a classifier.")
	parser.add_argument('dataset_path',
		help="Location of input training dataset")
	parser.add_argument('features_path',
		help="Intermediate location to save extracted features")
	parser.add_argument('--model-path', dest='model_path', 
		nargs='?', const="model.joblib", default="model.joblib",
		help="File name for output trained model (Default: \"model.joblib\")")
	parser.add_argument('--clf', nargs='?', const='rf', default='rf',
		help='Classifiers: "rf", "svc", "knn" (Default: "rf")')
	args = parser.parse_args();

	DATASET_PATH = args.dataset_path
	FEATURES_PATH = args.features_path
	TRAIN_PATH = os.path.join(DATASET_PATH, "train")

	dataset_files = os.listdir(TRAIN_PATH)
	feature_extraction.extract_dataset_features(TRAIN_PATH, FEATURES_PATH)

	extracted_features = glob.glob(os.path.join(FEATURES_PATH, '*.csv'))

	features = np.loadtxt(extracted_features[0], delimiter=',')
	for i in tqdm(range(1, len(extracted_features))):
		features = np.concatenate((
			features, np.loadtxt(extracted_features[i], delimiter=',')), axis=0)

	if args.clf == "rf":
		clf = RandomForestClassifier(n_estimators=100)
	elif args.clf == "linear-svc":
		clf = make_pipeline(StandardScaler(),
				LinearSVC(random_state=0, tol=1e-5))
	elif args.clf == "knn":
		clf = KNeighborsClassifier()

	clf.fit(features[:, 0:-1], features[:, -1])

	joblib.dump(clf, args.model_path)
