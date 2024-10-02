import argparse
import glob
from joblib import Parallel, delayed
import laspy
import math
import numpy as np
import os
import pyvista as pv
import random
from scipy import spatial
import sklearn.metrics as metrics
import sys
import torch
from torch import nn, optim
from torch.utils.data import Dataset
from timebudget import timebudget
from tqdm import tqdm

from .. import feature_extraction
from .. import ground_extraction
from .. import utils
from ...config import classes
from ...data import PointCloud
from ...processing import VoxelFilter, StatisticalOutlierFilter


# def save_point_neighborhoods(point_cloud, classifications):

num_neighbors = 20
batch_size = 20
learn_rate = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv1d(4, 16, 3)
		self.pool = nn.MaxPool1d(2, 2)
		self.conv2 = nn.Conv1d(16, 64, 3)
		self.conv3 = nn.Conv1d(64, 128, 3)
		self.fc1 = nn.Linear(192, 64)
		self.fc2 = nn.Linear(64, 16)
		self.fc3 = nn.Linear(16, 4)
		self.softmax = nn.Softmax(dim=0)

	def forward(self, x):
		x = self.pool(nn.functional.relu(self.conv1(x)))
		x = self.pool(nn.functional.relu(self.conv2(x)))
		# x = self.pool(nn.functional.relu(self.conv3(x)))
		x = torch.flatten(x, 1) # flatten all dimensions except batch
		x = nn.functional.relu(self.fc1(x))
		x = nn.functional.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class Dataset(Dataset):
	def __init__(self, dataset_path, features_path, transform=None, target_transform=None):
		las_files = os.listdir(dataset_path)

		self.length = 0
		for file in las_files:
			num_points = laspy.open(os.path.join(dataset_path, file)).read().xyz.shape[0]
			self.length += num_points


def preprocess(DATASET_PATH, FEATURES_PATH, TRAIN_PATH):
	if not os.path.isdir(FEATURES_PATH):
		os.mkdir(FEATURES_PATH)

	dataset_files = os.listdir(TRAIN_PATH)

	for file in tqdm(dataset_files):
		output_filename = os.path.basename(file).split(".")[0] + '.csv'
		if os.path.isfile(DATASET_PATH) and output_filename in os.listdir(FEATURES_PATH) \
			or os.path.isdir(DATASET_PATH) and output_filename in os.listdir(FEATURES_PATH):
			continue
		output_filename_xyz = os.path.basename(file).split(".")[0] + '_xyz.txt'
		output_filename_class = os.path.basename(file).split(".")[0] + '_class.txt'
		output_filename_idx = os.path.basename(file).split(".")[0] + '_idx.txt'

		las_data = laspy.open(os.path.join(TRAIN_PATH, file)).read()
		points = np.asarray(las_data.xyz)
		classifications = utils.remap_classes(las_data.classification, classes.DALES_CLASSES)

		points = PointCloud(points, classifications)
		filters = [VoxelFilter(resolution=0.5),
			   StatisticalOutlierFilter()]
		points = utils.apply_filters(points, filters)

		query_indices = np.nonzero(points.classification != classes.GROUND)
		query_points = points.point_cloud[query_indices]

		points_kdtree = spatial.KDTree(data=points.point_cloud)
		results = points_kdtree.query(query_points, k=num_neighbors)

		np.savetxt(os.path.join(FEATURES_PATH, output_filename_xyz), np.array(points.point_cloud), delimiter=',')
		np.savetxt(os.path.join(FEATURES_PATH, output_filename_class), np.array(points.classification), delimiter=',')
		np.savetxt(os.path.join(FEATURES_PATH, output_filename_idx), np.array(query_indices), delimiter=',')
		np.savetxt(os.path.join(FEATURES_PATH, output_filename), np.array(results[1]), delimiter=',')


def train(features_path):
	net = Net().to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=learn_rate, momentum=0.9)

	num_epochs = 100
	min_loss = np.finfo(float).max
	counter = 0
	start_epoch = 0
	losses = []

	if "checkpoint.pth" in os.listdir():
		checkpoint = torch.load("checkpoint.pth")
		net.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		start_epoch = checkpoint['epoch']
		min_loss = checkpoint['loss']
		losses = checkpoint['losses']

	for epoch in tqdm(range(start_epoch, num_epochs)):

		running_loss = 0.0
		num_batches = 0

		files = glob.glob(os.path.join(features_path, '*.csv'))
		random.shuffle(files)
		for file in tqdm(files):

			neighbors = np.loadtxt(file, delimiter=',').astype(int)

			points = np.loadtxt(os.path.join(features_path, os.path.basename(file).split('.')[0]+"_xyz.txt"), delimiter=',')
			classifications = np.loadtxt(os.path.join(features_path, os.path.basename(file).split('.')[0]+"_class.txt"), delimiter=',').astype(int)
			orig_indices = np.loadtxt(os.path.join(features_path, os.path.basename(file).split('.')[0]+"_idx.txt"), delimiter=',').astype(int)

			query_indices = np.arange(0, neighbors.shape[0])

			assert query_indices.shape[0] == orig_indices.shape[0]
			assert points.shape[0] == classifications.shape[0]
			assert neighbors.shape[0] == query_indices.shape[0]

			rand = np.random.permutation(orig_indices.shape[0])
			query_indices = query_indices[rand]
			orig_indices = orig_indices[rand]

			classifications = utils.remap_classes(classifications, classes.MY_CLASSES)

			# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
			#     shuffle=False, num_workers=2)

			batch_neighbors = np.transpose(points[neighbors[query_indices]], axes=(0, 2, 1))
			batch_neighbors[:, :, :] -= np.repeat(np.expand_dims(batch_neighbors[:, :, 0], 2), batch_neighbors.shape[2], axis=2)

			batch_classes = classifications[neighbors[query_indices]]
			batch_is_ground = np.where(batch_classes == 1, 0, 1)

			batch_input = np.concatenate((batch_neighbors, np.expand_dims(batch_is_ground, 1)), axis=1).astype(np.float32)

			# orig_indices = neighbors[query_indices, 0]
			batch_labels = classifications[orig_indices]

			for i in range(0, batch_input.shape[0]-batch_size, batch_size):

				labels_oh = nn.functional.one_hot(torch.from_numpy(batch_labels), 4)

				# get the inputs; data is a list of [inputs, labels]
				# inputs, labels = data
				inputs = batch_input[i:i+batch_size]
				labels = batch_labels[i:i+batch_size]

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward + backward + optimize
				outputs = net(torch.from_numpy(inputs).to(device))
				loss = criterion(outputs, torch.from_numpy(labels))
				loss.backward()
				optimizer.step()

				# print statistics
				num_batches += 1
				running_loss *= (num_batches - 1) / num_batches
				running_loss += loss.item() / num_batches

		print(f'epoch {epoch + 1} loss: {running_loss:.3f}\n')
		
		if running_loss < min_loss:
			min_loss = running_loss
			counter = 0
			torch.save({
	            'epoch': epoch,
	            'model_state_dict': net.state_dict(),
	            'optimizer_state_dict': optimizer.state_dict(),
	            'loss': min_loss,
	            'losses': losses,
	            }, "best.pth")
		elif running_loss > min_loss:
			counter += 1
			if counter == 3:
				break

		torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': min_loss,
            'losses': losses,
            }, "checkpoint.pth")
		losses.append(running_loss)


def test(dataset_path):
	DATASET_PATH = dataset_path
	TEST_PATH = os.path.join(DATASET_PATH, "test")
	test_dataset = os.listdir(TEST_PATH)

	net = Net().to(device)

	checkpoint = torch.load("best.pth")
	net.load_state_dict(checkpoint['model_state_dict'])

	filters = [VoxelFilter(resolution=0.5),
			   StatisticalOutlierFilter()]

	accuracy_scores = []
	f1_scores = []
	for file in test_dataset:
		las_data = laspy.open(os.path.join(TEST_PATH, file)).read()
		points = np.asarray(las_data.xyz)
		classifications = utils.remap_classes(las_data.classification, classes.DALES_CLASSES)

		points = PointCloud(points, classifications)
		filters = [VoxelFilter(resolution=0.5),
			   StatisticalOutlierFilter()]
		points = utils.apply_filters(points, filters)

		query_indices = np.nonzero(points.classification != classes.GROUND)
		query_points = points.point_cloud[query_indices]

		points_kdtree = spatial.KDTree(data=points.point_cloud)
		neighbors = points_kdtree.query(query_points, k=num_neighbors)[1]

		classifications = classifications[classifications != classes.GROUND]

		orig_indices = query_indices

		query_indices = np.arange(0, neighbors.shape[0])

		classifications = utils.remap_classes(classifications, classes.MY_CLASSES)

		batch_neighbors = np.transpose(points[neighbors[query_indices]], axes=(0, 2, 1))
		batch_neighbors[:, :, :] -= np.repeat(np.expand_dims(batch_neighbors[:, :, 0], 2), batch_neighbors.shape[2], axis=2)

		batch_classes = classifications[neighbors[query_indices]]
		batch_is_ground = np.where(batch_classes == 1, 0, 1)

		batch_input = np.concatenate((batch_neighbors, np.expand_dims(batch_is_ground, 1)), axis=1).astype(np.float32)

		batch_labels = classifications[orig_indices]

		with torch.no_grad():
			pred = net(torch.from_numpy(batch_input).to(device))

			# m = nn.Softmax(dim=1)
			# pred = m(pred)

			pred = np.argmax(pred, axis=0)
			# pred = pred.numpy()

			f1_score = metrics.f1_score(labels_oh, pred, average='micro')
			accuracy_score = metrics.accuracy_score(labels_oh, pred)
			precision_score = metrics.precision_score(labels_oh, pred, average='micro')
			recall_score = metrics.recall_score(labels_oh, pred, average='micro')

			print(f"F1: {f1_score}")
			print(f"Accuracy: {accuracy_score}")
			print(f"Precision: {precision_score}")
			print(f"Recall: {recall_score}")
			# if clf.feature_importances_:
			# 	print(f"Importances: {clf.feature_importances_}")

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

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Given a train dataset of classified \
		LiDAR .las/.laz files, trains a classifier.")
	parser.add_argument('dataset_path',
		help="Location of input training dataset")
	parser.add_argument('features_path',
		nargs='?', const="features", default="features",
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

	preprocess(DATASET_PATH, FEATURES_PATH, TRAIN_PATH)

	# train(FEATURES_PATH)

	test(DATASET_PATH)
