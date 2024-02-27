from pathlib import Path
import os
import torch

# Dataset Parameters
DATASET_PATH = os.path.join(Path(__file__).parent, "../../dataset/")
TRAIN_PATH = os.path.join(DATASET_PATH, '')

IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, f"images_png{os.path.sep}")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, f"masks_png{os.path.sep}")

LIDAR_RESOLUTION = 4.0

INPUT_WIDTH = 128
INPUT_HEIGHT = 128

DIVIDE_WIDTH = INPUT_WIDTH * LIDAR_RESOLUTION
DIVIDE_HEIGHT = INPUT_HEIGHT * LIDAR_RESOLUTION

# Pytorch Parameters
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

PIN_MEMORY = True if DEVICE == "cuda" else False

LEARN_RATE = 0.0001
NUM_EPOCHS = 150
BATCH_SIZE = 64

TEST_SPLIT = 0.15

THRESHOLD = 0.5

BASE_OUTPUT = ""

USE_CHECKPOINT = True
PRESERVE_CHECKPOINTS = True

MODEL_PATH = os.path.join(Path(__file__).parent, "../../model/model.pth")
CHECKPOINT_PATH = os.path.join(Path(__file__).parent, "../../model/checkpoint.pth")
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
