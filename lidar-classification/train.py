import albumentations
from albumentations.pytorch import ToTensorV2
import os
from segmentation_models_pytorch.losses import DiceLoss
from sklearn.model_selection import train_test_split
import torch
from torch import nn, save
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import config
from dataset import SegmentationDataset
#from aerial import UNet
from model import UNet
#from utils import EarlyStopper

import numpy as np

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    train_steps = len(dataloader.dataset) // config.BATCH_SIZE

    model.train()
    total_loss = 0

    for (batch, (X, y)) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()

        (X, y) = (X.to(config.DEVICE), y.to(config.DEVICE))

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss

    return total_loss / train_steps


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    test_steps = len(dataloader.dataset) // config.BATCH_SIZE
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in tqdm(dataloader):
            (X, y) = (X.to(config.DEVICE), y.to(config.DEVICE))
            pred = model(X)
            test_loss += loss_fn(pred, y)

    print(f"Test Error: Avg loss: {test_loss / num_batches:>8f} \n")
    return test_loss / test_steps

if __name__ == "__main__":
    imagePaths = sorted(os.path.join(config.IMAGE_DATASET_PATH,file) for file in os.listdir(config.IMAGE_DATASET_PATH))
    maskPaths = sorted(os.path.join(config.MASK_DATASET_PATH,file) for file in os.listdir(config.MASK_DATASET_PATH))

    split = train_test_split(imagePaths, maskPaths,
        test_size=config.TEST_SPLIT, random_state=42)
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

    # input_transforms = albumentations.Compose([albumentations.HorizontalFlip(p=0.5),
    #     albumentations.VerticalFlip(p=0.5),
    #     albumentations.RandomCrop(height=config.INPUT_HEIGHT,
    #         width=config.INPUT_WIDTH)])
    input_transforms = albumentations.Compose([albumentations.CenterCrop(height=config.INPUT_HEIGHT,
            width=config.INPUT_WIDTH)])

    train_dataset = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
        transforms=input_transforms, data_type="image")
    test_dataset = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
        transforms=input_transforms, data_type="image")
    print(f"[INFO] found {len(train_dataset)} examples in the training set...")
    print(f"[INFO] found {len(test_dataset)} examples in the test set...")

    print(train_dataset.classes)

    train_dataloader = DataLoader(train_dataset, shuffle=True,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=os.cpu_count())
    test_dataloader = DataLoader(test_dataset, shuffle=False,
        batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
        num_workers=os.cpu_count())

    #model = UNet(n_classes = len(train_dataset.classes), n_channels=1)
    model = UNet(nbClasses = len(train_dataset.classes))
    #model = UNet(num_classes = len(train_dataset.classes), in_channels=1).to(config.DEVICE)

    #loss_fn = nn.CrossEntropyLoss()    
    loss_fn = DiceLoss(mode='multiclass', classes=range(len(train_dataset.classes)))
    optimizer = Adam(model.parameters(), lr=config.LEARN_RATE)

    train_steps = len(train_dataset) // config.BATCH_SIZE
    test_steps = len(test_dataset) // config.BATCH_SIZE

    epoch = 0
    early_stopper = EarlyStopper(patience=3, min_delta=0.5)

    if os.path.exists(config.CHECKPOINT_PATH) and config.USE_CHECKPOINT:
        checkpoint = torch.load(config.CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        print(f"Found checkpoint at {config.CHECKPOINT_PATH}; resuming from epoch {epoch+1}")

    while epoch < config.NUM_EPOCHS:
        print(f"Epoch {epoch+1}\n-------------------------------")
        
        print("Training")
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)

        print("Testing")
        test_loss = test_loop(test_dataloader, model, loss_fn)

        epoch += 1

        print("Train loss: {:.6f}, Test loss: {:.4f}".format(
            train_loss, test_loss))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, config.CHECKPOINT_PATH)
        torch.save(model, config.MODEL_PATH[:-4] + "_checkpoint.pth")

        if epoch % 5 == 0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, config.CHECKPOINT_PATH[:-4]+f"_{epoch}.pth")

        if early_stopper.early_stop(test_loss):             
            break

    print("Done!")

    torch.save(model, config.MODEL_PATH)
    os.remove(config.CHECKPOINT_PATH)
