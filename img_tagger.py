#!/usr/bin/env python3


"""
Implementation of image tagger using convolution network classifier.

"""


import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, hub
import torchvision
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.transforms import v2
from torchvision.io import read_image
import wandb

from tempfile import TemporaryDirectory


parser = argparse.ArgumentParser()
parser.add_argument("--img_dir_train", type=str, help="Path to the directory with training images")
parser.add_argument("--img_dir_val", type=str, help="Path to the directory with evaluation images")
parser.add_argument("--train_labels", type=str, help="Path to the text file with training labels")
parser.add_argument("--val_labels", type=str, help="Path to the text file with evaluation labels")


img_exts = (".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG")


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    wandb.login()
    run = wandb.init(project="image_tagger", dir="/home/m/Desktop/wandb/")

    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    net = efficientnet_v2_s(weights = weights)
    img_prep = weights.transforms() # preprocess each image - resize, crop, normalize

    data_transforms = {
        'train': v2.Compose([
            v2.RandomResizedCrop(img_prep.crop_size),
            v2.RandomHorizontalFlip(),
            v2.Normalize(img_prep.mean, img_prep.std),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.0),
        ]),
        'val': img_prep
    }

    image_datasets = {"train": TagDataset(args.img_dir_train, args.train_labels, data_transforms["train"]),
                      "val": TagDataset(args.img_dir_val, args.val_labels, data_transforms["val"])}

    dataloaders = {"train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=4,
                                                shuffle=True, num_workers=4),
                   "val": torch.utils.data.DataLoader(image_datasets["val"], batch_size=4,
                                                shuffle=True, num_workers=4)}
    
    dataset_sizes = {"train": len(image_datasets["train"]), 
                     "val": len(image_datasets["val"])}
    class_names = image_datasets['train'].classes

    cls_num = len(class_names)

    # Freeze the backbone layers
    for param in net.parameters():
        param.requires_grad = False

    # Replace the classifier - new layers are not freezed
    net.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=cls_num, bias=True),
        )

    net = net.to(device)

    

    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    optimizer = torch.optim.SGD(net.classifier.parameters(), lr=1e-3, momentum=0.9)
    # Decay LR by a factor of 0.1 every 10 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 100

    wandb.watch(net, log_freq=10)
    net = train_model(net, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs, device)


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25, device="cpu"):
    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in tqdm(range(num_epochs)):
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over epochs
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        # get one-hot binary predictions
                        prob_threshold = 0.5
                        preds = (outputs > prob_threshold).float()

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            if epoch % 1 == 0:
                wandb.log({"loss": epoch_loss, "acc": epoch_acc})

        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model
    

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(3.0)  # pause a bit so that plots are updated


class TagDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir_path, img_labels_path, transform=None):
        self.img_dir_path = img_dir_path
        self.img_labels = {}
        self.classes = []
        self.transform = transform

        self.read_labels(img_labels_path)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = list(self.img_labels.keys())[idx]
        img_labels = torch.Tensor(self.img_labels[img_name])
        img_path = os.path.join(self.img_dir_path, img_name)
        img_torch = read_image(img_path)
        img_torch = img_torch / 255.0
        if self.transform:
            img_torch = self.transform(img_torch)

        return img_torch, img_labels
    
    def read_labels(self, label_path):
        """
        Read the labels from a text file and return the list of all possible labels
        and a dictionary with the one-hot encoded labels for each image.

        Parameters:
        label_path (str): The path to the text file containing the labels.

        Returns:
        label_names (list): The list of all possible labels.
        img_labels (dict): The dictionary with the label indices for each image.
        """

        assert os.path.exists(label_path), f"Label file not found: {label_path}"

        label_names = []
        img_labels = {}
        with open(label_path, "rt") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue
                
                if not label_names:
                    label_names = [l.strip() for l in line.split()]
                
                else:
                    words = [l.strip() for l in line.split()]
                    img_name = words[0]

                    if len(words) > 1:
                        labels = words[1:]
                    else:
                        labels = []
                    
                    onehot_labels = [0] * len(label_names)
                    for label in labels:
                        onehot_labels[label_names.index(label)] = 1
                    img_labels[img_name] = onehot_labels

                    # label_idxs = []
                    # for label in labels:
                        # label_idxs.append(label_names.index(label))
                    # img_labels[img_name] = label_idxs

        self.classes = label_names
        self.img_labels = img_labels


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)