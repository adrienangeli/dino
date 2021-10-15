"""Fine tuning DINO on Food101"""
import os
import sys
import pickle
import argparse
import random

from collections import defaultdict
import collections
from shutil import copy
from shutil import copytree, rmtree

import torch
from torch import nn
from torchvision import models as torchvision_models
from torchvision import transforms as pth_transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageFile
import numpy as np

import utils
import vision_transformer as vits



class FeaturesDataset(torch.utils.data.Dataset):
    def __init__(self, root, prefix, train):
        self.root = root
        self.split = "train" if train else "test"
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.prefix = prefix
        p_ = '' if self.prefix == '' else self.prefix + "_"
        fp_ = os.path.join(self.root, p_ + self.split + "_features.pt")
        self.features = torch.load(fp_, map_location=torch.device(device))

        lp_ = os.path.join(self.root, p_ + self.split + "_labels.pt")
        self.labels = torch.load(lp_, map_location=torch.device(device))

        assert self.features.shape[0] == self.labels.shape[0]

    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, index):
        feat = self.features[index, :]
        label = self.labels[index, :]
        return feat, label.item()

    def features_dim(self):
        return self.features.shape[1]

    def num_classes(self):
        return len(set(self.labels[:, 0].tolist()))







def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):                        
        # Compute prediction and loss
        print("train_loop() -> class prediction & loss calculation")
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        print("train_loop() -> backward prop & optim step")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 1 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fine tuning DINO on Food101')
    parser.add_argument('--data_path', default='/destination/path/for/features/dataset', type=str)
    parser.add_argument('--input_prefix', default='', type=str, help="Prefix prepended to input features and labels filenames.")
    parser.add_argument('--batch_size', default=512, type=int, help='Batch size')
    parser.add_argument('--nlayers', default=1, type=int, help='Number of layers in the MLP')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='Learning rate SGD')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD')
    parser.add_argument('--epochs', default=10, type=int)
    
    args = parser.parse_args()

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    train_dataset = FeaturesDataset(args.data_path, args.input_prefix, True)
    test_dataset = FeaturesDataset(args.data_path, args.input_prefix, False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    features_dim = train_dataset.features_dim()
    num_classes = train_dataset.num_classes()
    #print((features_dim, num_classes))
    model = vits.DINOHead(nlayers=args.nlayers, norm_last_layer=True,
                          in_dim=features_dim,
                          out_dim=num_classes).to('cpu')
    model.eval()
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    
    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")
    

    
    

