"""Fine tuning DINO on Food101"""
import os
import sys
import pickle
import argparse
import random
import json
from typing import Optional, Dict

from collections import defaultdict
import collections
from shutil import copy
from shutil import copytree, rmtree

import valohai

import torch
from torch import nn
from torchvision import models as torchvision_models
from torchvision import transforms as pth_transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from PIL import Image, ImageFile
import numpy as np

import utils
import vision_transformer as vits
from eval_linear import LinearClassifier



class FeaturesDataset(torch.utils.data.Dataset):
    def __init__(self, root, prefix, train, device):
        self.root = root
        self.split = "train" if train else "test"
        
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



def train_loop(dataloader, model, loss_fn, optimizer, use_cuda=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0
    
    for batch, (X, y) in enumerate(dataloader):                        
        # Compute prediction and loss
        samples = X.cuda(non_blocking=True) if use_cuda else X.cpu()
        pred = model(samples)

        labels = y.cuda(non_blocking=True) if use_cuda else y.cpu()
        loss = loss_fn(pred, labels)
        train_loss += loss.item()
        correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if use_cuda:
            torch.cuda.synchronize()
        
        loss, current = loss.item(), batch * len(samples)
        if batch % 100 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    correct *= 100./size

    return train_loss, correct
        


def test_loop(dataloader, model, loss_fn, use_cuda=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            samples = X.cuda(non_blocking=True) if use_cuda else X.cpu()
            pred = model(samples)

            labels = y.cuda(non_blocking=True) if use_cuda else y.cpu()
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()

            if use_cuda:
                torch.cuda.synchronize()

    test_loss /= num_batches
    correct *= 100./size
    print(f"Test Error: \n Accuracy: {(correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct
    
    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fine tuning DINO on Food101')
    parser.add_argument('--data_path', default='/destination/path/for/features/dataset', type=str)
    parser.add_argument('--input_prefix', default='', type=str, help="Prefix prepended to input features and labels filenames.")
    parser.add_argument('--batch_size', default=512, type=int, help='Batch size')
    parser.add_argument('--nlayers', default=1, type=int, help='Number of layers in the MLP')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='Learning rate SGD')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--distributed', default=False, type=utils.bool_flag)
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    
    args = parser.parse_args()

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))

    if args.distributed and args.use_cuda:
        utils.init_distributed_mode(args)
        cudnn.benchmark = True
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = FeaturesDataset(args.data_path, args.input_prefix, True, device)
    test_dataset = FeaturesDataset(args.data_path, args.input_prefix, False, device)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    features_dim = train_dataset.features_dim()
    num_classes = train_dataset.num_classes()
    #print((features_dim, num_classes))

    model = None
    if args.nlayers > 1:
        model = vits.DINOHead(nlayers=args.nlayers, norm_last_layer=True,
                              in_dim=features_dim,
                              out_dim=num_classes)
    else:
        model = LinearClassifier(features_dim, num_labels=num_classes)
    
    if args.distributed and args.use_cuda:
        model.cuda()
    
    model.eval()
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)
    
    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss, train_acc = train_loop(train_dataloader, model, loss_fn, optimizer, (args.distributed and args.use_cuda))
        test_loss, test_acc = test_loop(test_dataloader, model, loss_fn, (args.distributed and args.use_cuda))

        with valohai.logger() as logger:
            logger.log('epoch', t)
            logger.log('accuracy', train_acc)
            logger.log('loss', train_loss)
            logger.log('test_accuracy', test_acc)
            logger.log('test_loss', test_loss)

    print("Done!")
    

    
    

