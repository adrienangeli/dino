# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import pickle
import argparse

from collections import defaultdict
import collections
from shutil import copy
from shutil import copytree, rmtree

import torch
from torch import nn

import torchvision
from torchvision import models as torchvision_models
from torchvision import transforms as pth_transforms

from PIL import Image, ImageFile
import numpy as np
import pandas as pd

import utils
import vision_transformer as vits


# Helper method to split dataset into train and test folders
def prepare_data(filepath, src, dest):
    classes_images = defaultdict(list)
    with open(filepath, 'r') as txt:
        paths = [read.strip() for read in txt.readlines()]
        for p in paths:
            food = p.split('/')
            classes_images[food[0]].append(food[1] + '.jpg')

    for food in classes_images.keys():
        print("\nCopying images into ",food)
        if not os.path.exists(os.path.join(dest,food)):
            os.makedirs(os.path.join(dest,food))
        for i in classes_images[food]:
            copy(os.path.join(src,food,i), os.path.join(dest,food,i))
    print("Copying Done!")


class Food101Dataset(torchvision.datasets.ImageFolder):
    def __init__(self, root_dir, transform, train):
        self.food_path = root_dir
        self.img_path = os.path.join(self.food_path, "images")
        self.img_ext = ".jpg"
        self.meta_path = os.path.join(self.food_path, "meta")

        self.split_dirname = "train" if train else "valid"
        self.split_fname = "train.txt" if train else "test.txt"
        self.split_fname = os.path.join(self.meta_path, self.split_fname)
        self.split_samples = pd.read_csv(self.split_fname, header=None)
        self.split_path = os.path.join(self.food_path, self.split_dirname)
        prepare_data(self.split_fname, self.img_path, self.split_path)
        
        super().__init__(self.split_path, transform)


    def __len__(self):
        return super().__len__()

    
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return img, index, torch.as_tensor(label).unsqueeze(0)



@torch.no_grad()
def extract_features_sequential(model, data_loader, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features, labels = None, None
    for samples, index, lbls in metric_logger.log_every(data_loader, 10):    
        samples = samples.cpu()
        index = index.cpu()
        lbls = lbls.cpu()
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()

        # init storage feature matrix
        if features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            print(f"Storing features into tensor of shape {features.shape}")

        if labels is None:
            labels = torch.zeros(len(data_loader.dataset), lbls.shape[-1], dtype=torch.long)
            print(f"Storing feature labels into tensor of shape {labels.shape}")
        
        # update storage feature matrix
        features.index_copy_(0, index.cpu(), feats.cpu())
        labels.index_copy_(0, index.cpu(), lbls.cpu())
    return features, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO feature computation on an image dataset')
    parser.add_argument('--data_path', default='/path/to/dataset/')
    parser.add_argument('--output_path', default='/path/to/where/computed/features/will/be/saved')
    parser.add_argument('--multiscale', default=False, type=utils.bool_flag)
    parser.add_argument('--imsize', default=224, type=int, help='Image size')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    args = parser.parse_args()

    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Resize(args.imsize),
        #pth_transforms.CenterCrop(args.imsize),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.eval()

    # load pretrained weights
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(args.pretrained_weights, msg))
    elif args.arch == "vit_small" and args.patch_size == 16:
        print("Since no pretrained weights have been provided, we load pretrained DINO weights on Google Landmark v2.")
        model.load_state_dict(torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/dino_vitsmall16_googlelandmark_pretrain/dino_vitsmall16_googlelandmark_pretrain.pth", map_location="cpu"))
    else:
        print("Warning: We use random weights.")


    for train in [True, False]:
        mode = "TRAIN" if train is True else "TEST"
        print(mode + " mode!")
        print("--------------------------------------------------------")
        
        dataset_train = Food101Dataset(args.data_path, transform, train)
        sampler_train = torch.utils.data.SequentialSampler(dataset_train)
    
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=1,
            pin_memory=True,
            drop_last=False,
        )

        print(f"dataset: {len(dataset_train)} imgs")

        ############################################################################
        # extract features
        train_features, train_labels = extract_features_sequential(model, data_loader_train, multiscale=args.multiscale)
        
        # normalize features
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        
        # save features
        prefix = "train" if train is True else "test"
        torch.save(train_features, os.path.join(args.output_path, prefix + '_features.pt'))
        
        # save labels
        torch.save(train_labels, os.path.join(args.output_path, prefix + '_labels.pt'))
    
    
