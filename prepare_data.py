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
import torch.distributed as dist
import torch.backends.cudnn as cudnn

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
    def __init__(self, root_dir, transform, dataset_fraction, train):
        self.food_path = root_dir
        self.img_path = os.path.join(self.food_path, "images")
        self.img_ext = ".jpg"
        self.meta_path = os.path.join(self.food_path, "meta")
        self.dataset_fraction = dataset_fraction

        self.split_dirname = "train" if train else "valid"
        self.split_fname = "train.txt" if train else "test.txt"
        self.split_fname = os.path.join(self.meta_path, self.split_fname)
        self.split_samples = pd.read_csv(self.split_fname, header=None)
        self.split_path = os.path.join(self.food_path, self.split_dirname)
        prepare_data(self.split_fname, self.img_path, self.split_path)
        
        super().__init__(self.split_path, transform)


    def __len__(self):
        return int(dataset_fraction * super().__len__())

    
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


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for samples, index, lbls in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        lbls = lbls.cpu()
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            labels = torch.zeros(len(data_loader.dataset), lbls.shape[-1], dtype=torch.long)
            if use_cuda:
                features = features.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")
            print(f"Storing feature labels into tensor of shape {labels.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        label_l = list(lbls_all.unbind(0))
        label_all_reduce = torch.distributed.all_gather(label_l, lbls, async_op=True)
        label_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
                labels.index_copy_(0, index_all, torch.cat(label_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
                labels.index_copy_(0, index_all.cpu(), torch.cat(label_l).cpu())
    return features, labels



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO feature computation on an image dataset')
    parser.add_argument('--data_path', default='/path/to/dataset/')
    parser.add_argument('--dataset_fraction', default=1, type=float, help="Floating number between 0 and 1 representing the fraction of the dataset we will use (e.g., 0.1 means we use 10% of all images in the dataset)")
    parser.add_argument('--output_path', default='/path/to/where/computed/features/will/be/saved', type=str)
    parser.add_argument('--output_prefix', default='dino', type=str, help="Prefix prepended to output features and labels filenames.")
    parser.add_argument('--multiscale', default=False, type=utils.bool_flag)
    parser.add_argument('--imsize', default=224, type=int, help='Image size')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--distributed', default=False, type=utils.bool_flag)
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    args = parser.parse_args()

    if args.distributed and args.use_cuda:
        utils.init_distributed_mode(args)
        cudnn.benchmark = True
    

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

    if args.distributed and args.use_cuda:
        model.cuda()
    
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
        
        dataset_train = Food101Dataset(args.data_path, transform, dataset_fraction, train)
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
        train_features, train_labels = None, None
        if args.distributed:
            train_features, train_labels = extract_features(model, data_loader_train, args.use_cuda, multiscale=args.multiscale)
        else:
            train_features, train_labels = extract_features_sequential(model, data_loader_train, multiscale=args.multiscale)

        if not args.distributed or utils.get_rank() == 0:
            # normalize features
            train_features = nn.functional.normalize(train_features, dim=1, p=2)
        
            # save features
            train_or_test_prefix = "train" if train is True else "test"
            torch.save(train_features,
                       os.path.join(args.output_path,
                                    args.output_prefix + '_' + train_or_test_prefix + '_features.pt'))
        
            # save labels
            torch.save(train_labels,
                       os.path.join(args.output_path,
                                    args.output_prefix + '_' + train_or_test_prefix + '_labels.pt'))
    
    
