from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import math
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import json
import random
import copy as cp
import numpy as np
import time
from utils.ImageNette import Imagenette
from utils.GTSRB import GTSRB
from utils.model_zoo import ResNet18, SimpleNet
from utils.util import pert_est_class_pair, data_split

parser = argparse.ArgumentParser(description='Test transferability of estimated perturbation')
parser.add_argument("--mode", default="patch", type=str)
parser.add_argument("--RUN", default=-1, type=int)
parser.add_argument("--SETTING", default="", type=str)
parser.add_argument("--ATTACK", default="", type=str)
parser.add_argument("--DEVICE", default=-1, type=int)
parser.add_argument("--DATASET", default="", type=str)
args = parser.parse_args()

# Load attack configuration
with open('config.json') as config_file:
    config = json.load(config_file)

if args.RUN >= 0:
    config["RUN"] = args.RUN
if args.DEVICE >= 0:
    config["DEVICE"] = args.DEVICE
if args.SETTING == "A2A" or args.SETTING == "A2O" or args.SETTING == "rand" or args.SETTING == "x2x":
    config["SETTING"] = args.SETTING
if args.ATTACK == "patch" or args.ATTACK == "perturbation" or args.ATTACK == "CLA" or args.ATTACK == "clean":
    config["PATTERN_TYPE"] = args.ATTACK
if args.DATASET == "cifar10" or args.DATASET == "gtsrb" or args.DATASET == "imagenette":
    config["DATASET"] = args.DATASET

device = config["DEVICE"]
start_time = time.time()
random.seed()

# Load model to be inspected
model_path = 'attacks/{}/{}/{}/{}'.format(config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"])
RED_path = '{}_estimated/{}/{}/{}/{}'.format(args.mode, config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"])
ckpt_path = 'color_maps_{}/{}/{}/{}/{}'.format(args.mode, config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"])

if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)


print("Detect: {}, Dataset: {}, Mode: {}, Type: {},  Run: {}".format(args.mode, config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"]))

# Load clean images for detection
print('==> Preparing data..')
if config["DATASET"] == "cifar10":
    config["NUM_CLASS"] = 10
    transform_test = transforms.Compose([transforms.ToTensor()])
    detectset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    model = ResNet18(num_classes=10)
elif config["DATASET"] == "gtsrb":
    config["NUM_CLASS"] = 43
    transform_test = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])
    detectset = GTSRB(root='./data', split='test', download=False, transform=transform_test)
    model = SimpleNet()
elif config["DATASET"] == "imagenette":
    config["NUM_CLASS"] = 10
    transform_test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    detectset = Imagenette(root='./data/imagenette2', train=False, transform=transform_test)
    model = ResNet18(num_classes=10)
model = model.to(device)
model.load_state_dict(torch.load(os.path.join(model_path, 'model_contam.pth'),  map_location=torch.device(device)))
model.eval()
NC = config["NUM_CLASS"]     # Number of classes
NI = 10  
# # Perform patch estimation for each class pair
correct_path = os.path.join(RED_path, "correct.npy")
target_path = os.path.join(RED_path, "targets.npy")
if os.path.exists(correct_path) and os.path.exists(target_path):
    print("Loading correctly classified images")
    correct = np.load(correct_path)
    targets = np.load(target_path)
else: 
    imgs = []
    labels = []
    index = []
    for i in range(len(detectset.targets)):
        sample, label = detectset.__getitem__(i)
        imgs.append(sample)
        labels.append(label)
        index.append(i)
    imgs = torch.stack(imgs)
    labels = torch.tensor(labels)
    index = torch.tensor(index)
    correct = []
    targets = []

    bs = 128
    for img, label, i in zip(imgs.chunk(math.ceil(len(imgs) / bs)),
                                labels.chunk(math.ceil(len(imgs) / bs)), index.chunk(math.ceil(len(imgs) / bs))):
        img = img.to(device)
        target = label.to(device)
        i = i.to(device)
        with torch.no_grad():
            _, _, outputs = model(img)
            _, predicted = outputs.max(1)
        correct.extend(i[predicted.eq(target)].cpu().numpy())
        targets.extend(target[predicted.eq(target)].cpu().numpy())
images_all = []
ind_all = []
for c in range(NC):
    ind = [correct[i] for i, label in enumerate(targets) if label == c]
    ind = np.random.choice(ind, NI, replace=False)
    images_all.append(torch.stack([detectset[i][0] for i in ind]))
    ind_all.append(ind)

for t in range(NC):
    for s in range(NC):
        if s == t:
            continue
        # Get the estimated perturbation
        if args.mode == 'pert':
            pert = torch.load(os.path.join(RED_path, 'pert_{}_{}'.format(s, t))).to(device)
        elif args.mode == 'patch':
            pattern = torch.load(os.path.join(RED_path, 'pattern_{}_{}'.format(s, t))).to(device)
            mask = torch.load(os.path.join(RED_path, 'mask_{}_{}'.format(s, t))).to(device)
        acc_map = torch.zeros((NC, NC))

        for s_trans in range(NC):
            images = images_all[s_trans].to(device)
            with torch.no_grad():
                if args.mode == 'pert':
                    images_perturbed = torch.clamp(images + pert, min=0, max=1)
                elif args.mode == 'patch':
                    images_perturbed = torch.clamp(images * (1 - mask) + pattern * mask, min=0, max=1)
                _, _, outputs = model(images_perturbed)
                _, predicted = outputs.max(1)
            freq = torch.zeros((NC,))
            predicted = predicted.cpu()
            for i in range(len(freq)):
                freq[i] = len(np.where(predicted == i)[0])
            freq[s_trans] = 0
            if s_trans == s:
                freq[t] = 0
            acc_map[s_trans, :] = freq / NI
        acc_map = acc_map.detach().cpu().numpy()
        torch.save(acc_map, os.path.join(ckpt_path, 'color_map_{}_{}'.format(s, t)))



