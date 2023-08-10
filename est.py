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
# import matplotlib.pyplot as plt
from PIL import Image
import PIL
import random
import copy as cp
import numpy as np
import time
import json
from utils.GTSRB import GTSRB
from utils.model_zoo import ResNet18, SimpleNet
from utils.util import pert_est_class_pair, data_split, pm_est_class_pair
from utils.ImageNette import Imagenette

parser = argparse.ArgumentParser(description='Reverse engineer backdoor pattern')
parser.add_argument("--mode", default="pert", type=str)
parser.add_argument("--RUN", default=-1, type=int)
parser.add_argument("--SETTING", default="", type=str)
parser.add_argument("--ATTACK", default="", type=str)
parser.add_argument("--DATASET", default="", type=str)
parser.add_argument("--DEVICE", default=-1, type=int)
args = parser.parse_args()
with open('config.json') as config_file:
    config = json.load(config_file)


if args.RUN >= 0:
# if args.RUN is not None:
    config["RUN"] = args.RUN
if args.DEVICE >= 0:
    config["DEVICE"] = args.DEVICE
if args.SETTING == "A2A" or args.SETTING == "A2O" or args.SETTING == "rand" or args.SETTING == "x2x":
    config["SETTING"] = args.SETTING
if args.ATTACK == "patch" or args.ATTACK == "perturbation" or args.ATTACK == "CLA" or args.ATTACK == "clean":
    config["PATTERN_TYPE"] = args.ATTACK
if args.DATASET == "cifar10" or args.DATASET == "gtsrb" or args.DATASET == "imagenette":
    config["DATASET"] = args.DATASET
start_time = time.time()
random.seed()
device = config["DEVICE"]
TRIAL = 1
NI = 10
# Create saving path for results
ckpt_path = '{}_estimated/{}/{}/{}/{}'.format(args.mode, config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"])
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
if args.ATTACK != "clean":
    poisoned_pairs = torch.load(os.path.join('./attacks/{}/{}/{}/{}'.format(config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"]), "pairs"))
else:
    poisoned_pairs = []
print("Detect: {}, Dataset: {}, Mode: {}, Type: {},  Run: {}".format(args.mode, config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"]))
# Load clean images for detection
print("Expected pairs are: ")
print(poisoned_pairs)
print('==> Preparing data..')
poisoned_pairs = np.array(poisoned_pairs)
# LR2 = 1e-1
if config["DATASET"] == "cifar10":
    config["NUM_CLASS"] = 10
    PI = 0.9
    if args.mode == "patch":
        TRIAL = 5
        NI = 20
    LR = 1e-5
    transform_test = transforms.Compose([transforms.ToTensor()])
    detectset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
elif config["DATASET"] == "gtsrb":
    config["NUM_CLASS"] = 43
    PI = 0.95
    if args.mode == "patch":
        TRIAL = 3
        NI = 20
    LR = 1e-3
    transform_test = transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
    ])
    detectset = GTSRB(root='./data', split='test', download=False, transform=transform_test)
elif config["DATASET"] == "imagenette":
    config["NUM_CLASS"] = 10
    PI = 0.85
    LR = 1e-4
    if args.mode == "patch":
        TRIAL = 5
        NI = 20
    transform_test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    detectset = Imagenette(root='./data/imagenette2', train=False, transform=transform_test)

# Detection parameters
NC = config["NUM_CLASS"]     # Number of classes
# NI = 20     # Number of images per class used for detection
print("Num trials : {}, Misclassification : {}, # Images: {}".format(TRIAL, PI, NI))
model = ResNet18(num_classes=NC) if config["DATASET"] == "cifar10" or config["DATASET"] == "imagenette" else SimpleNet()
model = model.to(device)
model.load_state_dict(torch.load('./attacks/{}/{}/{}/{}/model_contam.pth'.format(config['DATASET'], config['SETTING'], config['PATTERN_TYPE'],config["RUN"]), map_location=torch.device(device)))
model.eval()
correct_path = os.path.join(ckpt_path, "correct.npy")
target_path = os.path.join(ckpt_path, "targets.npy")
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

np.save(os.path.join(ckpt_path, "correct.npy"), correct)
np.save(os.path.join(ckpt_path, "targets.npy"), targets)
images_all = []
ind_all = []
for c in range(NC):
    ind = [correct[i] for i, label in enumerate(targets) if label == c]
    ind = np.random.choice(ind, NI, replace=False)
    images_all.append(torch.stack([detectset[i][0] for i in ind]))
    ind_all.append(ind)
images_all = [images.to(device) for images in images_all]
np.save(os.path.join(ckpt_path, 'ind.npy'), ind_all)
for s in range(NC):
    for t in range(NC):
        # skip the case where s = t
        if s == t:
            continue
        images = images_all[s]
        labels = (torch.ones((len(images),)) * t).long().to(device)

        # CORE STEP: perturbation esitmation for (s, t) pair
        norm_best = 1000000.
        pattern_best = None
        pert_best = None
        mask_best = None
        rho_best = None
        for trial_run in range(TRIAL):
            if args.mode == "patch":
                pattern, mask, rho = pm_est_class_pair(images=images, model=model, target=t, device=device,labels=labels, pi=PI, batch_size=NI,  verbose=False)
                if torch.abs(mask).sum() < norm_best:
                    norm_best = torch.abs(mask).sum()
                    pattern_best, mask_best, rho_best = pattern, mask, rho
                pattern, mask, rho = pattern_best, mask_best, rho_best
            elif args.mode == "pert":
                pert, rho = pert_est_class_pair(source=s, target=t, model=model, images=images, device=device , labels=labels, pi=PI, lr=LR , init=None, verbose=False)
                if torch.norm(pert) < norm_best:
                    norm_best = torch.norm(pert)
                    pert_best = pert
                    rho_best = rho
            else:
                print('Detection Mode Is Not Supported!')
                sys.exit(0)
        if args.mode == "patch":
            print(s, t, torch.abs(mask).sum().item(), rho)
            torch.save(pattern.detach().cpu(), os.path.join(ckpt_path, 'pattern_{}_{}'.format(s, t)))
            torch.save(mask.detach().cpu(), os.path.join(ckpt_path, 'mask_{}_{}'.format( s, t)))
        elif args.mode == "pert":
            print(s, t, torch.norm(pert).item(), rho)
            torch.save(pert.detach().cpu(), os.path.join(ckpt_path, 'pert_{}_{}'.format(s, t)))
print("--- %s seconds ---" % (time.time() - start_time))
torch.save((time.time() - start_time), os.path.join(ckpt_path, 'time'))
