# -*- coding: utf-8 -*-  

"""
Created on 2021/12/21

@author: Ruoyu Chen
"""

import argparse
import os
import numpy as np

import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from dataset import VGGFace2Dataset
from models.iresnet import iresnet50, iresnet100
from models.metrics import ArcFace, CosFace, FocalLoss

from tqdm import tqdm

def define_backbone(net_type, nun_classes):
    """
    define the model
    """
    if net_type == "resnet50":
        backbone = iresnet50(nun_classes)
    elif net_type == "resnet100":
        backbone = iresnet100(nun_classes)
    return backbone

def define_metric(metric_type):
    if metric_type == "arcface":
        metric = ArcFace()
    elif metric_type == "cosface":
        metric = CosFace()
    return metric

def eval_model(backbone, validation_loader, device):
    backbone.eval()

    acc = 0

    with torch.no_grad():
        for ii, (data,label) in tqdm(enumerate(validation_loader)):
            data = data.to(device)
            label = label.to(device).long()
            
            output = backbone.identification(data)

            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()

            acc += np.sum((output==label).astype(int))
    acc = acc / len(validation_loader.dataset)
    return acc

def train(args):
    """
    Train the network
    """
    device = torch.device(args.device)

    # Dataloader
    train_dataset = VGGFace2Dataset(dataset_root=args.dataset_root,dataset_list=args.train_list)
    validation_dataset = VGGFace2Dataset(dataset_root=args.dataset_root,dataset_list=args.val_list)

    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    validation_loader = DataLoader(validation_dataset,batch_size=args.batch_size,shuffle=True)

    backbone = define_backbone(args.backbone, args.num_classes)
    metric = define_metric(args.metric)      # ArcFace etc.

    if args.pre_trained is not None and os.path.exists(args.pre_trained):
        # No related
        try:
            backbone.load_state_dict(torch.load(args.pre_trained))
            print("Success load pre-trained model {}".format(args.pre_trained))
        except:
            print("Failed to load pre-trained model {}".format(args.pre_trained))

    # Set to device
    backbone.to(device)
    metric.to(device)

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    if args.opt == "sgd":
        optimizer = torch.optim.SGD([{'params':backbone.parameters()}],
                lr = args.lr / 512 * args.batch_size, weight_decay = 0.01)
    elif args.opt == "adam":
        optimizer = torch.optim.Adam([{'params':backbone.parameters()}],
                lr = args.lr / 512 * args.batch_size, weight_decay = 0.01)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for i in range(1,args.epoch+1):
        scheduler.step()

        backbone.train()
        for ii, (data,label) in enumerate(train_loader):
            # Load data
            data = Variable(data).to(device)
            label = Variable(label).to(device).long()

            # Get the output
            cosine = backbone.identification(data)
            # Get the id cls
            output = metric(cosine, label)

            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i* len(train_loader) + ii

            # View the train process
            if iters % 100 == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()

                acc = np.mean((output==label).astype(int))

                print("train epoch {} iter {} loss {} acc {}.".format(i,ii,loss.item(),acc))

        # Save model
        if i % 10 == 0:
            torch.save(backbone, "./checkpoint/backbone-item-epoch-"+str(i)+'.pth')

def parse_args():
    parser = argparse.ArgumentParser(description='Train paramters.')
    # general
    parser.add_argument('--dataset-root', type=str,
                        default='/home/cry/data2/VGGFace2/train_align_arcface',
                        help='')
    parser.add_argument('--train-list', type=str,
                        default='./data/train.txt',
                        help='')
    parser.add_argument('--val-list', type=str,
                        default='./data/val.txt',
                        help='')
    parser.add_argument('--num-classes', type=int,
                        default=8631,
                        help='')
    parser.add_argument('--backbone', type=str,
                        default='resnet50',
                        choices=['resnet50','resnet100'],
                        help='')
    parser.add_argument('--pre-trained', type=str,
                        default='./pretrained/ms1mv3_arcface_r50_fp16/backbone.pth',
                        help='')
    parser.add_argument('--metric', type=str,
                        default='arcface',
                        choices=['arcface','cosface'],
                        help='')
    parser.add_argument('--batch-size', type=int,
                        default=256,
                        help='')
    parser.add_argument('--epoch', type=int,
                        default=100,
                        help='')
    parser.add_argument('--lr', type=float,
                        default=0.1,
                        help='')
    parser.add_argument('--opt', type=str,
                        default='sgd',
                        choices=['sgd','adam'],
                        help='')
    parser.add_argument('--device', type=str,
                        default='cuda',
                        choices=['cuda','cpu'],
                        help='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)