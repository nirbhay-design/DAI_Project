import yaml
from yaml.loader import SafeLoader
import torch
import torchvision
import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
import random
import time,json
import copy,sys
import math
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report,auc,roc_curve,precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")
        
        
def load_dataset(config):
    """
    dataset_name
    pin_memory
    n_clients
    n_workers
    batch_size
    """
    
    each_client_dataloader = []
    
    dataset = config['dataset']
    dataset_path = config['dataset_path']
    pin_memory = config['pin_memory']
    n_clients = config['n_clients']
    n_workers = config['n_workers']
    img_size = config['img_size']
    batch_size = config['batch_size']
    iid = config['iid']
    
    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor()
    ])
    
    if dataset == 'cifar10':
        if config.get("augment", False):
            print("augmenting dataset")
            transform = torchvision.transforms.Compose([
                transform,
                torchvision.transforms.RandomHorizontalFlip(0.6)
            ])
        if config.get('standardize',False):
            print("standardizing dataset")
            normalize_transform = torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2470, 0.2435, 0.2616])
            transform = torchvision.transforms.Compose([
                transform,
                normalize_transform
            ])
        train_data = torchvision.datasets.CIFAR10(dataset_path,train=True,download=True,transform=transform)
        
    elif dataset == "cifar100":
        if config.get("augment", False):
            print("augmenting dataset")
            transform = torchvision.transforms.Compose([
                transform,
                torchvision.transforms.RandomRotation(10),
                torchvision.transforms.RandomHorizontalFlip(0.6)
            ])
        if config.get('standardize',False):
            print("standardizing")
            normalize_transform = torchvision.transforms.Normalize([0.5071, 0.4867, 0.4408],[0.2675, 0.2565, 0.2761])
            transform = torchvision.transforms.Compose([
                transform,
                normalize_transform
            ])
        train_data = torchvision.datasets.CIFAR100(dataset_path,train=True,download=True,transform=transform)
             
    split_data = len(train_data)
    print(len(train_data))
    
    client_distribution = None
    
    if iid:
        print("iid data loading")
        each_client_data = split_data // n_clients
        non_uniform = split_data % n_clients
        clients_list = [each_client_data for i in range(n_clients)]
        clients_list[-1] = clients_list[-1]+non_uniform
        print(clients_list)
        clients_list = torch.tensor(clients_list)
        client_distribution = copy.copy(clients_list/torch.sum(clients_list))
        each_client_data = torch.utils.data.random_split(train_data, clients_list)
    
    else:
        print("non iid data loading")
        beta = config['beta']
        client_list = torch.tensor(beta).repeat(n_clients)
        non_iid_dirichlet = (torch.distributions.dirichlet.Dirichlet(client_list).sample()*split_data).type(torch.int64)
        remaining_data = split_data - non_iid_dirichlet.sum()
        non_iid_dirichlet[-1] += remaining_data
        print(non_iid_dirichlet)
        client_distribution = non_iid_dirichlet/torch.sum(non_iid_dirichlet)
        each_client_data = torch.utils.data.random_split(train_data,non_iid_dirichlet)
        
    for i in range(n_clients):
        ci_dataloader = torch.utils.data.DataLoader(
            each_client_data[i],
            shuffle=True,
            batch_size = batch_size,
            pin_memory=pin_memory,
            num_workers = n_workers
        )
        each_client_dataloader.append(ci_dataloader)
    
    return each_client_dataloader, client_distribution

def load_dataset_test(config):
    """
    dataset_name
    pin_memory
    n_workers
    batch_size
    img_size
    """
    
    each_client_dataloader = []
    
    dataset = config['test_dataset']
    dataset_path = config['test_dataset_path']
    pin_memory = config['pin_memory']
    n_workers = config['n_workers']
    img_size = config['img_size']
    batch_size = config['batch_size']
    
    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor()
    ])
    
    if dataset == 'cifar10':
        if config.get('standardize',False):
            normalize_transform = torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2470, 0.2435, 0.2616])
            transform = torchvision.transforms.Compose([
                transform,
                normalize_transform
            ])
        test_data = torchvision.datasets.CIFAR10(dataset_path,train=False,download=True,transform=transform)
    
    elif dataset == "cifar100":
        if config.get('standardize',False):
            normalize_transform = torchvision.transforms.Normalize([0.5071, 0.4867, 0.4408],[0.2675, 0.2565, 0.2761])
            transform = torchvision.transforms.Compose([
                transform,
                normalize_transform
            ])
        test_data = torchvision.datasets.CIFAR100(dataset_path,train=False,download=True,transform=transform)
        
    test_loader = torch.utils.data.DataLoader(
        test_data,
        shuffle=True,
        batch_size = batch_size,
        pin_memory=pin_memory,
        num_workers = n_workers
    )
    
    return test_loader

__all__ = ['wrn']


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.dropout = nn.Dropout( dropout_rate )
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        out = self.dropout(out)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropout_rate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropout_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropout_rate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropout_rate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropout_rate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropout_rate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropout_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, return_features=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, (1,1))
        features = out.view(-1, self.nChannels)
        out = self.fc(features)

        if return_features:
            return out, features
        else:
            return out

def wrn_16_1(num_classes, dropout_rate=0):
    return WideResNet(depth=16, num_classes=num_classes, widen_factor=1, dropout_rate=dropout_rate)

def wrn_16_2(num_classes, dropout_rate=0):
    return WideResNet(depth=16, num_classes=num_classes, widen_factor=2, dropout_rate=dropout_rate)

def wrn_40_1(num_classes, dropout_rate=0):
    return WideResNet(depth=40, num_classes=num_classes, widen_factor=1, dropout_rate=dropout_rate)

def wrn_40_2(num_classes, dropout_rate=0):
    return WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropout_rate=dropout_rate)


def load_model(model_name, nclass, channel=3, pretrained=True):
    if model_name == 'Resnet18':
        global_model = torchvision.models.resnet18(pretrained=pretrained)
        global_model.conv1 = nn.Conv2d(channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        global_model.fc = nn.Linear(in_features=512, out_features=nclass, bias=True)
    elif model_name == "Resnet34":
        global_model = torchvision.models.resnet34(pretrained=pretrained)
        global_model.conv1 = nn.Conv2d(channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        global_model.fc = nn.Linear(in_features=512, out_features=nclass, bias=True)
    elif model_name == 'Resnet50':
        global_model = torchvision.models.resnet50(pretrained=pretrained)
        global_model.conv1 = nn.Conv2d(channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        global_model.fc = nn.Linear(in_features=2048, out_features=nclass, bias=True)
    elif model_name == "Mobilenetv2":
        global_model = torchvision.models.mobilenet_v2(pretrained=pretrained)
        global_model.features[0][0] = nn.Conv2d(channel, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        global_model.classifier[1] = nn.Linear(in_features = 1280, out_features = nclass, bias = True)
    elif model_name == "Mobilenetv3":
        global_model = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
        global_model.features[0][0] = nn.Conv2d(channel, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        global_model.classifier[3] = nn.Linear(in_features = 1024, out_features = nclass, bias = True)
    elif model_name == "Shufflenet":
        global_model = torchvision.models.shufflenet_v2_x1_0(pretrained = pretrained)
        global_model.conv1[0] = nn.Conv2d(channel, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        global_model.fc = nn.Linear(in_features = 1024, out_features = nclass, bias=True)
    elif model_name == "WRN_40":
        global_model = wrn_40_1(nclass)
    return global_model


    
if __name__ == "__main__":
    pass