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
    return global_model


    
if __name__ == "__main__":
    pass