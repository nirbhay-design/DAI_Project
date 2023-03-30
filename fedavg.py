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
from inference import evaluate_single_server, evaluate_no_server
from data_model_loading import load_dataset, load_model as return_model, load_dataset_test
import warnings
warnings.filterwarnings("ignore")


def yaml_loader(yaml_file):
    with open(yaml_file,'r') as f:
        config_data = yaml.load(f,Loader=SafeLoader)
        
    return config_data

def progress(current,total):
    progress_percent = (current * 50 / total)
    progress_percent_int = int(progress_percent)
    print(f" |{chr(9608)* progress_percent_int}{' '*(50-progress_percent_int)}|{current}/{total}",end='\r')
    if (current == total):
        print()
        
        
class Client():
    def __init__(self, mdl_name, distribution, config):
        self.lr = config['lr']
        self.opt = config['opt']
        self.nc = config['nclass']
        self.mom = config['momentum']
        self.adv = config['adv']
        self.mdl = return_model(mdl_name, self.nc)
        self.opt = optim.SGD(self.mdl.parameters(), lr=self.lr, momentum=self.mom)
        self.lossfn = nn.CrossEntropyLoss()