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

def yaml_loader(yaml_file):
    with open(yaml_file,'r') as f:
        config_data = yaml.load(f,Loader=SafeLoader)
        
    return config_data

def progress(current,total):
    progress_percent = (current * 50 / total)
    progress_percent_int = int(progress_percent)
    print(f" |{chr(9608)* progress_percent_int}{' '*(50-progress_percent_int)}|{current}/{total}",end='\r')
    if (current == total):
        print("\n")
        
def evaluate(model, loader ,n_classes, device, transformations, fta_path=None, return_logs=False):
    correct = 0;samples =0
    fpr_tpr_auc = {}
    pre_prob = []
    lab = []
    predicted_labels = []

    with torch.no_grad():
        loader_len = len(loader)
        for idx,(x,y) in enumerate(loader):
            x = transformations(x)
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            predict_prob = F.softmax(scores,dim=1)
            _,predictions = predict_prob.max(1)

            predictions = predictions.to('cpu')
            y = y.to('cpu')
            predict_prob = predict_prob.to('cpu')

            predicted_labels.extend(list(predictions.numpy()))
            pre_prob.extend(list(predict_prob.numpy()))
            lab.extend(list(y.numpy()))

            correct += (predictions == y).sum()
            samples += predictions.size(0)
        
            if return_logs:
                progress(idx+1,loader_len)
        
        print('correct are {:.3f}'.format(correct/samples))

    lab = np.array(lab)
    predicted_labels = np.array(predicted_labels)
    pre_prob = np.array(pre_prob)
    return fpr_tpr_auc,lab,predicted_labels,pre_prob, correct/samples

def evaluate_single_server(config, server_model, test_loader, transformations, device):
    server_model = server_model.to(device)
    server_model.train()

    if config.get("eval_mode",False) == True:
        print("putting model into eval mode")
        server_model.eval()

    test_fta,y_true,y_pred,prob,acc = evaluate(server_model, test_loader, config['nclass'], device, transformations, return_logs=config['return_logs'])
    
    return acc


if __name__ == "__main__":
    
    pass