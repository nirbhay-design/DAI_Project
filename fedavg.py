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
from inference import evaluate_single_server
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
    def __init__(self, mdl_name, train_loader, distribution, config):
        self.lr = config['lr']
        self.nc = config['nclass']
        self.mom = config['momentum']
        self.adv = config['adv']
        self.return_logs = config['return_logs']
        self.distri = distribution
        self.train_loader = train_loader
        self.mdl = return_model(mdl_name, self.nc)
        self.opt = optim.SGD(self.mdl.parameters(), lr=self.lr, momentum=self.mom)
        self.lossfn = nn.CrossEntropyLoss()
        
    def train_client(self, transformations, n_epochs, device): 
        self.mdl = self.mdl.to(device)
        tval = {'trainacc':[],"trainloss":[]}
        self.mdl.train()
        len_train = len(self.train_loader)
        for epochs in range(n_epochs):
            cur_loss = 0
            curacc = 0
            for idx , (data,target) in enumerate(self.train_loader):
                if data.shape[0] == 1:
                    continue
                data = transformations(data)    
                data = data.to(device)
                target = target.to(device)

                scores = self.mdl(data)    
                loss = self.lossfn(scores,target)            

                cur_loss += loss.item() / (len_train)
                scores = F.softmax(scores,dim = 1)
                _,predicted = torch.max(scores,dim = 1)
                correct = (predicted == target).sum()
                samples = scores.shape[0]
                curacc += correct / (samples * len_train)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if self.return_logs:
                    progress(idx+1,len_train)

            tval['trainacc'].append(float(curacc))
            tval['trainloss'].append(float(cur_loss))

            print(f"epochs: [{epochs+1}/{n_epochs}] train_acc: {curacc:.3f} train_loss: {cur_loss:.3f}")
            
        return tval 
    
    def replace_mdl(self, server_mdl):
        self.mdl = copy.deepcopy(server_mdl)
    
    
class Server():
    def __init__(self, config, device):
        self.nc = config['nclass']
        self.mdl = return_model(config['model'], self.nc)
        self.device = device
    
    def aggregate_models(self, clients_model, distribution):
        aggregated_model = {name:params * distribution[0] for name,params in clients_model[0].state_dict().items()}

        for idx, client_mdl in enumerate(clients_model[1:]):
            idx_clt_sd = client_mdl.state_dict()
            for name, param in idx_clt_sd.items():
                aggregated_model[name] += param * distribution[idx+1]
                
        print(self.mdl.load_state_dict(aggregated_model))
        
#     def aggregate_models(self, clients_model, distribution):
#         n_clients = len(clients_model)
#         params_dict = {}
#         server_params = {**self.mdl.state_dict()}
#         for name, params in server_params.items():
#             params_dict[name] = torch.zeros_like(server_params[name])
#             for idx, client in enumerate(clients_model):
#                 client_dict = client.state_dict()
#                 if params_dict[name].dtype == torch.int64:
#                     params_dict[name] += client_dict[name] // n_clients
#                 else:    
#                     params_dict[name] += client_dict[name] * distribution[idx]

#         print(self.mdl.load_state_dict(params_dict))
        
class FedAvg():
    def __init__(self, clients_data, distri, test_data, config):
        model_name = config['model']
        self.device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.nclients = config['n_clients']
        self.clientiter = config['client_iterations']
        self.totaliter = config['total_iterations']
        self.test_data = test_data
        self.sample_cli = int(config['sample_clients'] * self.nclients)
        
        self.clients = []
        
        for i in range(self.nclients):
            cur_client = Client(
                model_name,
                clients_data[i],
                distri[i],
                config
            )
            self.clients.append(cur_client)
            
        self.server = Server(config, self.device)
        
    def train(self, transformations):
        for idx in range(self.totaliter):
            print(f"iteration [{idx+1}/{self.totaliter}]")
            clients_selected = random.sample([i for i in range(self.nclients)], self.sample_cli)
            distribution = [self.clients[i].distri for i in clients_selected]
            for jdx in clients_selected:
                print(f"############## client {jdx} ##############")
                self.clients[jdx].train_client(
                    transformations,
                    self.clientiter,
                    self.device,
                )

            print("############## server ##############")
            self.server.aggregate_models(
                [self.clients[i].mdl for i in clients_selected],
                distribution,
            )

            single_acc = evaluate_single_server(
                self.config,
                self.server.mdl,
                self.test_data,
                transformations,
                self.device
            )
            
            for pdx in clients_selected:
                self.clients[pdx].replace_mdl(self.server.mdl)
            
            print(f'cur_acc: {single_acc.item():.3f}')
            
if __name__ == "__main__":
    
    config = yaml_loader(sys.argv[1])
    
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed(config["SEED"])
    torch.backends.cudnn.benchmarks = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True    
            
    client_data, distri = load_dataset(config)
    test_data = load_dataset_test(config)

    transformations = transforms.Compose([])
    fedavg = FedAvg(
        client_data,
        distri,
        test_data,
        config
    )
    fedavg.train(transformations)