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
from collections import OrderedDict
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
        
class NTDLoss(nn.Module):
    def __init__(self, nclass, beta = 0.1):
        super(NTDLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.nclass = nclass
        self.beta = beta
        
    def forward(self, client_log, server_log, target):
        ce_loss = self.ce(client_log, target)
        client_ntd = self._ntd_logits(client_log, target)
        server_ntd = self._ntd_logits(server_log, target)
        kl_div = self.kl(
            F.log_softmax(client_ntd, dim =1),
            F.softmax(server_ntd, dim =1)
        )
        return ce_loss + self.beta * kl_div
    
    def _ntd_logits(self, log, tar):
        ntd_log = torch.arange(0, self.nclass).to(log.device)
        ntd_log = ntd_log.repeat(log.shape[0], 1)
        ntd_log = ntd_log[ntd_log[:,:] != tar.reshape(-1,1)]
        ntd_log = ntd_log.reshape(-1, self.nclass - 1)
        
        final_ntd_log = torch.gather(log, 1, ntd_log) 
        
        return final_ntd_log
         
class Client():
    def __init__(self, mdl_name, train_loader, distribution, config):
        self.lr = config['lr']
        self.nc = config['nclass']
        self.mom = config['momentum']
        self.adv = config['adv']
        self.return_logs = config['return_logs']
        self.beta = config['beta']
        self.distri = distribution
        self.train_loader = train_loader
        self.mdl = return_model(mdl_name, self.nc)
        self.lossfn = NTDLoss(self.nc, self.beta)
        self.serv_mdl = return_model(mdl_name, self.nc)
        
    def train_client(self, transformations, n_epochs, device): 
        self.mdl = self.mdl.to(device)
        self.serv_mdl = self.serv_mdl.to(device)
        self.opt = optim.SGD(self.mdl.parameters(), lr=self.lr, momentum=self.mom)
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
                
                with torch.no_grad():
                    server_scr = self.serv_mdl(data)
                
                loss = self.lossfn(scores, server_scr, target)            

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
            
        self.mdl = copy.deepcopy(self.mdl)
        
        return tval 
    
    def replace_mdl(self, server_mdl):
        self.mdl = copy.copy(server_mdl)
        self.serv_mdl = copy.copy(server_mdl)
    
class Server():
    def __init__(self, config, device):
        self.nc = config['nclass']
        self.mdl = return_model(config['model'], self.nc)
        self.device = device
        
    def aggregate_models(self, clients_model):
        update_state = OrderedDict()
        n_clients = len(clients_model)
        for k, client in enumerate(clients_model):
            local_state = client.state_dict()
            for key in self.mdl.state_dict().keys():
                if k == 0:
                    update_state[key] = local_state[key] / n_clients 
                else:
                    update_state[key] += local_state[key] / n_clients
      
        print(self.mdl.load_state_dict(update_state))
        
class FedNTD():
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
        start_time = time.perf_counter()
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
                [self.clients[i].mdl for i in clients_selected]
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
            
        end_time = time.perf_counter()
        elapsed_time = int(end_time - start_time)
        hr = elapsed_time // 3600
        mi = (elapsed_time - hr * 3600) // 60
        print(f"training done in {hr} H {mi} M")
            
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
    
    print("environment: ")
    print(f"YAML: {sys.argv[1]}")
    for key, value in config.items():
        print(f"==> {key}: {value}")
            
    client_data, distri = load_dataset(config)
    test_data = load_dataset_test(config)

    transformations = transforms.Compose([])
    fedavg = FedNTD(
        client_data,
        distri,
        test_data,
        config
    )
    fedavg.train(transformations)