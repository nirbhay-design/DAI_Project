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
from ampreg import AMP
import warnings
warnings.filterwarnings("ignore")


def yaml_loader(yaml_file):
    with open(yaml_file,'r') as f:
        config_data = yaml.load(f,Loader=SafeLoader)
        
    args = sys.argv
        
    if '-s' in args:
        config_data['SEED'] = int(args[args.index('-s') + 1]) 
    
    if '-e' in args:
        config_data['total_iterations'] = int(args[args.index('-e') + 1])

    if '-b' in args:
        config_data['beta'] = float(args[args.index('-b') + 1])

    if '-nc' in args:
        config_data['n_clients'] = int(args[args.index('-nc') + 1])
    
    if '-sc' in args:
        config_data['sample_clients'] = float(args[args.index('-sc') + 1])

    if '-rl' in args:
        config_data['return_logs'] = eval(args[args.index('-rl') + 1])

    if '-ci' in args:
        config_data['client_iterations'] = int(args[args.index('-ci') + 1])

    if '-clr' in args:
        config_data['lr'] = float(args[args.index('-clr') + 1])

    if '-g' in args:
        gpu_id = int(args[args.index('-g') + 1])
        config_data['gpu'] = gpu_id
        config_data['server_gpu'] = gpu_id
        
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
        self.lossfn = nn.CrossEntropyLoss()
        self.serv_mdl = return_model(mdl_name, self.nc)
        self.cpu = torch.device('cpu')
        self.alpha = config['alpha']
        
        if self.adv:
            self.epsilon = config['epsilon']
            self.inner_lr = config['inner_lr']
            self.inner_iter = config['inner_iter']
            self.train_client = self._train_client_adv
        else:
            self.train_client = self._train_client
        
    def _train_client(self, transformations, n_epochs, device): 
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
                loss = self.lossfn(scores,target) + self.prox_reg()         

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
    
    def _train_client_adv(self, transformations, n_epochs, device): 
        self.mdl = self.mdl.to(device)
        self.serv_mdl = self.serv_mdl.to(device)
        self.opt = AMP(params=filter(lambda p: p.requires_grad, self.mdl.parameters()),
                        lr=self.lr,
                        epsilon=self.epsilon,
                        inner_lr=self.inner_lr,
                        inner_iter=self.inner_iter,
                        base_optimizer=torch.optim.SGD,
                        momentum=self.mom,
                        weight_decay=0.0,
                        nesterov=True)
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
                
                def closure():
                    self.opt.zero_grad()
                    scores = self.mdl(data)
                    loss = self.lossfn(scores, target) + self.prox_reg()
                    loss.backward()
                    return scores, loss 
                
                scores, loss = self.opt.step(closure)

                cur_loss += loss.item() / (len_train)
                scores = F.softmax(scores,dim = 1)
                _,predicted = torch.max(scores,dim = 1)
                correct = (predicted == target).sum()
                samples = scores.shape[0]
                curacc += correct / (samples * len_train)

                if self.return_logs:
                    progress(idx+1,len_train)

            tval['trainacc'].append(float(curacc))
            tval['trainloss'].append(float(cur_loss))

            print(f"epochs: [{epochs+1}/{n_epochs}] train_acc: {curacc:.3f} train_loss: {cur_loss:.3f}")
            
        self.mdl = copy.deepcopy(self.mdl)
        
        return tval
    
    def replace_mdl(self, server_mdl):
        self.mdl = copy.deepcopy(server_mdl)
        self.serv_mdl = copy.deepcopy(server_mdl)
        
    @torch.no_grad()
    def prox_reg(self):
        params1 = dict(self.mdl.named_parameters())
        params2 = dict(self.serv_mdl.named_parameters())
        
        loss_val = 0
        
        for name, params in params1.items():
            norm_val = torch.norm(params1[name] - params2[name]) ** 2
            loss_val += self.alpha * 0.5 * norm_val
            
        return loss_val

    def cpu_model(self, cpu_device):
        self.mdl = self.mdl.to(cpu_device)
        self.serv_mdl = self.serv_mdl.to(cpu_device)

    
class Server():
    def __init__(self, config, device):
        self.nc = config['nclass']
        self.mdl = return_model(config['model'], self.nc)
        self.device = device
        
    @torch.no_grad()
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
        
class FedProx():
    def __init__(self, clients_data, distri, test_data, config):
        model_name = config['model']
        self.device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.nclients = config['n_clients']
        self.clientiter = config['client_iterations']
        self.totaliter = config['total_iterations']
        self.test_data = test_data
        self.sample_cli = int(config['sample_clients'] * self.nclients)
        self.cpu = torch.device('cpu')
        
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
                self.clients[pdx].cpu_model(self.cpu)
            
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
    fedavg = FedProx(
        client_data,
        distri,
        test_data,
        config
    )
    fedavg.train(transformations)
