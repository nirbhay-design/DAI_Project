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
        self.ci = {
            name: torch.zeros_like(params)
            for name, params in self.mdl.named_parameters()
        }
        self.c = {
            name: torch.zeros_like(params)
            for name, params in self.mdl.named_parameters()
        }
        self.serv_mdl = return_model(mdl_name, self.nc)
        self.cpu = torch.device('cpu')
        
        if self.adv:
            self.epsilon = config['epsilon']
            self.inner_lr = config['inner_lr']
            self.inner_iter = config['inner_iter']
            self.train_client = self._train_client_adv
        else:
            self.train_client = self._train_client
        
    def _train_client(self, transformations, n_epochs, device): 
        self.put_to_device(device)
        c_diff = self.c_diff()
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
                loss = self.lossfn(scores,target)            

                cur_loss += loss.item() / (len_train)
                scores = F.softmax(scores,dim = 1)
                _,predicted = torch.max(scores,dim = 1)
                correct = (predicted == target).sum()
                samples = scores.shape[0]
                curacc += correct / (samples * len_train)

                self.opt.zero_grad()
                loss.backward()
                
                for name, params in self.mdl.named_parameters():
                    params.grad += c_diff[name]

                self.opt.step()

                if self.return_logs:
                    progress(idx+1,len_train)

            tval['trainacc'].append(float(curacc))
            tval['trainloss'].append(float(cur_loss))

            print(f"epochs: [{epochs+1}/{n_epochs}] train_acc: {curacc:.3f} train_loss: {cur_loss:.3f}")
            
        self.mdl = copy.deepcopy(self.mdl)
        
        
        self.mdl_diff = self.diff_mdl()
        self.c, self.diff_c = self.update_c(c_diff, self.mdl_diff, n_epochs)
        
        self.off_device()
        
        return tval 
    
    def _train_client_adv(self, transformations, n_epochs, device): 
        self.put_to_device(device)
        c_diff = self.c_diff()
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
                    loss = self.lossfn(scores, target)
                    loss.backward()
                    for name, params in self.mdl.named_parameters():
                        params.grad += c_diff[name]
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
        
        self.mdl_diff = self.diff_mdl()
        self.c, self.diff_c = self.update_c(c_diff, self.mdl_diff, n_epochs)
        
        self.off_device()
        
        return tval
    
    def put_to_device(self, device):
        for name, params in self.c.items():
            self.c[name] = self.c[name].to(device)
            self.ci[name] = self.ci[name].to(device)
            
    def off_device(self):
        for name, params in self.c.items():
            self.c[name] = self.c[name].to(self.cpu)
            self.ci[name] = self.ci[name].to(self.cpu)
    
    def c_diff(self):
        # c - c_i
        c_diff = OrderedDict()
        for name, params in self.c.items():
            c_diff[name] = self.c[name] - self.ci[name]
        return c_diff
    
    def diff_mdl(self):
        # x - y_i
        mdl_diff = OrderedDict()
        server_parms = dict(self.serv_mdl.named_parameters())
        for name, params in self.mdl.named_parameters():
            mdl_diff[name] = params - server_parms[name]
        return mdl_diff
    
    @torch.no_grad()
    def update_c(self, c_diff, mdl_diff, K):
        # c+
        alpha = 1 / (K * self.lr)
        
        update_c = OrderedDict()
        diff_c = OrderedDict()
        for name, params in c_diff.items():
            val = alpha * mdl_diff[name]
            update_c[name] = val - c_diff[name]
            diff_c[name] = val - self.c[name]
        return update_c, diff_c
    
    def replace_mdl(self, server_mdl, server_c):
        self.mdl = copy.copy(server_mdl)
        self.serv_mdl = copy.copy(server_mdl)
        self.c = server_c
    
    
class Server():
    def __init__(self, config, device):
        self.nc = config['nclass']
        self.total_clients = config['n_clients']
        self.mdl = return_model(config['model'], self.nc)
        self.device = device
        self.c = {
            name: torch.zeros_like(params)
            for name, params in self.mdl.named_parameters()
        }
        self.lr = 0.001
        
        self.put_to_device(self.device)
        
    def put_to_device(self, device):
        self.mdl = self.mdl.to(device)
        # for name, params in self.c.items():
        #     self.c[name] = self.c[name].to(device)
    
    @torch.no_grad()
    def aggregate_models(self, clients_model, c_diff):
        update_state = OrderedDict()
        avg_cv = OrderedDict()
        
        n_clients = len(clients_model)

        for k, client in enumerate(clients_model):
            for key in client.keys():
                if k == 0:
                    update_state[key] = client[key] / n_clients 
                else:
                    update_state[key] += client[key] / n_clients
                    
        for k, cv_diff in enumerate(c_diff):
            for name, params in cv_diff.items():
                if k == 0:
                    avg_cv[name] = cv_diff[name]
                else:
                    avg_cv[name] += cv_diff[name]
        
        for name, params in avg_cv.items():
            self.c[name] +=  avg_cv[name] / self.total_clients
            
        for name, params in self.mdl.named_parameters():
            params = params - self.lr * update_state[name]
        
    def aggregate_models1(self, clients_model, c_diff):
        update_state = OrderedDict()
        avg_cv = OrderedDict()
        final_state = OrderedDict()
        
        n_clients = len(clients_model)
        mdl_state_dict = self.mdl.state_dict()

        for k, client in enumerate(clients_model):
            for key in mdl_state_dict.keys():
                if k == 0:
                    update_state[key] = client[key] / n_clients 
                else:
                    update_state[key] += client[key] / n_clients
                    
        for k, cv_diff in enumerate(c_diff):
            for name, params in cv_diff.items():
                if k == 0:
                    avg_cv[name] = cv_diff[name] / n_clients
                else:
                    avg_cv[name] += cv_diff[name] / n_clients
        
        for name, params in avg_cv.items():
            self.c[name] += (n_clients / self.total_clients) * avg_cv[name]
            
        for name, params in mdl_state_dict.items():
            final_state[name] = mdl_state_dict[name] - self.lr * update_state[name]
      
        print(self.mdl.load_state_dict(final_state))
        
class SCAFFOLD():
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
            
            for pdx in clients_selected:
                self.clients[pdx].replace_mdl(self.server.mdl, self.server.c)
                
            for jdx in clients_selected:
                print(f"############## client {jdx} ##############")
                self.clients[jdx].train_client(
                    transformations,
                    self.clientiter,
                    self.device,
                )

            print("############## server ##############")
            self.server.aggregate_models(
                [self.clients[i].mdl_diff for i in clients_selected],
                [self.clients[i].diff_c for i in clients_selected]
            )

            single_acc = evaluate_single_server(
                self.config,
                self.server.mdl,
                self.test_data,
                transformations,
                self.device
            )
            
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
    fedavg = SCAFFOLD(
        client_data,
        distri,
        test_data,
        config
    )
    fedavg.train(transformations)
