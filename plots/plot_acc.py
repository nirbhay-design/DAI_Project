import matplotlib.pyplot as plt
import sys
import os

file_name = sys.argv[1]
files = sys.argv[2:]

def plot_acc(dict_val):
    plt.figure(figsize=(5,4))
    for key in dict_val.keys():
        plt_vals = dict_val[key]
        plt.plot(list(range(1,len(plt_vals)+1)), plt_vals, label=key)
    plt.legend()
    plt.xlabel('# of Rounds')
    plt.ylabel('Test Acc')
    plt.title('Acc vs Rounds')
    plt.xlim(0,100)
    plt.savefig(file_name)

def read_file(file_name):
    data_pts = []
    with open(file_name, 'r') as f:
        data = f.readlines()
        for i in data:
            if 'cur_acc' in i:
                data_pts.append(float(i.split(':')[-1])*100)
                
    return data_pts
                
        
if __name__ == "__main__":
    data_path = '../logfiles'
    data_dict = {}
    
    for file in files:
        data = read_file(os.path.join(data_path,file))
        if 'fedavg' in file:
            data_dict['FedAVG'] = data
            
        elif 'fedprox' in file:
            data_dict['FedProx'] = data
            
        elif 'fedntd' in file:
            data_dict['FedNTD'] = data
            
        elif 'scaffold' in file:
            data_dict['SCAFFOLD'] = data
            
    plot_acc(data_dict)
        