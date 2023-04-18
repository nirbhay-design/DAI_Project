#!/bin/bash


python plot_acc.py c10_10.png fedavg.log fedprox.log fedntd.log scaffold.log
python plot_acc.py c10_30.png fedavg1.log fedprox1.log fedntd1.log scaffold1.log
python plot_acc.py c100_30.png fedavg_c100.log fedprox_c100.log fedntd_c100.log scaffold_c100.log
python plot_acc.py c10_30_amp.png fedavg_amp.log fedprox_amp.log fedntd_amp.log scaffold_amp.log
python plot_acc.py c100_30_amp.png fedavg_c100_amp.log fedprox_c100_amp.log fedntd_c100_amp.log scaffold_c100_amp.log