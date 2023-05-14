#!/bin/bash


python fedntd.py configs/fedntd1.yaml -s 72 -e 50 > logfiles/fedntd1_72.log

python fedavg.py configs/fedavg1.yaml -s 72 -e 50 > logfiles/fedavg1_72.log

python fedprox.py configs/fedprox1.yaml -s 72 -e 50 > logfiles/fedprox1_72.log

python scaffold.py configs/scaffold1.yaml -s 72 -e 50 > logfiles/scaffold1_72.log
