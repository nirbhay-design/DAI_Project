#!/bin/bash


python fedavg.py configs/fedavg1.yaml -s 52 -e 50 > logfiles/fedavg1_52.log

python fedntd.py configs/fedntd1.yaml -s 52 -e 50 > logfiles/fedntd1_52.log

python scaffold.py configs/scaffold1.yaml -s 52 -e 50 > logfiles/scaffold1_52.log

python fedprox.py configs/fedprox1.yaml -s 52 -e 50 > logfiles/fedprox1_52.log
