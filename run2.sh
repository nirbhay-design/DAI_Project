#!/bin/bash

python fedprox.py configs/fedprox_c100.yaml > logfiles/fedprox_c100.log
python fedavg.py configs/fedavg_c100.yaml > logfiles/fedavg_c100.log