#!/bin/bash

python fedprox.py configs/fedprox_amp.yaml > logfiles/fedprox_amp.log
python fedavg.py configs/fedavg_amp.yaml > logfiles/fedavg_amp.log