#!/bin/bash

python fedprox.py configs/fedprox_c100_amp.yaml > logfiles/fedprox_c100_amp.log
python fedavg.py configs/fedavg_c100_amp.yaml > logfiles/fedavg_c100_amp.log