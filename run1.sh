#!/bin/bash

python scaffold.py configs/scaffold_amp.yaml > logfiles/scaffold_amp.log
python fedntd.py configs/fedntd_amp.yaml > logfiles/fedntd_amp.log