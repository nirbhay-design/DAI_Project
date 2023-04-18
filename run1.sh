#!/bin/bash

python scaffold.py configs/scaffold_c100_amp.yaml > logfiles/scaffold_c100_amp.log
python fedntd.py configs/fedntd_c100_amp.yaml > logfiles/fedntd_c100_amp.log