#!/bin/bash

python scaffold.py configs/scaffold_c100.yaml > logfiles/scaffold_c100.log
python fedntd.py configs/fedntd_c100.yaml > logfiles/fedntd_c100.log