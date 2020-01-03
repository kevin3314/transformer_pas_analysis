#!/usr/bin/env bash

set -exu

gpu=2
seed=56567

# python src/train.py -c config/BaselineModel/kwdlc/3e/large-coref-ocz-nocase-noun.json -d $gpu --seed $seed

latest_checkpoint=$(ls result/BaselineModel-kwdlc-3e-large-coref-ocz-nocase-noun/*/checkpoint-epoch3.pth | sort | tail -1)
python src/train.py -r $latest_checkpoint -c config/BaselineModel/kwdlc/6e/large-coref-cz-nocase-noun.json -d $gpu --seed $seed

latest_checkpoint=$(ls result/BaselineModel-kwdlc-6e-large-coref-cz-nocase-noun/*/checkpoint-epoch6.pth | sort | tail -1)
python src/test.py -r $latest_checkpoint -d $gpu
