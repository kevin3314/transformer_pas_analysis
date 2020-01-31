#!/usr/bin/env bash

set -exu

gpu=3

python src/train.py -c config/BaselineModel/kwdlc/3e/large-coref-o.json -d $gpu

latest_checkpoint=$(ls result/BaselineModel-kwdlc-3e-large-coref-o/*/checkpoint-epoch3.pth | sort | tail -1)
python src/train.py -r $latest_checkpoint -c config/BaselineModel/kwdlc/6e/large-coref-oc.json -d $gpu

latest_checkpoint=$(ls result/BaselineModel-kwdlc-6e-large-coref-oc/*/checkpoint-epoch6.pth | sort | tail -1)
python src/train.py -r $latest_checkpoint -c config/BaselineModel/kwdlc/9e/large-coref-cz.json -d $gpu

latest_checkpoint=$(ls result/BaselineModel-kwdlc-9e-large-coref-cz/*/checkpoint-epoch9.pth | sort | tail -1)
python src/test.py -r $latest_checkpoint -d $gpu
