#!/usr/bin/env bash

set -exu

gpu=2

python src/train.py -c config/BaselineModel/kwdlc/3e/large-coref-o-nocase-noun.json -d $gpu

latest_checkpoint=$(ls result/BaselineModel-kwdlc-3e-large-coref-o-nocase-noun/*/checkpoint-epoch3.pth | sort | tail -1)
python src/train.py -r $latest_checkpoint -c config/BaselineModel/kwdlc/6e/large-coref-oc-nocase-noun.json -d $gpu

latest_checkpoint=$(ls result/BaselineModel-kwdlc-6e-large-coref-oc-nocase-noun/*/checkpoint-epoch6.pth | sort | tail -1)
python src/train.py -r $latest_checkpoint -c config/BaselineModel/kwdlc/9e/large-coref-cz-nocase.json -d $gpu

latest_checkpoint=$(ls result/BaselineModel-kwdlc-9e-large-coref-cz-nocase/*/checkpoint-epoch9.pth | sort | tail -1)
python src/test.py -r $latest_checkpoint -d $gpu
