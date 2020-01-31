#!/usr/bin/env bash
set -exu

# usage:
# n=5 gpu=0 ./train.sh config/BaselineModel/kwdlc/4e/large-coref-cz.json

for i in $(seq $n); do
    python src/train.py -c $1 -d $gpu --seed $RANDOM
done
