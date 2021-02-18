"""Rewrite result weight trained on PAS such that contains only weight info.
Result weight without any modification leads to load errors because it includes
model information.
"""

import sys
sys.path.append('src')

import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn

from utils.parse_config import ConfigParser
import model.model as module_arch


def main(args) -> None:
    with open(args.config) as f:
        config = json.load(f)
    vocab_size = config['vocab_size']

    # build model architecture
    weight = torch.load(args.source, map_location=torch.device('cpu'))
    print("Weight is loaded")

    state_dict = weight["state_dict"]
    state_dict["module.bert.embeddings.word_embeddings.weight"] = state_dict["module.bert.embeddings.word_embeddings.weight"][:vocab_size, :]
    key_names = list(state_dict.keys())
    for key_name in key_names:
        new_key_name = key_name.replace("module.", "")
        state_dict[new_key_name] = state_dict.pop(key_name)
    print(list(state_dict.keys())[:20])

    res_parent = Path(args.result).parent
    res_parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing weight to {args.result}")
    torch.save(state_dict, args.result)


if __name__ == '__main__':
    print("n_gpu", str(torch.cuda.device_count()))

    parser = argparse.ArgumentParser()
    parser.add_argument('source', default=None, type=str, help='path to source checkpoint')
    parser.add_argument('result', default=None, type=str, help='path to result checkpoint')
    parser.add_argument('config', default=None, type=str, help="path to original pretrained model's config")

    args = parser.parse_args()
    main(args)
