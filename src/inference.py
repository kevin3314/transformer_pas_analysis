import sys
import argparse
from typing import List
from pathlib import Path

import torch

import model.model as module_arch
from parse_config import ConfigParser
from model.metric import PredictionKNPWriter
from scorer import Scorer
from kwdlc_reader import Document
from data_loader.dataset import PASDataset
from test import prepare_device


def main(config, tab: bool):
    logger = config.get_logger('test')

    dataset_config: dict = config['test_dataset']['args']

    knp_string = ''.join(sys.stdin.readlines())
    input_dir = Path('input/')
    input_dir.mkdir(exist_ok=True)
    with (input_dir / '0.knp').open('w') as f:
        f.write(knp_string)
    # document = Document(path.parent,
    #                     target_cases=dataset_config['cases'],
    #                     target_corefs=['=', '=構', '=≒'],
    #                     target_exophors=dataset_config['exophors'],
    #                     extract_nes=False)

    dataset_config['path'] = str(input_dir)
    dataset = PASDataset(**dataset_config)

    # build model architecture
    model = config.initialize('arch', module_arch)
    model.expand_vocab(len(dataset.special_tokens))  # same as that in dataset.py.
    logger.info(model)

    device, device_ids = prepare_device(1, logger)

    # prepare model for testing
    logger.info(f'Loading checkpoint: {config.resume} ...')
    state_dict = torch.load(config.resume, map_location=device)['state_dict']
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        input_ids, input_mask, arguments_ids, ng_arg_mask, deps = dataset[0]
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)          # (1, seq)
        input_mask = torch.tensor(input_mask).to(device).unsqueeze(0)        # (1, seq)
        arguments_ids = torch.tensor(arguments_ids).to(device).unsqueeze(0)  # (1, seq, case)
        ng_arg_mask = torch.tensor(ng_arg_mask).to(device).unsqueeze(0)      # (1, seq, seq)
        deps = torch.tensor(deps).to(device).unsqueeze(0)                    # (1, seq, seq)

        output = model(input_ids, input_mask, ng_arg_mask, deps)  # (1, seq, case, seq)

        arguments_set = torch.argmax(output, dim=3)[:, :, :arguments_ids.size(2)]  # (1, seq, case)

    prediction_writer = PredictionKNPWriter(dataset,
                                            dataset_config,
                                            logger)
    if tab is True:
        prediction_writer.write(arguments_set.tolist(), None)
    else:
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        prediction_writer.write(arguments_set.tolist(), output_dir)
        scorer = Scorer(output_dir, dataset.reader)
        scorer.draw_first_tree()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to trained checkpoint (default: None)')
    parser.add_argument('-d', '--device', default='', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-tab', action='store_true', default=False,
                        help='output details')
    main(ConfigParser(parser, timestamp=False), parser.parse_args().tab)
