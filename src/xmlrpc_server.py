import os
import io
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import OrderedDict
from xmlrpc.server import SimpleXMLRPCServer

import torch

from data_loader.dataset import PASDataset
from writer.prediction_writer import PredictionKNPWriter
import model.model as module_arch

logger = logging.getLogger(__name__)


def show_usage():
    print('Usage:')
    print('$ env HOST=<HOST> PORT=<PORT> pipenv run server.py')


def analyze_raw_data_from_client(knp_result: str):
    dataset_config: dict = config['test_dataset']['args']
    dataset_config['path'] = None
    dataset_config['knp_string'] = knp_result
    dataset = PASDataset(**dataset_config)

    with torch.no_grad():
        input_ids, input_mask, arguments_ids, ng_arg_mask, deps = dataset[0]
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)  # (1, seq)
        input_mask = torch.tensor(input_mask).to(device).unsqueeze(0)  # (1, seq)
        arguments_ids = torch.tensor(arguments_ids).to(device).unsqueeze(0)  # (1, seq, case)
        ng_arg_mask = torch.tensor(ng_arg_mask).to(device).unsqueeze(0)  # (1, seq, seq)
        deps = torch.tensor(deps).to(device).unsqueeze(0)  # (1, seq, seq)

        output = model(input_ids, input_mask, ng_arg_mask, deps)  # (1, seq, case, seq)

        arguments_set = torch.argmax(output, dim=3)[:, :, :arguments_ids.size(2)]  # (1, seq, case)

    prediction_writer = PredictionKNPWriter(dataset, logger)
    with io.StringIO() as string:
        _ = prediction_writer.write(arguments_set.tolist(), string)
        knp_result = string.getvalue()
    return knp_result


if __name__ == '__main__':
    HOST = os.environ.get('HOST')
    PORT = os.environ.get('PORT')
    if not all((HOST, PORT)):
        show_usage()
        sys.exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str,
                        help='path to trained checkpoint')
    args = parser.parse_args()

    device = torch.device('cpu')

    config_path = Path(args.model).parent / 'config.json'
    with config_path.open() as handle:
        config = json.load(handle, object_hook=OrderedDict)

    # build model architecture
    model_args = dict(config['arch']['args'])
    model = getattr(module_arch, config['arch']['type'])(**model_args)
    coreference = config['test_dataset']['args']['coreference']
    model.expand_vocab(len(config['test_dataset']['args']['exophors']) + 1 + int(coreference))

    # prepare model for testing
    logger.info(f'Loading checkpoint: {args.model} ...')
    state_dict = torch.load(args.model, map_location=device)['state_dict']
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model.eval()

    server = SimpleXMLRPCServer((HOST, int(PORT)))
    server.register_function(analyze_raw_data_from_client)
    server.serve_forever()
