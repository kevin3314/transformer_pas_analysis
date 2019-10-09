import sys
import argparse

import torch
from pyknp import KNP

import model.model as module_arch
from utils.parse_config import ConfigParser
from writer.prediction_writer import PredictionKNPWriter
from kwdlc_reader import Document
from data_loader.dataset import PASDataset


def prepare_device(n_gpu_use, logger):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        logger.warning("Warning: There\'s no GPU available on this machine,"
                       "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                       "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def main(config, args):
    logger = config.get_logger('test')

    if args.input is not None:
        input_string = args.input
    else:
        input_string = ''.join(sys.stdin.readlines())
    input_string = ''.join(input_string.split())  # remove space character

    input_sentences = [s.strip() + '。' for s in input_string.rstrip('。').split('。')]
    knp = KNP(option='-tab')  # TODO: use BERTKNP
    knp_string = ''.join(knp.parse(input_sentence).all() for input_sentence in input_sentences)

    dataset_config: dict = config['test_kwdlc_dataset']['args']
    dataset_config['path'] = None
    dataset_config['knp_string'] = knp_string
    dataset = PASDataset(**dataset_config)

    # build model architecture
    model = config.initialize('arch', module_arch)
    model.expand_vocab(dataset.num_special_tokens)  # same as that in dataset.py.
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

    prediction_writer = PredictionKNPWriter(dataset, logger)
    if args.tab is True:
        prediction_writer.write(arguments_set.tolist(), sys.stdout)
    else:
        document_pred: Document = prediction_writer.write(arguments_set.tolist(), None)[0]
        for sid in document_pred.sid2sentence.keys():
            document_pred.draw_tree(sid, sys.stdout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--resume', required=True, type=str,
                        help='path to trained checkpoint')
    parser.add_argument('-d', '--device', default='', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--input', default=None, type=str,
                        help='sentences to analysis (if not specified, use stdin)')
    parser.add_argument('-tab', action='store_true', default=False,
                        help='whether to output details')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    main(ConfigParser(parser, timestamp=False), parser.parse_args())
