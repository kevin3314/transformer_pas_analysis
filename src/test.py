import os
import argparse
from typing import List

import torch
import torch.nn as nn

import data_loader.data_loaders as module_loader
import data_loader.dataset as module_dataset
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from model.metric import write_prediction


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    dataset = config.initialize('test_dataset', module_dataset)
    data_loader = config.initialize('test_data_loader', module_loader, dataset)

    # build model architecture
    model = config.initialize('arch', module_arch)
    model.expand_vocab(num_expand_vocab=5)  # same as that in dataset.py. TODO: consider resume case
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    # total_metrics = torch.zeros(len(metric_fns))
    arguments_sets: List[List[List[int]]] = []
    with torch.no_grad():
        for batch_idx, (input_ids, input_mask, arguments_ids, ng_arg_mask) in enumerate(data_loader):
            input_ids = input_ids.to(device)          # (b, seq)
            input_mask = input_mask.to(device)        # (b, seq)
            arguments_ids = arguments_ids.to(device)  # (b, seq, case)
            ng_arg_mask = ng_arg_mask.to(device)      # (b, seq, seq)

            output = model(input_ids, input_mask, ng_arg_mask)  # (b, seq, case, seq)

            arguments_set = torch.argmax(output, dim=3)  # (b, seq, case)
            arguments_sets += arguments_set.tolist()

            # computing loss, metrics on test set
            loss = loss_fn(output, arguments_ids)
            total_loss += loss.item() * input_ids.size(0)
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(ret_dict) * batch_size

    output_prediction_file: str = os.path.join(config.save_dir, 'test_out.conll')
    write_prediction(dataset.pas_examples,
                     dataset.features,
                     arguments_sets,
                     output_prediction_file,
                     config['test_dataset']['args'],
                     logger)

    log = {'loss': total_loss / data_loader.n_samples}
    # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    logger.info(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    main(ConfigParser(parser, timestamp=False))
