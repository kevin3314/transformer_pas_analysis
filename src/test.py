import argparse
from typing import List, Callable
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

import data_loader.data_loaders as module_loader
import data_loader.dataset as module_dataset
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from model.metric import PredictionKNPWriter
from scorer import Scorer


def eval_metrics(metrics: List[Callable], result: dict):
    f1_metrics = np.zeros(len(metrics))
    for i, metric in enumerate(metrics):
        f1_metrics[i] += metric(result)
    return f1_metrics


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


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    dataset = config.initialize('test_dataset', module_dataset)
    data_loader = config.initialize('test_data_loader', module_loader, dataset)

    # build model architecture
    model = config.initialize('arch', module_arch)
    model.expand_vocab(len(dataset.special_tokens))  # same as that in dataset.py.
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    device, device_ids = prepare_device(config['n_gpu'], logger)

    # prepare model for testing
    logger.info(f'Loading checkpoint: {config.resume} ...')
    state_dict = torch.load(config.resume, map_location=device)['state_dict']
    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    model = model.to(device)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    model.eval()

    total_loss = 0.0
    # total_metrics = torch.zeros(len(metric_fns))
    arguments_sets: List[List[List[int]]] = []
    with torch.no_grad():
        for batch_idx, (input_ids, input_mask, arguments_ids, ng_arg_mask, deps) in enumerate(data_loader):
            input_ids = input_ids.to(device)          # (b, seq)
            input_mask = input_mask.to(device)        # (b, seq)
            arguments_ids = arguments_ids.to(device)  # (b, seq, case)
            ng_arg_mask = ng_arg_mask.to(device)      # (b, seq, seq)
            deps = deps.to(device)                    # (b, seq, seq)

            output = model(input_ids, input_mask, ng_arg_mask, deps)  # (b, seq, case, seq)

            arguments_set = torch.argmax(output, dim=3)[:, :, :arguments_ids.size(2)]  # (b, seq, case)
            arguments_sets += arguments_set.tolist()

            # computing loss, metrics on test set
            loss = loss_fn(output, arguments_ids, deps)
            total_loss += loss.item() * input_ids.size(0)
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(ret_dict) * batch_size

    prediction_output_dir = config.save_dir / 'test_out_knp'
    prediction_writer = PredictionKNPWriter(data_loader.dataset,
                                            config['test_dataset']['args'],
                                            logger)
    documents_pred = prediction_writer.write(arguments_sets, prediction_output_dir)

    scorer = Scorer(documents_pred, data_loader.dataset.reader)
    scorer.print_result()
    scorer.write_html(config.save_dir / 'result.html')

    metrics = eval_metrics(metric_fns, scorer.result_dict())
    log = {'loss': total_loss / data_loader.n_samples}
    log.update({
        met.__name__: metrics[i] for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default='', type=str,
                        help='indices of GPUs to enable (default: all)')

    main(ConfigParser(parser, timestamp=False))
