import argparse
from typing import List, Callable

import torch
import torch.nn as nn
import numpy as np

import data_loader.data_loaders as module_loader
import data_loader.dataset as module_dataset
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from utils.parse_config import ConfigParser
from writer.prediction_writer import PredictionKNPWriter
from scorer import Scorer


class Tester:
    def __init__(self, model, loss, metrics, config, kwdlc_data_loader, kc_data_loader, logger):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.config = config
        self.kwdlc_data_loader = kwdlc_data_loader
        self.kc_data_loader = kc_data_loader
        self.logger = logger

        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self._load_model()
        self.model = model.to(self.device)
        self.model.eval()
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

    def test(self):
        log = {}
        test_log = self._test_epoch(self.kwdlc_data_loader, 'kwdlc')
        log.update(**{'kwdlc_' + k: v for k, v in test_log.items()})
        test_log = self._test_epoch(self.kc_data_loader, 'kc')
        log.update(**{'kc_' + k: v for k, v in test_log.items()})
        return log

    def _load_model(self):
        # prepare model for testing
        self.logger.info(f'Loading checkpoint: {self.config.resume} ...')
        state_dict = torch.load(self.config.resume, map_location=self.device)['state_dict']
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _eval_metrics(self, result: dict):
        f1_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            f1_metrics[i] += metric(result)
        return f1_metrics

    def _test_epoch(self, data_loader, label):
        total_loss = 0.0
        # total_metrics = torch.zeros(len(metric_fns))
        arguments_sets: List[List[List[int]]] = []
        with torch.no_grad():
            for batch_idx, (input_ids, input_mask, arguments_ids, ng_arg_mask, deps) in enumerate(data_loader):
                input_ids = input_ids.to(self.device)          # (b, seq)
                input_mask = input_mask.to(self.device)        # (b, seq)
                arguments_ids = arguments_ids.to(self.device)  # (b, seq, case)
                ng_arg_mask = ng_arg_mask.to(self.device)      # (b, seq, seq)
                deps = deps.to(self.device)                    # (b, seq, seq)

                output = self.model(input_ids, input_mask, ng_arg_mask, deps)  # (b, seq, case, seq)

                arguments_set = torch.argmax(output, dim=3)[:, :, :arguments_ids.size(2)]  # (b, seq, case)
                arguments_sets += arguments_set.tolist()

                # computing loss on test set
                loss = self.loss(output, arguments_ids, deps)
                total_loss += loss.item() * input_ids.size(0)

        prediction_output_dir = self.config.save_dir / 'test_out_knp'
        prediction_writer = PredictionKNPWriter(data_loader.dataset, self.logger)
        documents_pred = prediction_writer.write(arguments_sets, prediction_output_dir)

        scorer = Scorer(documents_pred, data_loader.dataset.documents, data_loader.dataset.kc)
        scorer.write_html(self.config.save_dir / f'result_{label}.html')
        scorer.export_txt(self.config.save_dir / f'result_{label}.txt')
        scorer.export_csv(self.config.save_dir / f'result_{label}.csv')

        metrics = self._eval_metrics(scorer.result_dict())
        log = {'loss': total_loss / data_loader.n_samples}
        log.update({
            met.__name__: metrics[i] for i, met in enumerate(self.metrics)
        })

        return log


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    kwdlc_dataset = config.initialize('test_kwdlc_dataset', module_dataset)
    kc_dataset = config.initialize('test_kc_dataset', module_dataset)
    kwdlc_data_loader = config.initialize('test_data_loader', module_loader, kwdlc_dataset)
    kc_data_loader = config.initialize('test_data_loader', module_loader, kc_dataset)

    # build model architecture
    model = config.initialize('arch', module_arch)
    model.expand_vocab(kwdlc_dataset.num_special_tokens)  # same as that in dataset.py.
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    tester = Tester(model, loss_fn, metric_fns, config, kwdlc_data_loader, kc_data_loader, logger)

    log = tester.test()

    # print logged information to the screen
    for key, value in log.items():
        logger.info('{:35s}: {:.4f}'.format(str(key), value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--resume', required=True, type=str,
                        help='path to checkpoint to test')
    parser.add_argument('-d', '--device', default='', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')

    main(ConfigParser(parser, timestamp=False))
