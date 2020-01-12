import argparse
from typing import List, Union, Tuple

import torch
import numpy as np

import data_loader.data_loaders as module_loader
import data_loader.dataset as module_dataset
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from utils.parse_config import ConfigParser
from utils import prepare_device
from writer.prediction_writer import PredictionKNPWriter
from scorer import Scorer
from base.base_model import BaseModel


class Tester:
    def __init__(self, model, loss, metrics, config, kwdlc_data_loader, kc_data_loader, target, logger, predict_overt):
        self.loss = loss
        self.metrics = metrics
        self.config: ConfigParser = config
        self.kwdlc_data_loader = kwdlc_data_loader
        self.kc_data_loader = kc_data_loader
        self.target: str = target
        self.logger = logger
        self.predict_overt: bool = predict_overt

        self.device, device_ids = prepare_device(config['n_gpu'], self.logger)
        self._load_model(model)
        self.model: BaseModel = model.to(self.device)
        self.model.eval()
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

    def test(self):
        log = {}
        if self.kwdlc_data_loader is not None:
            output, loss = self._test_epoch(self.model, self.kwdlc_data_loader)
            if isinstance(output, tuple):
                output_base, output = output
                arguments_sets_base = np.argmax(output_base, axis=3)
                kwdlc_log_base = self._eval(arguments_sets_base, loss, self.kwdlc_data_loader, suffix='kwdlc_base')
                log.update({f'{self.target}_kwdlc_{k}_base': v for k, v in kwdlc_log_base.items()})
            arguments_sets = np.argmax(output, axis=3)
            kwdlc_log = self._eval(arguments_sets, loss, self.kwdlc_data_loader, suffix='kwdlc')
            log.update({f'{self.target}_kwdlc_{k}': v for k, v in kwdlc_log.items()})
        if self.kc_data_loader is not None:
            output, loss = self._test_epoch(self.model, self.kwdlc_data_loader)
            if isinstance(output, tuple):
                output_base, output = output
                arguments_sets_base = np.argmax(output_base, axis=3)
                kc_log_base = self._eval(arguments_sets_base, loss, self.kwdlc_data_loader, suffix='kc_base')
                log.update({f'{self.target}_kc_{k}_base': v for k, v in kc_log_base.items()})
            arguments_sets = np.argmax(output, axis=3)
            kc_log = self._eval(arguments_sets, loss, self.kwdlc_data_loader, suffix='kc')
            log.update({f'{self.target}_kwdlc_{k}': v for k, v in kc_log.items()})
        return log

    def _load_model(self, model: BaseModel):
        # prepare model for testing
        self.logger.info(f'Loading checkpoint: {self.config.resume} ...')
        state_dict = torch.load(self.config.resume, map_location=self.device)['state_dict']
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})

    def _test_epoch(self, model, data_loader) -> Tuple[Union[tuple, np.ndarray], float]:
        total_loss = 0.0
        outputs_base = []
        outputs = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, arguments_ids, ng_token_mask, deps = batch

                output_ = model(input_ids, input_mask, ng_token_mask, deps)  # (b, seq, case, seq)
                if isinstance(output_, tuple):
                    output_base, output = output_
                    outputs_base.append(output_base.cpu().numpy())
                else:
                    output = output_
                output = output[:, :, :arguments_ids.size(2), :]
                outputs.append(output.cpu().numpy())

                # computing loss on test set
                loss = self.loss(output_, arguments_ids, deps)
                total_loss += loss.item() * input_ids.size(0)
        avg_loss = total_loss / data_loader.n_samples
        if outputs_base:
            return (np.concatenate(outputs_base, axis=0), np.concatenate(outputs, axis=0)), avg_loss
        else:
            return np.concatenate(outputs, axis=0), avg_loss

    def _eval(self, arguments_sets, loss, data_loader, suffix=''):
        prediction_output_dir = self.config.save_dir / f'{self.target}_out_{suffix}'
        prediction_writer = PredictionKNPWriter(data_loader.dataset,
                                                self.logger,
                                                use_gold_overt=(not self.predict_overt))
        documents_pred = prediction_writer.write(arguments_sets, prediction_output_dir)

        scorer = Scorer(documents_pred, data_loader.dataset.documents, data_loader.dataset.target_exophors,
                        coreference=data_loader.dataset.coreference,
                        kc=data_loader.dataset.kc,
                        eval_eventive_noun=False)
        if self.target != 'test':
            scorer.write_html(self.config.save_dir / f'result_{self.target}_{suffix}.html')
        scorer.export_txt(self.config.save_dir / f'result_{self.target}_{suffix}.txt')
        scorer.export_csv(self.config.save_dir / f'result_{self.target}_{suffix}.csv')

        metrics = self._eval_metrics(scorer.result_dict())
        log = {'loss': loss / data_loader.n_samples}
        log.update({
            met.__name__: metrics[i] for i, met in enumerate(self.metrics)
        })

        return log

    def _eval_metrics(self, result: dict):
        f1_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            f1_metrics[i] += metric(result)
        return f1_metrics


def main(config, args):
    logger = config.get_logger(args.target)

    # setup data_loader instances
    kwdlc_data_loader = None
    kc_data_loader = None
    expanded_vocab_size = None
    if config[f'{args.target}_kwdlc_dataset']['args']['path'] is not None:
        kwdlc_dataset = config.init_obj(f'{args.target}_kwdlc_dataset', module_dataset, logger=logger)
        kwdlc_data_loader = config.init_obj(f'{args.target}_data_loader', module_loader, kwdlc_dataset)
        expanded_vocab_size = kwdlc_dataset.expanded_vocab_size
    if config[f'{args.target}_kc_dataset']['args']['path'] is not None:
        kc_dataset = config.init_obj(f'{args.target}_kc_dataset', module_dataset, logger=logger)
        kc_data_loader = config.init_obj(f'{args.target}_data_loader', module_loader, kc_dataset)
        expanded_vocab_size = kc_dataset.expanded_vocab_size

    # build model architecture
    model: BaseModel = config.init_obj('arch', module_arch, vocab_size=expanded_vocab_size)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    tester = Tester(model, loss_fn, metric_fns, config, kwdlc_data_loader, kc_data_loader, args.target, logger,
                    args.predict_overt)

    log = tester.test()

    # print logged information to the screen
    for key, value in log.items():
        logger.info('{:41s}: {:.4f}'.format(str(key), value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--resume', required=True, type=str,
                        help='path to checkpoint to test')
    parser.add_argument('-d', '--device', default='', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('--target', default='test', type=str, choices=['valid', 'test'],
                        help='evaluation target')
    parser.add_argument('--predict-overt', action='store_true', default=False,
                        help='calculate scores for overt arguments instead of using gold')

    main(ConfigParser.from_parser(parser, inherit_save_dir=True), parser.parse_args())
