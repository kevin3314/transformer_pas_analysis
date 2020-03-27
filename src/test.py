import re
import argparse
from pathlib import Path
from typing import List, Union, Tuple, Callable

import torch
import numpy as np
from sklearn.metrics import f1_score

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
    def __init__(self, model, loss, metrics, config, kwdlc_data_loader, kc_data_loader, commonsense_data_loader,
                 target, logger, predict_overt):
        self.model: BaseModel = model
        self.loss: Callable = loss
        self.metrics: List[Callable] = metrics
        self.config = config
        self.kwdlc_data_loader = kwdlc_data_loader
        self.kc_data_loader = kc_data_loader
        self.commonsense_data_loader = commonsense_data_loader
        self.target: str = target
        self.logger = logger
        self.predict_overt: bool = predict_overt

        self.device, self.device_ids = prepare_device(config['n_gpu'], self.logger)
        self.checkpoints: List[Path] = list(config.save_dir.glob('**/model_best.pth'))
        self.save_dir: Path = config.save_dir / f'eval_{target}'
        self.save_dir.mkdir(exist_ok=True)
        eventive_noun = (kwdlc_data_loader and config[f'{target}_kwdlc_dataset']['args']['eventive_noun']) or \
                        (kc_data_loader and config[f'{target}_kc_dataset']['args']['eventive_noun'])
        self.pas_targets = ('pred', 'noun', 'all') if eventive_noun else ('pred',)

    def test(self):
        log = {}
        if self.kwdlc_data_loader is not None:
            log.update(self._test(self.kwdlc_data_loader, 'kwdlc'))
        if self.kc_data_loader is not None:
            log.update(self._test(self.kc_data_loader, 'kc'))
        if self.commonsense_data_loader is not None:
            log.update(self._test(self.commonsense_data_loader, 'commonsense'))
        return log

    def _test(self, data_loader, label: str):
        log = {}
        total_output = (np.array(0), np.array(0))
        output2 = None
        total_loss = 0.0
        for checkpoint in self.checkpoints:
            model = self._prepare_model(checkpoint)
            output, loss = self._test_epoch(model, data_loader)
            total_output = tuple(t + o for t, o in zip(total_output, output))
            total_loss += loss

        if re.match(r'(CaseInteractionModel2|Refinement|Duplicate)', self.config['arch']['type']):
            output_base, output = total_output
            arguments_sets_base = np.argmax(output_base, axis=3).tolist()
            result_base = self._eval_pas(arguments_sets_base, data_loader, corpus=label, suffix='_base')
            log.update({f'{self.target}_{label}_{k}_base': v for k, v in result_base.items()})
        elif self.config['arch']['type'] == 'CommonsenseModel':
            output, output2 = total_output  # (N, seq, case, seq), (N, 2)
        else:
            output = total_output[0]

        if label in ('kwdlc', 'kc'):
            arguments_set = np.argmax(output, axis=3).tolist()
            result = self._eval_pas(arguments_set, data_loader, corpus=label)
        elif label == 'commonsense':
            assert self.config['arch']['type'] == 'CommonsenseModel'
            contingency_set = np.argmax(output2, axis=1)  # (N)
            result = self._eval_commonsense(contingency_set, data_loader)
        else:
            raise ValueError(f'unknown label: {label}')
        result['loss'] = total_loss / data_loader.n_samples
        log.update({f'{self.target}_{label}_{k}': v for k, v in result.items()})

        return log

    def _prepare_model(self, checkpoint: Path):
        # prepare model for testing
        self.logger.info(f'Loading checkpoint: {checkpoint} ...')
        state_dict = torch.load(checkpoint, map_location=self.device)['state_dict']
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
        self.model.eval()
        model = self.model.to(self.device)
        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        return model

    def _test_epoch(self, model, data_loader) -> Tuple[Union[tuple], float]:
        total_loss = 0.0
        outputs = []
        outputs2 = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, target, ng_token_mask, deps, task = batch

                output_ = model(input_ids, input_mask, segment_ids, ng_token_mask, deps)  # (b, seq, case, seq)
                if isinstance(output_, tuple):
                    output, output2 = output_
                    outputs2.append(output2.cpu().numpy())
                else:
                    output = output_
                outputs.append(output.cpu().numpy())

                # computing loss on test set
                loss = self.loss(output_, target, deps, task)
                total_loss += loss.item() * input_ids.size(0)
        avg_loss = total_loss / data_loader.n_samples
        if outputs2:
            return (np.concatenate(outputs, axis=0), np.concatenate(outputs2, axis=0)), avg_loss
        else:
            return (np.concatenate(outputs, axis=0), ), avg_loss

    def _eval_pas(self, arguments_set, data_loader, corpus: str, suffix: str = ''):
        prediction_output_dir = self.save_dir / f'{corpus}_out{suffix}'
        prediction_writer = PredictionKNPWriter(data_loader.dataset,
                                                self.logger,
                                                use_gold_overt=(not self.predict_overt))
        documents_pred = prediction_writer.write(arguments_set, prediction_output_dir)

        result = {}
        for pas_target in self.pas_targets:
            scorer = Scorer(documents_pred, data_loader.dataset.documents,
                            target_cases=data_loader.dataset.target_cases,
                            target_exophors=data_loader.dataset.target_exophors,
                            coreference=data_loader.dataset.coreference,
                            kc=data_loader.dataset.kc,
                            pas_target=pas_target)
            if self.target != 'test':
                scorer.write_html(self.save_dir / f'{corpus}_{pas_target}{suffix}.html')
            scorer.export_txt(self.save_dir / f'{corpus}_{pas_target}{suffix}.txt')
            scorer.export_csv(self.save_dir / f'{corpus}_{pas_target}{suffix}.csv')

            metrics = self._eval_metrics(scorer.result_dict())
            for met, value in zip(self.metrics, metrics):
                met_name = met.__name__
                if 'case_analysis' in met_name or 'zero_anaphora' in met_name:
                    met_name = f'{pas_target}_{met_name}'
                result[met_name] = value

        return result

    def _eval_metrics(self, result: dict):
        f1_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            f1_metrics[i] += metric(result)
        return f1_metrics

    @staticmethod
    def _eval_commonsense(contingency_set: np.ndarray, data_loader) -> dict:
        gold = np.array([f.label for f in data_loader.dataset.features])
        return {'f1': f1_score(gold, contingency_set)}


def main(config, args):
    logger = config.get_logger(args.target)

    # setup data_loader instances
    expanded_vocab_size = None
    kwdlc_data_loader = None
    if config[f'{args.target}_kwdlc_dataset']['args']['path'] is not None:
        dataset = config.init_obj(f'{args.target}_kwdlc_dataset', module_dataset, logger=logger)
        kwdlc_data_loader = config.init_obj(f'{args.target}_data_loader', module_loader, dataset)
        expanded_vocab_size = dataset.expanded_vocab_size
    kc_data_loader = None
    if config[f'{args.target}_kc_dataset']['args']['path'] is not None:
        dataset = config.init_obj(f'{args.target}_kc_dataset', module_dataset, logger=logger)
        kc_data_loader = config.init_obj(f'{args.target}_data_loader', module_loader, dataset)
        expanded_vocab_size = dataset.expanded_vocab_size
    commonsense_data_loader = None
    if config.config.get(f'{args.target}_commonsense_dataset', None) is not None:
        dataset = config.init_obj(f'{args.target}_commonsense_dataset', module_dataset, logger=logger)
        commonsense_data_loader = config.init_obj(f'{args.target}_data_loader', module_loader, dataset)

    # build model architecture
    model: BaseModel = config.init_obj('arch', module_arch, vocab_size=expanded_vocab_size)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    tester = Tester(model, loss_fn, metric_fns, config, kwdlc_data_loader, kc_data_loader, commonsense_data_loader,
                    args.target, logger, args.predict_overt)

    log = tester.test()

    # print logged information to the screen
    for key, value in log.items():
        logger.info('{:42s}: {:.4f}'.format(str(key), value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to checkpoint to test')
    parser.add_argument('--ens', default=None, type=str,
                        help='path to directory where checkpoints to ensemble exist')
    parser.add_argument('-d', '--device', default='', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('--target', default='test', type=str, choices=['valid', 'test'],
                        help='evaluation target')
    parser.add_argument('--predict-overt', action='store_true', default=False,
                        help='calculate scores for overt arguments instead of using gold')
    parser.add_help = True

    parsed_args = parser.parse_args()
    config_args = {'run_id': ''} if parsed_args.resume is None else {'inherit_save_dir': True}
    main(ConfigParser.from_parser(parser, **config_args), parsed_args)
