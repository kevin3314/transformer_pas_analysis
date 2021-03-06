import re
import argparse
from pathlib import Path
from typing import List, Callable, Set

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import data_loader.data_loaders as module_loader
import data_loader.dataset as module_dataset
import model.metric as module_metric
import model.model as module_arch
from data_loader.dataset import PASDataset
from utils.parse_config import ConfigParser
from prediction.prediction_writer import PredictionKNPWriter
from prediction.inference import Inference
from scorer import Scorer


class Tester:
    def __init__(self, model, metrics, config, data_loaders, target, logger, predict_overt, precision_threshold,
                 recall_threshold, result_suffix):
        self.model: nn.Module = model
        self.metrics: List[Callable] = metrics
        self.config = config
        self.data_loaders = data_loaders
        self.target: str = target
        self.logger = logger
        self.predict_overt: bool = predict_overt

        self.inference = Inference(config, model,
                                   precision_threshold=precision_threshold,
                                   recall_threshold=recall_threshold,
                                   logger=logger)

        self.save_dir: Path = config.save_dir / f'eval_{target}{result_suffix}'
        self.save_dir.mkdir(exist_ok=True)
        pas_targets: Set[str] = set()
        for corpus, data_loader in self.data_loaders.items():
            if corpus != 'commonsense':
                pas_targets |= set(data_loader.dataset.pas_targets)
        self.pas_targets: List[str] = []
        if 'pred' in pas_targets:
            self.pas_targets.append('pred')
        if 'noun' in pas_targets:
            self.pas_targets.append('noun')
        if 'noun' in pas_targets and 'pred' in pas_targets:
            self.pas_targets.append('all')
        if not self.pas_targets:
            self.pas_targets.append('')

    def test(self):
        log = {}
        for corpus, data_loader in self.data_loaders.items():
            log.update(self._test(data_loader, corpus))
        return log

    def _test(self, data_loader, corpus: str):

        loss, *predictions = self.inference(data_loader)

        log = {}
        if corpus != 'commonsense':
            if re.match(r'.*(CaseInteraction|Refinement|Duplicate).*Model', self.config['arch']['type']):
                *pre_predictions, prediction = predictions
                for i, pre_prediction in enumerate(pre_predictions):
                    arguments_sets = pre_prediction.tolist()
                    result = self._eval_pas(arguments_sets, data_loader.dataset, corpus=corpus, suffix=f'_{i}')
                    log.update({f'{self.target}_{corpus}_{k}_{i}': v for k, v in result.items()})
            else:
                prediction = predictions[0]  # (N, seq, case, seq)
            result = self._eval_pas(prediction.tolist(), data_loader.dataset, corpus=corpus)
        else:
            assert self.config['arch']['type'] == 'CommonsenseModel'
            result = self._eval_commonsense(predictions[1].tolist(), data_loader)
        result['loss'] = loss

        log.update({f'{self.target}_{corpus}_{k}': v for k, v in result.items()})
        return log

    def _eval_pas(self, arguments_set, dataset: PASDataset, corpus: str, suffix: str = ''):
        prediction_output_dir = self.save_dir / f'{corpus}_out{suffix}'
        prediction_writer = PredictionKNPWriter(dataset,
                                                self.logger,
                                                use_knp_overt=(not self.predict_overt))
        documents_pred = prediction_writer.write(arguments_set, prediction_output_dir)
        documents_gold = dataset.joined_documents if corpus == 'kc' else dataset.documents

        result = {}
        for pas_target in self.pas_targets:
            scorer = Scorer(documents_pred, documents_gold,
                            target_cases=dataset.target_cases,
                            target_exophors=dataset.target_exophors,
                            coreference=dataset.coreference,
                            bridging=dataset.bridging,
                            pas_target=pas_target)

            stem = corpus
            if pas_target:
                stem += f'_{pas_target}'
            stem += suffix
            if self.target != 'test':
                scorer.write_html(self.save_dir / f'{stem}.html')
            scorer.export_txt(self.save_dir / f'{stem}.txt')
            scorer.export_csv(self.save_dir / f'{stem}.csv')

            metrics = self._eval_metrics(scorer.result_dict())
            for met, value in zip(self.metrics, metrics):
                met_name = met.__name__
                if 'case_analysis' in met_name or 'zero_anaphora' in met_name:
                    if pas_target:
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
        assert data_loader.dataset.__class__.__name__ == 'CommonsenseDataset'
        gold = np.array([f.label for f in data_loader.dataset.features])
        return {'f1': f1_score(gold, contingency_set)}


def main(config, args):
    logger = config.get_logger(args.target)

    # setup data_loader instances
    data_loaders = {}
    for corpus in config[f'{args.target}_datasets'].keys():
        dataset = config.init_obj(f'{args.target}_datasets.{corpus}', module_dataset, logger=logger)
        data_loaders[corpus] = config.init_obj(f'data_loaders.{args.target}', module_loader, dataset)

    # build model architecture
    model: nn.Module = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    tester = Tester(model, metric_fns, config, data_loaders, args.target, logger, args.predict_overt,
                    args.precision_threshold, args.recall_threshold, args.result_suffix)

    log = tester.test()

    # print logged information to the screen
    for key, value in log.items():
        logger.info('{:42s}: {:.4f}'.format(str(key), value))


if __name__ == '__main__':
    print("n_gpu", str(torch.cuda.device_count()))
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to checkpoint to test')
    parser.add_argument('--ens', '--ensemble', default=None, type=str,
                        help='path to directory where checkpoints to ensemble exist')
    parser.add_argument('-d', '--device', default='', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-t', '--target', default='test', type=str, choices=['valid', 'test'],
                        help='evaluation target')
    parser.add_argument('--predict-overt', action='store_true', default=False,
                        help='calculate scores for overt arguments instead of using gold')
    parser.add_argument('--precision-threshold', default=0.0, type=float,
                        help='threshold for argument existence. The higher you set, the higher precision gets. [0, 1]')
    parser.add_argument('--recall-threshold', default=0.0, type=float,
                        help='threshold for argument non-existence. The higher you set, the higher recall gets [0, 1]')
    parser.add_argument('--result-suffix', default='', type=str,
                        help='custom evaluation result directory name')
    parser.add_argument('--run-id', default=None, type=str,
                        help='custom experiment directory name')
    parsed_args = parser.parse_args()
    inherit_save_dir = (parsed_args.resume is not None and parsed_args.run_id is None)
    main(ConfigParser.from_args(parsed_args, run_id=parsed_args.run_id, inherit_save_dir=inherit_save_dir), parsed_args)
