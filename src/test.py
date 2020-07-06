import re
import argparse
from pathlib import Path
from typing import List, Tuple, Callable, Set

import torch
import numpy as np
from sklearn.metrics import f1_score

import data_loader.data_loaders as module_loader
import data_loader.dataset as module_dataset
import model.metric as module_metric
import model.model as module_arch
from utils.parse_config import ConfigParser
from utils import prepare_device
from writer.prediction_writer import PredictionKNPWriter
from scorer import Scorer
from base.base_model import BaseModel


class Tester:
    def __init__(self, model, metrics, config, kwdlc_data_loader, kc_data_loader, commonsense_data_loader,
                 target, logger, predict_overt, confidence_threshold, result_suffix):
        self.model: BaseModel = model
        self.metrics: List[Callable] = metrics
        self.config = config
        self.kwdlc_data_loader = kwdlc_data_loader
        self.kc_data_loader = kc_data_loader
        self.commonsense_data_loader = commonsense_data_loader
        self.target: str = target
        self.logger = logger
        self.predict_overt: bool = predict_overt
        self.threshold: float = confidence_threshold

        self.device, self.device_ids = prepare_device(config['n_gpu'], self.logger)
        self.checkpoints: List[Path] = [config.resume] if config.resume is not None \
            else list(config.save_dir.glob('**/model_best.pth'))
        self.save_dir: Path = config.save_dir / f'eval_{target}{result_suffix}'
        self.save_dir.mkdir(exist_ok=True)
        pas_targets: Set[str] = set()
        if kwdlc_data_loader is not None:
            pas_targets |= set(kwdlc_data_loader.dataset.pas_targets)
        if kc_data_loader is not None:
            pas_targets |= set(kc_data_loader.dataset.pas_targets)
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
        if self.kwdlc_data_loader is not None:
            log.update(self._test(self.kwdlc_data_loader, 'kwdlc'))
        if self.kc_data_loader is not None:
            log.update(self._test(self.kc_data_loader, 'kc'))
        if self.commonsense_data_loader is not None:
            log.update(self._test(self.commonsense_data_loader, 'commonsense'))
        return log

    def _test(self, data_loader, label: str):
        log = {}
        total_output = None
        total_loss = 0.0
        for checkpoint in self.checkpoints:
            model = self._prepare_model(checkpoint)
            loss, *output = self._test_epoch(model, data_loader)
            total_output = tuple(t + o for t, o in zip(total_output, output)) if total_output is not None else output
            total_loss += loss

        if re.match(r'.*(CaseInteraction|Refinement|Duplicate)Model', self.config['arch']['type']):
            *pre_outputs, output = total_output
            for i, pre_output in enumerate(pre_outputs):
                arguments_sets = np.argmax(pre_output, axis=3).tolist()
                result = self._eval_pas(arguments_sets, data_loader, corpus=label, suffix=f'_{i}')
                log.update({f'{self.target}_{label}_{k}_{i}': v for k, v in result.items()})
        else:
            output = total_output[0]  # (N, seq, case, seq)

        if label in ('kwdlc', 'kc'):
            output = Tester._softmax(output, axis=3)
            null_idx = data_loader.dataset.special_to_index['NULL']
            if data_loader.dataset.coreference:
                output[:, :, :-1, null_idx] += (output[:, :, :-1] < self.threshold).all(axis=3).astype(np.int) * 1024
                na_idx = data_loader.dataset.special_to_index['NA']
                output[:, :, -1, na_idx] += (output[:, :, -1] < self.threshold).all(axis=2).astype(np.int) * 1024
            else:
                output[:, :, :, null_idx] += (output < self.threshold).all(axis=3).astype(np.int) * 1024
            arguments_set = np.argmax(output, axis=3).tolist()
            result = self._eval_pas(arguments_set, data_loader, corpus=label)
        elif label == 'commonsense':
            assert self.config['arch']['type'] == 'CommonsenseModel'
            contingency_set = (total_output[1] > 0.5).astype(np.int)  # (N)
            result = self._eval_commonsense(contingency_set, data_loader)
        else:
            raise ValueError(f'unknown label: {label}')
        result['loss'] = total_loss / data_loader.n_samples
        log.update({f'{self.target}_{label}_{k}': v for k, v in result.items()})

        return log

    @staticmethod
    def _softmax(x: np.ndarray, axis: int):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum(axis=axis, keepdims=True) + 1e-8)

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

    def _test_epoch(self, model, data_loader) -> Tuple[float, ...]:
        total_loss = 0.0
        outputs: List[Tuple[np.ndarray, ...]] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # (input_ids, input_mask, segment_ids, ng_token_mask, target, deps, task)
                batch = tuple(t.to(self.device) for t in batch)

                loss, *output = model(*batch)

                if len(loss.size()) > 0:
                    loss = loss.mean()
                outputs.append(tuple(o.cpu().numpy() for o in output))
                total_loss += loss.item() * output[0].size(0)
        avg_loss: float = total_loss / data_loader.n_samples
        return (avg_loss, *(np.concatenate(outs, axis=0) for outs in zip(*outputs)))

    def _eval_pas(self, arguments_set, data_loader, corpus: str, suffix: str = ''):
        prediction_output_dir = self.save_dir / f'{corpus}_out{suffix}'
        prediction_writer = PredictionKNPWriter(data_loader.dataset,
                                                self.logger,
                                                use_gold_overt=(not self.predict_overt))
        documents_pred = prediction_writer.write(arguments_set, prediction_output_dir)
        if corpus == 'kc':
            documents_gold = data_loader.dataset.joined_documents
        else:
            documents_gold = data_loader.dataset.documents

        result = {}
        for pas_target in self.pas_targets:
            scorer = Scorer(documents_pred, documents_gold,
                            target_cases=data_loader.dataset.target_cases,
                            target_exophors=data_loader.dataset.target_exophors,
                            coreference=data_loader.dataset.coreference,
                            bridging=data_loader.dataset.bridging,
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

    # get function handles of metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    tester = Tester(model, metric_fns, config, kwdlc_data_loader, kc_data_loader, commonsense_data_loader,
                    args.target, logger, args.predict_overt, args.confidence_threshold, args.result_suffix)

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
    parser.add_argument('-t', '--target', default='test', type=str, choices=['valid', 'test'],
                        help='evaluation target')
    parser.add_argument('--predict-overt', action='store_true', default=False,
                        help='calculate scores for overt arguments instead of using gold')
    parser.add_argument('--confidence-threshold', default=0.0, type=float,
                        help='threshold for argument existence [0, 1] (default: 0.0)')
    parser.add_argument('--result-suffix', default='', type=str,
                        help='custom evaluation result directory name')
    parser.add_help = True

    parsed_args = parser.parse_args()
    config_args = {'run_id': ''} if parsed_args.resume is None else {'inherit_save_dir': True}
    main(ConfigParser.from_parser(parser, **config_args), parsed_args)
