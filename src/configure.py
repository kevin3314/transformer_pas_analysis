import json
import math
import copy
import argparse
from typing import List
from pathlib import Path
import itertools
import inspect

import model.model as module_arch
import transformers.optimization as module_optim


class Config:
    def __init__(self, config_dir: Path) -> None:
        config_dir.mkdir(exist_ok=True)
        self.config_dir = config_dir
        self.log = []

    def write(self, **config) -> None:
        config_path = self.config_dir / f'{config["name"]}.json'
        if config_path in self.log:
            return
        with config_path.open('w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        self.log.append(config_path)
        print(config_path)


def main() -> None:
    all_models = [m[0] for m in inspect.getmembers(module_arch, inspect.isclass)
                  if m[1].__module__ == module_arch.__name__]
    all_lr_schedulers = [m[0][4:] for m in inspect.getmembers(module_optim, inspect.isfunction)
                         if m[1].__module__ == module_optim.__name__ and m[0].startswith('get_')]
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=(lambda p: Path(p)), default='config',
                        help='path to output directory')
    parser.add_argument('-d', '--dataset', type=(lambda p: Path(p)),
                        help='path to dataset directory')
    parser.add_argument('-m', '--model', choices=all_models, default=all_models, nargs='*',
                        help='model name')
    parser.add_argument('-e', '--epoch', type=int, default=[4], nargs='*',
                        help='number of training epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=16,
                        help='number of batch size')
    parser.add_argument('--max-bpg', type=int, default=None,
                        help='max batch size per GPU')
    parser.add_argument('--eval-batch-size', type=int, default=None,
                        help='number of batch size for evaluation (default: same as that of training)')
    parser.add_argument('--coreference', '--coref', action='store_true', default=False,
                        help='Perform coreference resolution.')
    parser.add_argument('--case-string', type=str, default='ガ,ヲ,ニ,ガ２',
                        help='Case strings. Separate by ","')
    parser.add_argument('--exophors', '--exo', type=str, default='著者,読者,不特定:人',
                        help='Special tokens. Separate by ",".')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout ratio')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate')
    parser.add_argument('--lr-schedule', choices=all_lr_schedulers, type=str, default='linear_schedule_with_warmup',
                        help='lr scheduler')
    parser.add_argument('--warmup-proportion', type=float, default=0.1,
                        help='Proportion of training to perform linear learning rate warmup for')
    parser.add_argument('--warmup-steps', type=int, default=None,
                        help='Linear warmup over warmup_steps.')
    parser.add_argument('--additional-name', type=str, default=None,
                        help='additional config file name')
    parser.add_argument('--gpus', type=int, default=2,
                        help='number of gpus to use')
    parser.add_argument('--refinement-type', '--rtype', type=int, choices=[1, 2, 3], default=1,
                        help='refinement layer type for RefinementModel')
    parser.add_argument('--save-start-epoch', type=int, default=1,
                        help='you can skip saving of initial checkpoints, which reduces writing overhead')
    parser.add_argument('--corpus', choices=['kwdlc', 'kc', 'all'], default=['all'], nargs='*',
                        help='corpus to use in training')
    parser.add_argument('--train-target', choices=['overt', 'case', 'zero'], default=['case', 'zero'], nargs='*',
                        help='dependency type to train')
    parser.add_argument('--eventive-noun', '--noun', action='store_true', default=False,
                        help='analyze eventive noun as predicate')
    parser.add_argument('--refinement-iter', '--riter', type=int, default=[3], nargs='*',
                        help='number of refinement iteration (IterativeRefinementModel)')
    parser.add_argument('--conditional-model', choices=['emb', 'atn', 'out'], default=['atn'], nargs='*',
                        help='how to insert pre-output to model (IterativeRefinementModel)')
    parser.add_argument('--output-aggr', choices=['hard', 'hard2', 'soft', 'confidence'], default=['hard'], nargs='*',
                        help='pre-output aggregation method (IterativeRefinementModel with AttentionConditionalModel)')
    parser.add_argument('--atn-target', choices=['k', 'v', 'kv'], default=['kv'], nargs='*',
                        help='rel embedding addition target (IterativeRefinementModel with AttentionConditionalModel)')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode')
    parser.add_argument('--disable-pas', action='store_true', default=False,
                        help='do not perform predicate argument structure analysis')
    args = parser.parse_args()

    config = Config(args.config)
    data_root: Path = args.dataset.resolve()
    with data_root.joinpath('config.json').open() as f:
        dataset_config = json.load(f)
    cases: List[str] = args.case_string.split(',') if args.case_string else []

    for model, corpus, n_epoch, conditional_model, output_aggr, refinement_iter, atn_target in \
            itertools.product(args.model, args.corpus, args.epoch, args.conditional_model, args.output_aggr,
                              args.refinement_iter, args.atn_target):
        items = [model]
        if 'IterativeRefinement' in model:
            items.append(refinement_iter)
        items += [corpus, f'{n_epoch}e', dataset_config['bert_name']]
        if args.disable_pas:
            items.append('nopas')
        if args.coreference:
            items.append('coref')
        if set(cases) - {'ノ'}:
            items.append(''.join(tgt[0] for tgt in ('overt', 'case', 'zero') if tgt in args.train_target))
        if 'ノ' in cases:
            items.append('nocase')
        if args.eventive_noun:
            items.append('noun')
        if model in ('RefinementModel', 'RefinementModel2'):
            items.append(f'{dataset_config["bert_name"]}{args.refinement_type}')
        if 'ConditionalModel' in model or 'IterativeRefinement' in model:
            items.append(conditional_model)
            if conditional_model == 'atn':
                items.append(output_aggr)
                items.append(atn_target)
        if args.debug:
            items.append('debug')
        if args.additional_name:
            items.append(args.additional_name)
        name = '-'.join(str(x) for x in items)

        num_train_examples = 0
        if corpus in ('kwdlc', 'all'):
            num_train_examples += dataset_config['num_examples']['kwdlc']['train']
        if corpus in ('kc', 'all'):
            num_train_examples += dataset_config['num_examples']['kc']['train']
        if model == 'CommonsenseModel':
            num_train_examples += dataset_config['num_examples']['commonsense']['train']

        arch = {
            'type': model,
            'args': {
                'bert_model': dataset_config['bert_path'],
                'dropout': args.dropout,
                'num_case': len(cases),
                'coreference': args.coreference,
            },
        }
        if model in ('RefinementModel', 'RefinementModel2'):
            arch['args'].update({'refinement_type': args.refinement_type,
                                 'refinement_bert_model': dataset_config['bert_path']})
        if 'IterativeRefinement' in model:
            arch['args'].update({'num_iter': refinement_iter})
        if 'ConditionalModel' in model or 'IterativeRefinement' in model:
            arch['args'].update({'conditional_model': conditional_model})
            if conditional_model == 'atn':
                arch['args'].update({'output_aggr': output_aggr, 'atn_target': atn_target})

        dataset = {
            'type': 'PASDataset',
            'args': {
                'path': None,
                'max_seq_length': dataset_config['max_seq_length'],
                'cases': cases,
                'exophors': args.exophors.split(','),
                'coreference': args.coreference,
                'dataset_config': dataset_config,
                'training': None,
                'bert_model': dataset_config['bert_path'],
                'kc': None,
                'train_target': args.train_target,
                'eventive_noun': args.eventive_noun,
                'disable_pas': args.disable_pas
            },
        }

        train_kwdlc_dataset = copy.deepcopy(dataset)
        train_kwdlc_dataset['args']['path'] = str(data_root / 'kwdlc' / 'train') if corpus in ('kwdlc', 'all') else None
        train_kwdlc_dataset['args']['training'] = True
        train_kwdlc_dataset['args']['kc'] = False

        valid_kwdlc_dataset = copy.deepcopy(dataset)
        valid_kwdlc_dataset['args']['path'] = str(data_root / 'kwdlc' / 'valid') if corpus in ('kwdlc', 'all') else None
        valid_kwdlc_dataset['args']['training'] = False
        valid_kwdlc_dataset['args']['kc'] = False

        test_kwdlc_dataset = copy.deepcopy(dataset)
        test_kwdlc_dataset['args']['path'] = str(data_root / 'kwdlc' / 'test') if corpus in ('kwdlc', 'all') else None
        test_kwdlc_dataset['args']['training'] = False
        test_kwdlc_dataset['args']['kc'] = False

        train_kc_dataset = copy.deepcopy(train_kwdlc_dataset)
        train_kc_dataset['args']['path'] = str(data_root / 'kc' / 'train') if corpus in ('kc', 'all') else None
        train_kc_dataset['args']['kc'] = True

        valid_kc_dataset = copy.deepcopy(valid_kwdlc_dataset)
        valid_kc_dataset['args']['path'] = str(data_root / 'kc' / 'valid') if corpus in ('kc', 'all') else None
        valid_kc_dataset['args']['kc'] = True

        test_kc_dataset = copy.deepcopy(test_kwdlc_dataset)
        test_kc_dataset['args']['path'] = str(data_root / 'kc' / 'test') if corpus in ('kc', 'all') else None
        test_kc_dataset['args']['kc'] = True

        if model == 'CommonsenseModel':
            commonsense_dataset = {
                'type': 'CommonsenseDataset',
                'args': {
                    'path': None,
                    'max_seq_length': dataset_config['max_seq_length'],
                    'num_special_tokens': len(args.exophors.split(',')) + 1 + int(args.coreference),
                    'bert_model': dataset_config['bert_path'],
                },
            }
            train_commonsense_dataset = copy.deepcopy(commonsense_dataset)
            train_commonsense_dataset['args']['path'] = str(data_root / 'commonsense' / 'train.pkl')
            valid_commonsense_dataset = copy.deepcopy(commonsense_dataset)
            valid_commonsense_dataset['args']['path'] = str(data_root / 'commonsense' / 'valid.pkl')
            test_commonsense_dataset = copy.deepcopy(commonsense_dataset)
            test_commonsense_dataset['args']['path'] = str(data_root / 'commonsense' / 'test.pkl')
        else:
            train_commonsense_dataset = None
            valid_commonsense_dataset = None
            test_commonsense_dataset = None

        data_loader = {
            'type': 'PASDataLoader',
            'args': {
                'batch_size': None,
                'shuffle': None,
                'validation_split': 0.0,
                'num_workers': 2,
            },
        }

        train_data_loader = copy.deepcopy(data_loader)
        train_data_loader['args']['shuffle'] = (not args.debug)

        valid_data_loader = copy.deepcopy(data_loader)
        valid_data_loader['args']['batch_size'] = args.eval_batch_size if args.eval_batch_size else args.batch_size
        valid_data_loader['args']['shuffle'] = False

        test_data_loader = copy.deepcopy(data_loader)
        test_data_loader['args']['batch_size'] = args.eval_batch_size if args.eval_batch_size else args.batch_size
        test_data_loader['args']['shuffle'] = False

        optimizer = {
            'type': 'AdamW',
            'args': {
                'lr': args.lr,
                'eps': 1e-8,
                'weight_decay': 0.01,
            },
        }

        metrics = []
        if 'ガ' in cases:
            metrics.append('case_analysis_f1_ga')
        if 'ヲ' in cases:
            metrics.append('case_analysis_f1_wo')
        if 'ニ' in cases:
            metrics.append('case_analysis_f1_ni')
        if 'ガ２' in cases:
            metrics.append('case_analysis_f1_ga2')
        if any(met.startswith('case_analysis_f1_') for met in metrics):
            metrics.append('case_analysis_f1')
        if 'ガ' in cases:
            metrics.append('zero_anaphora_f1_ga')
        if 'ヲ' in cases:
            metrics.append('zero_anaphora_f1_wo')
        if 'ニ' in cases:
            metrics.append('zero_anaphora_f1_ni')
        if 'ガ２' in cases:
            metrics.append('zero_anaphora_f1_ga2')
        if any(met.startswith('zero_anaphora_f1_') for met in metrics):
            metrics += [
                'zero_anaphora_f1_inter',
                'zero_anaphora_f1_intra',
                'zero_anaphora_f1_exophora',
                'zero_anaphora_f1',
            ]
        if args.coreference:
            metrics.append('coreference_f1')
        if 'ノ' in cases:
            metrics.append('bridging_anaphora_f1')

        t_total = math.ceil(num_train_examples / args.batch_size) * n_epoch
        warmup_steps = args.warmup_steps if args.warmup_steps is not None else t_total * args.warmup_proportion
        lr_scheduler = {'type': 'get_' + args.lr_schedule}
        if args.lr_schedule == 'constant_schedule':
            lr_scheduler['args'] = {}
        elif args.lr_schedule == 'constant_schedule_with_warmup':
            lr_scheduler['args'] = {'num_warmup_steps': warmup_steps}
        elif args.lr_schedule in ('linear_schedule_with_warmup',
                                  'cosine_schedule_with_warmup',
                                  'cosine_with_hard_restarts_schedule_with_warmup'):
            lr_scheduler['args'] = {'num_warmup_steps': warmup_steps, 'num_training_steps': t_total}
        else:
            raise ValueError(f'unknown lr schedule: {args.lr_schedule}')

        mnt_mode = 'max'
        if 'zero_anaphora_f1' in metrics:
            mnt_metric = 'zero_anaphora_f1'
        elif 'coreference_f1' in metrics:
            mnt_metric = 'coreference_f1'
        elif 'bridging_anaphora_f1' in metrics:
            mnt_metric = 'bridging_anaphora_f1'
        else:
            mnt_metric = 'loss'
            mnt_mode = 'min'
        if corpus in ('kwdlc', 'all'):
            mnt_metric = 'val_kwdlc_' + mnt_metric
        else:
            mnt_metric = 'val_kc_' + mnt_metric
        trainer = {
            'epochs': n_epoch,
            'batch_size': args.batch_size,
            'max_bpg': args.max_bpg if args.max_bpg is not None else args.batch_size,
            'save_dir': 'result/',
            'save_start_epoch': args.save_start_epoch,
            'verbosity': 2,
            'monitor': f'{mnt_mode} {mnt_metric}',
            'early_stop': 10,
            'tensorboard': True,
        }

        config.write(
            name=name,
            n_gpu=args.gpus,
            arch=arch,
            train_kwdlc_dataset=train_kwdlc_dataset,
            train_kc_dataset=train_kc_dataset,
            train_commonsense_dataset=train_commonsense_dataset,
            valid_kwdlc_dataset=valid_kwdlc_dataset,
            valid_kc_dataset=valid_kc_dataset,
            valid_commonsense_dataset=valid_commonsense_dataset,
            test_kwdlc_dataset=test_kwdlc_dataset,
            test_kc_dataset=test_kc_dataset,
            test_commonsense_dataset=test_commonsense_dataset,
            train_data_loader=train_data_loader,
            valid_data_loader=valid_data_loader,
            test_data_loader=test_data_loader,
            optimizer=optimizer,
            metrics=metrics,
            lr_scheduler=lr_scheduler,
            trainer=trainer,
        )


if __name__ == '__main__':
    main()
