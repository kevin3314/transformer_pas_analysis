import os
import json
import math
import copy
import pathlib
import argparse
from typing import List
import itertools


class Config:
    def __init__(self, **kwargs) -> None:
        self.config = {}
        for key, value in kwargs.items():
            self.config.update({key: value})

    def dump(self, config_dir: pathlib.Path) -> None:
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / f'{self.config["name"]}.json'
        with config_path.open('w') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(config_path)


class Path:
    bert_model = {
        'local': {
            'base': f'{os.environ["HOME"]}/Data/bert/Wikipedia/L-12_H-768_A-12_E-30_BPE',
        },
        'server': {
            'base': '/larch/share/bert/Japanese_models/Wikipedia/L-12_H-768_A-12_E-30_BPE',
            'large': '/larch/share/bert/Japanese_models/Wikipedia/L-24_H-1024_A-16_E-25_BPE',
            'large-wwm': '/larch/share/bert/Japanese_models/Wikipedia/L-24_H-1024_A-16_E-30_BPE_WWM',
        }
    }


def main() -> None:
    all_models = ['BaselineModel', 'DependencyModel', 'LayerAttentionModel', 'MultitaskDepModel',
                  'CaseInteractionModel', 'CaseInteractionModel2', 'RefinementModel', 'EnsembleModel',
                  'CommonsenseModel', 'BaselineModel2', 'BaselineModel3']
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='path to output directory')
    parser.add_argument('-d', '--dataset', type=str,
                        help='path to dataset directory')
    parser.add_argument('-m', '--model', choices=all_models, default=all_models, nargs='*',
                        help='model name')
    parser.add_argument('-e', '--epoch', type=int, default=3, nargs='*',
                        help='number of training epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='number of batch size')
    parser.add_argument('--max-seq-length', type=int, default=128,
                        help='The maximum total input sequence length after WordPiece tokenization. Sequences '
                             'longer than this will be truncated, and sequences shorter than this will be padded.')
    parser.add_argument('--coreference', action='store_true', default=False,
                        help='Perform coreference resolution.')
    parser.add_argument('--case-string', type=str, default='ガ,ヲ,ニ,ガ２',
                        help='Case strings. Separate by ","')
    parser.add_argument('--exophors', type=str, default='著者,読者,不特定:人',
                        help='Special tokens. Separate by ",".')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout ratio')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate')
    parser.add_argument('--lr-schedule', type=str, default='linear_schedule_with_warmup',
                        choices=['constant_schedule', 'constant_schedule_with_warmup', 'linear_schedule_with_warmup',
                                 'cosine_schedule_with_warmup', 'cosine_with_hard_restarts_schedule_with_warmup'],
                        help='lr scheduler')
    parser.add_argument('--warmup-proportion', default=0.1, type=float,
                        help='Proportion of training to perform linear learning rate warmup for. '
                             'E.g., 0.1 = 10% of training.')
    parser.add_argument("--warmup-steps", default=None, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--env', choices=['local', 'server'], default='server',
                        help='development environment')
    parser.add_argument('--additional-name', type=str, default=None,
                        help='additional config file name')
    parser.add_argument('--gpus', type=int, default=2,
                        help='number of gpus to use')
    parser.add_argument('--bert', choices=['base', 'large', 'large-wwm'], default='base',
                        help='BERT model')
    parser.add_argument('--refinement-bert', choices=['base', 'large', 'large-wwm'], default='base',
                        help='BERT model type used for RefinementModel')
    parser.add_argument('--refinement-type', type=int, default=1, choices=[1, 2, 3],
                        help='refinement layer type for RefinementModel')
    parser.add_argument('--save-start-epoch', type=int, default=1,
                        help='you can skip saving of initial checkpoints, which reduces writing overhead')
    parser.add_argument('--corpus', choices=['kwdlc', 'kc', 'all'], default=['kwdlc', 'kc', 'all'], nargs='*',
                        help='corpus to use in training')
    parser.add_argument('--train-target', choices=['overt', 'case', 'zero'], default=['case', 'zero'], nargs='*',
                        help='dependency type to train')
    parser.add_argument('--eventive-noun', action='store_true', default=False,
                        help='analyze eventive noun as predicate')
    args = parser.parse_args()

    config_dir = pathlib.Path(args.config)
    bert_model = Path.bert_model[args.env][args.bert]
    data_root = pathlib.Path(args.dataset).resolve()
    with data_root.joinpath('config.json').open() as f:
        dataset_config = json.load(f)
    cases: List[str] = args.case_string.split(',') if args.case_string else []

    for model, corpus, n_epoch in itertools.product(args.model, args.corpus, args.epoch):
        name = f'{model}-{corpus}-{n_epoch}e-{args.bert}'
        name += '-coref' if args.coreference else ''
        name += '-' + ''.join(tgt[0] for tgt in ('overt', 'case', 'zero') if tgt in args.train_target)
        name += '-nocase' if 'ノ' in cases else ''
        name += '-noun' if args.eventive_noun else ''
        if model == 'RefinementModel':
            name += '-largeref' if args.refinement_bert in ('large', 'large-wwm') else ''
            name += '-reftype' + str(args.refinement_type)
        name += f'-{args.additional_name}' if args.additional_name is not None else ''

        train_kwdlc_dir = data_root / 'kwdlc' / 'train'
        train_kc_dir = data_root / 'kc' / 'train'
        glob_pat = '*.' + dataset_config['pickle_ext']
        num_train_examples = 0
        if corpus in ('kwdlc', 'all'):
            num_train_examples += sum(1 for _ in train_kwdlc_dir.glob(glob_pat))
        if corpus in ('kc', 'all'):
            num_train_examples += sum(1 for _ in train_kc_dir.glob(glob_pat))

        arch = {
            'type': model,
            'args': {
                'bert_model': bert_model,
                'dropout': args.dropout,
                'num_case': len(cases),
                'coreference': args.coreference,
            },
        }
        if model == 'RefinementModel':
            refinement_bert_model = Path.bert_model[args.env][args.refinement_bert]
            arch['args'].update({'refinement_type': args.refinement_type,
                                 'refinement_bert_model': refinement_bert_model})

        dataset = {
            'type': 'PASDataset',
            'args': {
                'path': None,
                'max_seq_length': args.max_seq_length,
                'cases': cases,
                'exophors': args.exophors.split(','),
                'coreference': args.coreference,
                'dataset_config': dataset_config,
                'training': None,
                'bert_model': bert_model,
                'kc': None,
                'train_target': args.train_target,
                'eventive_noun': args.eventive_noun,
            },
        }

        train_kwdlc_dataset = copy.deepcopy(dataset)
        train_kwdlc_dataset['args']['path'] = str(train_kwdlc_dir) if corpus in ('kwdlc', 'all') else None
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
        train_kc_dataset['args']['path'] = str(train_kc_dir) if corpus in ('kc', 'all') else None
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
                    'max_seq_length': args.max_seq_length,
                    'bert_model': bert_model,
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
                'batch_size': args.batch_size,
                'shuffle': None,
                'validation_split': 0.0,
                'num_workers': 2,
            },
        }

        train_data_loader = copy.deepcopy(data_loader)
        train_data_loader['args']['shuffle'] = True

        valid_data_loader = copy.deepcopy(data_loader)
        valid_data_loader['args']['shuffle'] = False

        test_data_loader = copy.deepcopy(data_loader)
        test_data_loader['args']['shuffle'] = False

        optimizer = {
            'type': 'AdamW',
            'args': {
                'lr': args.lr,
                'eps': 1e-8,
                'weight_decay': 0.01,
            },
        }

        if model == 'MultitaskDepModel':
            loss = 'cross_entropy_pas_dep_loss'
        elif model in ('CaseInteractionModel2', 'RefinementModel', 'EnsembleModel'):
            loss = 'multi_cross_entropy_pas_loss'
        elif model == 'CommonsenseModel':
            loss = 'cross_entropy_pas_commonsense_loss'
        else:
            loss = 'cross_entropy_pas_loss'

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
            metrics += [
                'bridging_anaphora_f1_inter',
                'bridging_anaphora_f1_intra',
                'bridging_anaphora_f1_exophora',
                'bridging_anaphora_f1',
            ]

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
            'save_dir': 'result/',
            'save_start_epoch': args.save_start_epoch,
            'verbosity': 2,
            'monitor': f'{mnt_mode} {mnt_metric}',
            'early_stop': 10,
            'tensorboard': True,
        }

        config = Config(
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
            loss=loss,
            metrics=metrics,
            lr_scheduler=lr_scheduler,
            trainer=trainer,
        )
        config.dump(config_dir)


if __name__ == '__main__':
    main()
