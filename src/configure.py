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

    def dump(self, config_dir: pathlib.Path, base_name: str) -> None:
        config_dir.mkdir(exist_ok=True, parents=True)
        config_path = config_dir / f'{base_name}.json'
        with config_path.open('w') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        print(config_path)


class Path:
    bert_model = {
        'local': {
            'base': '/Users/NobuhiroUeda/Data/bert/Wikipedia/L-12_H-768_A-12_E-30_BPE',
            'large': None
        },
        'server': {
            'base': '/larch/share/bert/Japanese_models/Wikipedia/L-12_H-768_A-12_E-30_BPE',
            'large': '/larch/share/bert/Japanese_models/Wikipedia/L-24_H-1024_A-16_E-25_BPE'
        }
    }


def main() -> None:
    all_models = ['BaselineModel', 'DependencyModel', 'LayerAttentionModel', 'MultitaskDepModel',
                  'CaseInteractionModel', 'RefinementModel', 'RefinementTwiceModel']
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
    parser.add_argument('--exophors', type=str, default='著者,読者,不特定:人',
                        help='Special tokens. Separate by ",".')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout ratio')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate')
    parser.add_argument('--lr-schedule', type=str, default='WarmupLinearSchedule',
                        choices=['ConstantLRSchedule', 'WarmupConstantSchedule', 'WarmupLinearSchedule',
                                 'WarmupCosineSchedule', 'WarmupCosineWithHardRestartsSchedule'],
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
    parser.add_argument('--use-bert-large', action='store_true', default=False,
                        help='whether to use BERT_LARGE model')
    parser.add_argument('--refinement-bert', choices=['base', 'large'], default='base',
                        help='BERT model type used for refinement model')
    parser.add_argument('--refinement-type', type=int, default=1, choices=[1, 2, 3],
                        help='refinement layer type for RefinementModel')
    parser.add_argument('--no-save-model', action='store_true', default=False,
                        help='whether to save trained model')
    parser.add_argument('--corpus', choices=['kwdlc', 'kc', 'all'], default=['kwdlc', 'kc', 'all'], nargs='*',
                        help='corpus to use in training')
    parser.add_argument('--train-target', choices=['overt', 'case', 'zero'], default=['case', 'zero'], nargs='*',
                        help='dependency type to train')
    parser.add_argument('--eventive-noun', action='store_true', default=False,
                        help='analyze eventive noun as predicate')
    args = parser.parse_args()

    bert_model = Path.bert_model[args.env]['large' if args.use_bert_large else 'base']
    n_gpu: int = args.gpus
    data_root = pathlib.Path(args.dataset).resolve()
    with data_root.joinpath('config.json').open() as f:
        dataset_config = json.load(f)
    cases: List[str] = dataset_config['target_cases']
    corefs: List[str] = dataset_config['target_corefs']

    for model, corpus, n_epoch in itertools.product(args.model, args.corpus, args.epoch):
        config_dir = pathlib.Path(model) / corpus / f'{n_epoch}e'
        base_name = 'large' if args.use_bert_large else 'base'
        base_name += '-coref' if args.coreference else ''
        base_name += '-' + ''.join(tgt[0] for tgt in ('overt', 'case', 'zero') if tgt in args.train_target)
        base_name += '-nocase' if 'ノ' in cases else ''
        base_name += '-noun' if args.eventive_noun else ''
        if 'Refinement' in model:
            base_name += '-largeref' if args.refinement_bert == 'large' else ''
            base_name += '-reftype' + str(args.refinement_type)
        base_name += f'-{args.additional_name}' if args.additional_name is not None else ''

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
        if 'Refinement' in model:
            refinement_bert_model = Path.bert_model[args.env][args.refinement_bert]
            arch['args'].update({'refinement_type': args.refinement_type,
                                 'refinement_bert_model': refinement_bert_model})

        dataset = {
            'type': 'PASDataset',
            'args': {
                'path': None,
                'max_seq_length': args.max_seq_length,
                'cases': cases,
                'corefs': corefs,
                'coreference': args.coreference,
                'exophors': args.exophors.split(','),
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

        data_loader = {
            'type': 'ConllDataLoader',
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
                'weight_decay': 0.01,
            },
        }

        if model == 'MultitaskDepModel':
            loss = 'cross_entropy_pas_dep_loss'
        elif 'Refinement' in model:
            loss = 'multi_cross_entropy_pas_loss'
        else:
            loss = 'cross_entropy_pas_loss'

        metrics = [
            'case_analysis_f1_ga',
            'case_analysis_f1_wo',
            'case_analysis_f1_ni',
            'case_analysis_f1_ga2',
            'case_analysis_f1',
            'zero_anaphora_f1_ga',
            'zero_anaphora_f1_wo',
            'zero_anaphora_f1_ni',
            'zero_anaphora_f1_ga2',
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
        warmup_steps = t_total * args.warmup_proportion if args.warmup_steps is None else args.warmup_steps
        lr_scheduler = {'type': args.lr_schedule}
        if args.lr_schedule == 'ConstantLRSchedule':
            lr_scheduler['args'] = {}
        elif args.lr_schedule == 'WarmupConstantSchedule':
            lr_scheduler['args'] = {'warmup_steps': warmup_steps}
        elif args.lr_schedule in ('WarmupLinearSchedule',
                                  'WarmupCosineSchedule',
                                  'WarmupCosineWithHardRestartsSchedule'):
            lr_scheduler['args'] = {'warmup_steps': warmup_steps, 't_total': t_total}
        else:
            raise ValueError(f'unknown lr schedule: {args.lr_schedule}')

        trainer = {
            'epochs': n_epoch,
            'save_dir': 'result/',
            'save_period': 1,
            'save_model': not args.no_save_model,
            'verbosity': 2,
            'monitor': 'max val_kwdlc_zero_anaphora_f1',
            'early_stop': 10,
            'tensorboard': True,
        }

        config = Config(
            name=str(config_dir / base_name).replace('/', '-'),
            n_gpu=n_gpu,
            arch=arch,
            train_kwdlc_dataset=train_kwdlc_dataset,
            train_kc_dataset=train_kc_dataset,
            valid_kwdlc_dataset=valid_kwdlc_dataset,
            valid_kc_dataset=valid_kc_dataset,
            test_kwdlc_dataset=test_kwdlc_dataset,
            test_kc_dataset=test_kc_dataset,
            train_data_loader=train_data_loader,
            valid_data_loader=valid_data_loader,
            test_data_loader=test_data_loader,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            lr_scheduler=lr_scheduler,
            trainer=trainer,
        )
        config.dump(args.config / config_dir, base_name)


if __name__ == '__main__':
    main()
