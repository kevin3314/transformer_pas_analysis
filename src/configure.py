import os
import json
import math
import glob
import argparse
from typing import List


class Config:
    def __init__(self, **kwargs) -> None:
        self.config = {}
        for key, value in kwargs.items():
            self.config.update({key: value})
        self.uid = self.make_uid(self.config)

    def dump(self, path: str) -> None:
        with open(os.path.join(path, f'{self.uid}.json'), 'w') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    @staticmethod
    def make_uid(config: dict):
        return config['name']


class Path:
    bert_model = {
        'local': '/Users/NobuhiroUeda/Data/bert/Wikipedia/L-12_H-768_A-12_E-30_BPE',
        'server': '/mnt/larch/share/bert/Japanese_models/Wikipedia/L-12_H-768_A-12_E-30_BPE'
    }
    bert_large_model = {
        'local': None,
        'server': '/larch/share/bert/Japanese_models/Wikipedia/L-24_H-1024_A-16_E-20_BPE'
    }
    train_dir = {
        'local': '/Users/NobuhiroUeda/PycharmProjects/bert_pas_analysis/data/kwdlc/train',
        # 'server': '/share/tool/nn_based_anaphora_resolution/corpus/kwdlc/conll/latest/train.conll'
        'server': '/mnt/hinoki/ueda/kwdlc/old/train'
    }
    valid_dir = {
        'local': '/Users/NobuhiroUeda/PycharmProjects/bert_pas_analysis/data/kwdlc/valid',
        # 'server': '/share/tool/nn_based_anaphora_resolution/corpus/kwdlc/conll/latest/dev.conll'
        'server': '/mnt/hinoki/ueda/kwdlc/old/valid'
    }
    test_dir = {
        'local': '/Users/NobuhiroUeda/PycharmProjects/bert_pas_analysis/data/kwdlc/test',
        # 'server': '/share/tool/nn_based_anaphora_resolution/corpus/kwdlc/conll/latest/test.conll'
        'server': '/mnt/hinoki/ueda/kwdlc/old/test'
    }


def main() -> None:
    all_models = ['BaselineModel', 'BaseAsymModel', 'DependencyModel', 'LayerAttentionModel', 'MultitaskDepModel']
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='path to output directory')
    parser.add_argument('--model', choices=all_models, default=all_models, nargs='*',
                        help='model name')
    parser.add_argument('--epoch', '-e', type=int, default=3,
                        help='number of training epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='number of batch size')
    parser.add_argument('--max-seq-length', type=int, default=512,
                        help='The maximum total input sequence length after WordPiece tokenization. Sequences '
                             'longer than this will be truncated, and sequences shorter than this will be padded.')
    parser.add_argument('--coreference', action='store_true', default=False,
                        help='Perform coreference resolution.')
    parser.add_argument('--exophors', type=str, default='著者,読者,不特定:人',
                        help='Special tokens. Separate by ",".')
    parser.add_argument('--case-string', type=str, default='ガ,ヲ,ニ,ガ２',
                        help='Case strings. Separate by ","')
    # parser.add_argument('--dropout', type=float, default=0.1, nargs='*',
    #                     help='dropout ratio')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate')
    parser.add_argument('--warmup-proportion', default=0.1, type=float,
                        help='Proportion of training to perform linear learning rate warmup for. '
                             'E.g., 0.1 = 10% of training.')
    parser.add_argument('--env', choices=['local', 'server'], default='server',
                        help='development environment')
    parser.add_argument('--additional-name', type=str, default=None,
                        help='additional config file name')
    parser.add_argument('--gpus', type=int, default=2,
                        help='number of gpus to use')
    parser.add_argument('--use-bert-large', action='store_true', default=False,
                        help='whether to use BERT_LARGE model')
    parser.add_argument('--no-save-model', action='store_true', default=False,
                        help='whether to save trained model')
    args = parser.parse_args()

    os.makedirs(args.config, exist_ok=True)
    cases: List[str] = args.case_string.split(',')
    num_train_examples = len(glob.glob(os.path.join(Path.train_dir[args.env], '*.knp')))

    models: List[str] = args.model if type(args.model) == list else [args.model]
    n_gpu: int = args.gpus
    bert_model: dict = Path.bert_large_model if args.use_bert_large else Path.bert_model

    for model in models:
        name = model + ('' if args.additional_name is None else args.additional_name)

        arch = {
            'type': model,
            'args': {
                'bert_model': bert_model[args.env],
                'parsing_algorithm': 'zhang',
                'num_case': len(cases) if not args.coreference else len(cases) + 1,
                'arc_representation_dim': 400,
            },
        }
        train_dataset = {
            'type': 'PASDataset',
            'args': {
                'path': Path.train_dir[args.env],
                'max_seq_length': args.max_seq_length,
                'cases': cases,
                'coreference': args.coreference,
                'exophors': args.exophors.split(','),
                'training': True,
                'bert_model': bert_model[args.env],
            },
        }
        valid_dataset = {
            'type': 'PASDataset',
            'args': {
                'path': Path.valid_dir[args.env],
                'max_seq_length': args.max_seq_length,
                'cases': cases,
                'coreference': args.coreference,
                'exophors': args.exophors.split(','),
                'training': False,
                'bert_model': bert_model[args.env],
            },
        }
        test_dataset = {
            'type': 'PASDataset',
            'args': {
                'path': Path.test_dir[args.env],
                'max_seq_length': args.max_seq_length,
                'cases': cases,
                'coreference': args.coreference,
                'exophors': args.exophors.split(','),
                'training': False,
                'bert_model': bert_model[args.env],
            },
        }
        train_data_loader = {
            'type': 'ConllDataLoader',
            'args': {
                'batch_size': args.batch_size,
                'shuffle': True,
                'validation_split': 0.0,
                'num_workers': 2,
            },
        }
        valid_data_loader = {
            'type': 'ConllDataLoader',
            'args': {
                'batch_size': args.batch_size,
                'shuffle': False,
                'validation_split': 0.0,
                'num_workers': 2,
            },
        }
        test_data_loader = {
            'type': 'ConllDataLoader',
            'args': {
                'batch_size': args.batch_size,
                'shuffle': False,
                'validation_split': 0.0,
                'num_workers': 2,
            },
        }
        optimizer = {
            'type': 'BertAdam',
            'args': {
                'lr': args.lr,
                'warmup': args.warmup_proportion,
                't_total': math.ceil(num_train_examples / args.batch_size) * args.epoch,
                'schedule': 'warmup_linear',
                'weight_decay': 0.01,
            },
        }
        if model == 'MultitaskDepModel':
            loss = 'cross_entropy_pas_dep_loss'
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
            'zero_anaphora_f1_writer_reader',
            'zero_anaphora_f1',
        ]
        trainer = {
            'epochs': args.epoch,
            'save_dir': 'result/',
            'save_period': 1,
            'save_model': not args.no_save_model,
            'verbosity': 2,
            'monitor': 'max val_zero_anaphora_f1',
            'early_stop': 10,
            'tensorboard': True,
        }
        config = Config(
            name=name,
            n_gpu=n_gpu,
            arch=arch,
            train_dataset=train_dataset,
            train_data_loader=train_data_loader,
            valid_dataset=valid_dataset,
            valid_data_loader=valid_data_loader,
            test_dataset=test_dataset,
            test_data_loader=test_data_loader,
            # drawer=drawer,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            trainer=trainer,
        )
        config.dump(args.config)


if __name__ == '__main__':
    main()
