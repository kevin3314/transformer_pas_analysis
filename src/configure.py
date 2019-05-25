import os
import json
import math
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
    train_file = {
        'local': '/Users/NobuhiroUeda/PycharmProjects/bert_pas_analysis/data/sample.train.conll',
        'server': '/share/tool/nn_based_anaphora_resolution/corpus/kwdlc/conll/latest/train.conll'
    }
    valid_file = {
        'local': '/Users/NobuhiroUeda/PycharmProjects/bert_pas_analysis/data/sample.dev.conll',
        'server': '/share/tool/nn_based_anaphora_resolution/corpus/kwdlc/conll/latest/dev.conll'
    }
    test_file = {
        'local': '/Users/NobuhiroUeda/PycharmProjects/bert_pas_analysis/data/sample.test.conll',
        'server': '/share/tool/nn_based_anaphora_resolution/corpus/kwdlc/conll/latest/test.conll'
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        help='path to output directory')
    # parser.add_argument('--model',
    #                     choices=['BaselineModel', 'EntityBufferModel'],
    #                     default='EntityBufferModel',
    #                     nargs='*',
    #                     help='model name')
    parser.add_argument('--epoch', '-e', type=int, default=3,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='number of batch size')
    parser.add_argument('--max-seq-length', type=int, default=512,
                        help='The maximum total input sequence length after WordPiece tokenization. Sequences '
                             'longer than this will be truncated, and sequences shorter than this will be padded.')
    parser.add_argument('--coreference', action='store_true', default=False,
                        help='Perform coreference resolution.')
    parser.add_argument('--special-tokens', type=str, default='著者,読者,不特定:人,NULL,NA',
                        help='Special tokens. Separate by ",".')
    parser.add_argument('--case-string', type=str, default="ガ,ヲ,ニ,ガ２",
                        help='Case strings. Separate by ","')
    # parser.add_argument('--dropout', type=float, default=0.1, nargs='*',
    #                     help='dropout ratio')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate')
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10% of training.")
    parser.add_argument('--env', choices=['local', 'server'], default='server',
                        help='development environment')
    parser.add_argument('--additional-name', type=str, default=None,
                        help='additional config file name')
    args = parser.parse_args()

    os.makedirs(args.config, exist_ok=True)
    cases: List[str] = args.case_string.split(',')

    name = 'PAS' + ('' if args.additional_name is None else args.additional_name)
    n_gpu = 1
    with open(Path.train_file[args.env]) as f:
        num_train_examples = f.readlines().count('\n') + 1

    arch = {
        "type": "BertPASAnalysisModel",
        "args": {
            "bert_model": Path.bert_model[args.env],
            "parsing_algorithm": "zhang",
            "num_case": len(cases) if not args.coreference else len(cases) + 1,
            "arc_representation_dim": 400,
        },
    }
    train_dataset = {
        "type": "PASDataset",
        "args": {
            "path": Path.train_file[args.env],
            "max_seq_length": args.max_seq_length,
            "cases": cases,
            "coreference": args.coreference,
            "special_tokens": args.special_tokens.split(','),
            "training": True,
            "bert_model": Path.bert_model[args.env],
        },
    }
    valid_dataset = {
        "type": "PASDataset",
        "args": {
            "path": Path.valid_file[args.env],
            "max_seq_length": args.max_seq_length,
            "cases": cases,
            "coreference": args.coreference,
            "special_tokens": args.special_tokens.split(','),
            "training": False,
            "bert_model": Path.bert_model[args.env],
        },
    }
    test_dataset = {
        "type": "PASDataset",
        "args": {
            "path": Path.test_file[args.env],
            "max_seq_length": args.max_seq_length,
            "cases": cases,
            "coreference": args.coreference,
            "special_tokens": args.special_tokens.split(','),
            "training": False,
            "bert_model": Path.bert_model[args.env],
        },
    }
    train_data_loader = {
        "type": "ConlluDataLoader",
        "args": {
            "batch_size": args.batch_size,
            "shuffle": True,
            "validation_split": 0.0,
            "num_workers": 2,
        },
    }
    valid_data_loader = {
        "type": "ConlluDataLoader",
        "args": {
            "batch_size": args.batch_size,
            "shuffle": False,
            "validation_split": 0.0,
            "num_workers": 2,
        },
    }
    test_data_loader = {
        "type": "ConlluDataLoader",
        "args": {
            "batch_size": args.batch_size,
            "shuffle": False,
            "validation_split": 0.0,
            "num_workers": 2,
        },
    }
    optimizer = {
        "type": "BertAdam",
        "args": {
            "lr": args.lr,
            "warmup": args.warmup_proportion,
            "t_total": math.ceil(num_train_examples / args.batch_size) * args.epoch,
            "weight_decay": 0.01,
        },
    }
    loss = "cross_entropy_loss"
    metrics = [
        "my_metric", "my_metric2",
    ]
    # lr_scheduler = {
    #     "type": "StepLR",
    #     "args": {
    #         "step_size": 50,
    #         "gamma": 0.1,
    #     },
    # }
    trainer = {
        "epochs": args.epoch,
        "save_dir": "result/",
        "save_period": 3,
        "verbosity": 2,
        "monitor": "off",
        "early_stop": 10,
        "tensorboardX": True,
    }
    visualization = {
        "tensorboardX": True,
        "log_dir": "result/runs"
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
        # lr_scheduler=lr_scheduler,
        trainer=trainer,
        visualization=visualization,
    )
    config.dump(args.config)


if __name__ == '__main__':
    main()
