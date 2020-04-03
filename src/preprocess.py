"""Preprocess dataset."""
import argparse
from pathlib import Path
import _pickle as cPickle
import json
from typing import List

from kyoto_reader import KyotoReader

from data_loader.dataset.commonsense_dataset import CommonsenseExample


def process(input_path: Path, output_path: Path, cases: List[str], corefs: List[str]):
    output_path.mkdir(exist_ok=True)
    reader = KyotoReader(input_path, cases, corefs, extract_nes=False)
    documents = reader.process_all_documents()
    for document in documents:
        with output_path.joinpath(document.doc_id + '.pkl').open(mode='wb') as f:
            cPickle.dump(document, f)


def process_commonsense(input_path: Path, output_path: Path):
    examples = []
    with input_path.open() as f:
        for line in f:
            label, string = line.strip().split(',')
            former_string, latter_string = string.split('@')
            examples.append(CommonsenseExample(former_string, latter_string, bool(int(label))))
    with output_path.open(mode='wb') as f:
        cPickle.dump(examples, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kwdlc', type=str, default=None,
                        help='path to directory where KWDLC data exists')
    parser.add_argument('--kc', type=str, default=None,
                        help='path to directory where Kyoto Corpus data exists')
    parser.add_argument('--commonsense', type=str, default=None,
                        help='path to directory where commonsense inference data exists')
    parser.add_argument('--out', type=str, required=True,
                        help='path to directory where dataset to be located')
    parser.add_argument('--coref-string', type=str, default='=,=構,=≒,=構≒',
                        help='Coreference strings. Separate by ","')
    parser.add_argument('--case-string', type=str, default='ガ,ヲ,ニ,ガ２,ノ,ノ？,判ガ',
                        help='Case strings. Separate by ","')
    args = parser.parse_args()

    # make directories to save dataset
    Path(args.out).mkdir(exist_ok=True)

    target_cases: List[str] = args.case_string.split(',')
    target_corefs: List[str] = args.coref_string.split(',')
    config = {
        'target_cases': target_cases,
        'target_corefs': target_corefs,
        'pickle_ext': 'pkl'
    }

    if args.kwdlc is not None:
        kwdlc_dir = Path(args.kwdlc).resolve()
        output_dir = Path(args.out) / 'kwdlc'
        output_dir.mkdir(exist_ok=True)
        process(kwdlc_dir / 'train', output_dir / 'train', target_cases, target_corefs)
        process(kwdlc_dir / 'valid', output_dir / 'valid', target_cases, target_corefs)
        process(kwdlc_dir / 'test', output_dir / 'test', target_cases, target_corefs)

    if args.kc is not None:
        kc_dir = Path(args.kc).resolve()
        output_dir = Path(args.out) / 'kc'
        output_dir.mkdir(exist_ok=True)
        process(kc_dir / 'train', output_dir / 'train', target_cases, target_corefs)
        process(kc_dir / 'valid', output_dir / 'valid', target_cases, target_corefs)
        process(kc_dir / 'test', output_dir / 'test', target_cases, target_corefs)

    if args.commonsense is not None:
        commonsense_dir = Path(args.commonsense).resolve()
        output_dir = Path(args.out) / 'commonsense'
        output_dir.mkdir(exist_ok=True)
        process_commonsense(commonsense_dir / 'train.csv', output_dir / 'train.pkl')
        process_commonsense(commonsense_dir / 'valid.csv', output_dir / 'valid.pkl')
        process_commonsense(commonsense_dir / 'test.csv', output_dir / 'test.pkl')

    with Path(args.out).joinpath('config.json').open(mode='wt') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
