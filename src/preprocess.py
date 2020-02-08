"""Preprocess dataset."""
import argparse
from pathlib import Path
import pickle
import json
from typing import List

from kwdlc_reader import KWDLCReader


def process(input_path: Path, output_path: Path, cases: List[str], corefs: List[str]):
    output_path.mkdir(exist_ok=True)
    reader = KWDLCReader(input_path, cases, corefs, extract_nes=False)
    documents = reader.process_all_documents()
    for document in documents:
        with output_path.joinpath(document.doc_id + '.pkl').open(mode='wb') as f:
            pickle.dump(document, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kwdlc', type=str, default=None,
                        help='path to directory where KWDLC data exists')
    parser.add_argument('--kc', type=str, default=None,
                        help='path to directory where Kyoto Corpus data exists')
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

    with Path(args.out).joinpath('config.json').open(mode='wt') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
