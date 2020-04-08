"""Preprocess dataset."""
import argparse
import json
from pathlib import Path
import _pickle as cPickle
from typing import List, Dict
from collections import defaultdict

from pyknp import BList
from kyoto_reader import KyotoReader
from transformers import BertTokenizer

from data_loader.dataset.commonsense_dataset import CommonsenseExample

MAX_SUBWORD_LEN = 121


def process_kwdlc(input_path: Path, output_path: Path, cases: List[str], corefs: List[str]):
    output_path.mkdir(exist_ok=True)
    reader = KyotoReader(input_path, cases, corefs, extract_nes=False)
    documents = reader.process_all_documents()
    for document in documents:
        with output_path.joinpath(document.doc_id + '.pkl').open(mode='wb') as f:
            cPickle.dump(document, f)


def split_kc(input_dir: Path, output_dir: Path, tokenizer: BertTokenizer):
    """
    各文書を，tokenize したあとの長さが MAX_SUBWORD_LEN 以下になるように複数の文書に分割する．
    1文に分割しても MAX_SUBWORD_LEN を超えるような，長い文はそのまま出力する
    """
    did2sids: Dict[str, List[str]] = defaultdict(list)
    did2cumlens: Dict[str, List[int]] = {}
    sid2knp: Dict[str, str] = {}

    # print(f'reading from {input_dir}...')
    for knp_file in input_dir.glob('*.knp'):
        with knp_file.open() as fin:
            did = knp_file.stem
            did2cumlens[did] = [0]
            buff = ''
            for line in fin:
                buff += line
                if line.strip() == 'EOS':
                    blist = BList(buff)
                    did2sids[did].append(blist.sid)
                    did2cumlens[did].append(
                        did2cumlens[did][-1] + len(tokenizer.tokenize(' '.join(m.midasi for m in blist.mrph_list())))
                    )
                    sid2knp[blist.sid] = buff
                    buff = ''

    # print(f'writing to {output_dir}...')
    for did, sids in did2sids.items():
        cum: List[int] = did2cumlens[did]
        end = 1
        # end を探索
        while end < len(sids) and cum[end+1] - cum[0] <= MAX_SUBWORD_LEN:
            end += 1

        idx = 0
        while end < len(sids) + 1:
            start = 0
            # start を探索
            while cum[end] - cum[start] > MAX_SUBWORD_LEN:
                start += 1
                if start == end - 1:
                    break
            with output_dir.joinpath(f'{did}-{idx:02}.knp').open('wt') as fout:
                fout.write(''.join(sid2knp[sid] for sid in sids[start:end]))  # start から end まで書き出し
            idx += 1
            end += 1


def process_kc(input_path: Path, output_path: Path, cases: List[str], corefs: List[str], tokenizer: BertTokenizer):
    tmp_dir = Path('/tmp/kc')
    tmp_dir.mkdir(exist_ok=True)
    # 京大コーパスは1文書が長いのでできるだけ多くの context を含むように複数文書に分割する
    split_kc(input_path, tmp_dir, tokenizer)

    output_path.mkdir(exist_ok=True)
    reader = KyotoReader(tmp_dir, cases, corefs, extract_nes=False)
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
    parser.add_argument('--bert-model', type=str, default=None,
                        help='path to pre-trained BERT model directory')
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
        print('processing kwdlc...')
        process_kwdlc(kwdlc_dir / 'train', output_dir / 'train', target_cases, target_corefs)
        process_kwdlc(kwdlc_dir / 'valid', output_dir / 'valid', target_cases, target_corefs)
        process_kwdlc(kwdlc_dir / 'test', output_dir / 'test', target_cases, target_corefs)

    if args.kc is not None:
        kc_dir = Path(args.kc).resolve()
        output_dir = Path(args.out) / 'kc'
        output_dir.mkdir(exist_ok=True)
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=False, tokenize_chinese_chars=False)
        print('processing kc...')
        process_kc(kc_dir / 'train', output_dir / 'train', target_cases, target_corefs, tokenizer)
        process_kc(kc_dir / 'valid', output_dir / 'valid', target_cases, target_corefs, tokenizer)
        process_kc(kc_dir / 'test', output_dir / 'test', target_cases, target_corefs, tokenizer)

    if args.commonsense is not None:
        commonsense_dir = Path(args.commonsense).resolve()
        output_dir = Path(args.out) / 'commonsense'
        output_dir.mkdir(exist_ok=True)
        print('processing commonsense...')
        process_commonsense(commonsense_dir / 'train.csv', output_dir / 'train.pkl')
        process_commonsense(commonsense_dir / 'valid.csv', output_dir / 'valid.pkl')
        process_commonsense(commonsense_dir / 'test.csv', output_dir / 'test.pkl')

    with Path(args.out).joinpath('config.json').open(mode='wt') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
