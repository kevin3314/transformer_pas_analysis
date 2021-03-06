"""Preprocess dataset."""
import argparse
from functools import partial
import json
import tempfile
from pathlib import Path
import _pickle as cPickle
from typing import List, Dict
from collections import defaultdict, namedtuple
from multiprocessing import Pool

from tqdm import tqdm
from pyknp import BList
from kyoto_reader import KyotoReader
from tokenizer import (
    BartSPMTokenizeHandler,
    BertTokenizeHandler,
    BiBartTokenizeHandler,
    T5TokenizeHandler,
    MBartTokenizerHandler,
    TokenizeHandlerMeta
)

from data_loader.dataset.commonsense_dataset import CommonsenseExample


ARC2TOKENIZER_INFO = {
    'bert': BertTokenizeHandler,
    # TODO: Provide real num of special tokens
    't5': T5TokenizeHandler,
    'bart': BartSPMTokenizeHandler,
    'bibart': BiBartTokenizeHandler,
    'mbart': MBartTokenizerHandler
}

DocumentDivideUnit = namedtuple("DocumentDivideUnit", ["did", "idx", "start", "end"])


def process(input_path: Path, output_path: Path, corpus: str) -> int:
    output_path.mkdir(exist_ok=True)
    reader = KyotoReader(input_path, extract_nes=False)
    for document in tqdm(reader.process_all_documents(backend="multiprocessing"), desc=corpus, total=len(reader)):
        with output_path.joinpath(document.doc_id + '.pkl').open(mode='wb') as f:
            cPickle.dump(document, f)
    return len(reader)


def write_partial_document(
    document_divide_unit: DocumentDivideUnit,
    did2sids: Dict[str, List[str]],
    sid2knp: Dict[str, str],
    output_dir: Path
):
    """Write partial document

    Args:
        document_divide_unit (DocumentDivideUnit): Information for partial document to write
        did2sid (Dict[str, List[str]])
        sid2knp (Dict[str, str])
        output_dir (Path):
    """
    did, idx, start, end = document_divide_unit
    sids = did2sids[did]
    with output_dir.joinpath(f'{did}-{idx:02}.knp').open('wt') as fout:
        fout.write(''.join(sid2knp[sid] for sid in sids[start:end]))  # start から end まで書き出し


def split_kc(input_dir: Path, output_dir: Path, max_subword_length: int, tokenizer: TokenizeHandlerMeta):
    """
    各文書を，tokenize したあとの長さが max_subword_length 以下になるように複数の文書に分割する．
    1文に分割しても max_subword_length を超えるような長い文はそのまま出力する
    """
    did2sids: Dict[str, List[str]] = defaultdict(list)
    did2cumlens: Dict[str, List[int]] = {}
    sid2knp: Dict[str, str] = {}

    max_all_tokens_len = 0

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
                    all_tokens, *_ = tokenizer.get_tokenized_tokens(list(m.midasi for m in blist.mrph_list()))
                    max_all_tokens_len = max(max_all_tokens_len, len(all_tokens))
                    did2cumlens[did].append(
                        did2cumlens[did][-1] + len(all_tokens)
                        # did2cumlens[did][-1] + len(tokenizer.tokenize(' '.join(m.midasi for m in blist.mrph_list())))
                    )
                    sid2knp[blist.sid] = buff
                    buff = ''

    print(f"max_tokens_length per sentence -> {max_all_tokens_len}")
    # assert max_all_tokens_len <= max_subword_length
    # if max_all_tokens_len > max_subword_length:
    #     raise ValueError(f"max_tokens_length exceeded max_subword_length\n{max_all_tokens_len}>{max_subword_length}")
    document_divide_unit_list = []
    for did, sids in did2sids.items():
        cum: List[int] = did2cumlens[did]
        end = 1
        # end を探索
        while end < len(sids) and cum[end+1] - cum[0] <= max_subword_length:
            end += 1

        idx = 0
        while end < len(sids) + 1:
            start = 0
            # start を探索
            while cum[end] - cum[start] > max_subword_length:
                start += 1
                if start == end - 1:
                    break
            document_divide_unit_list.append(
                DocumentDivideUnit(did, idx, start, end)
            )
            # with output_dir.joinpath(f'{did}-{idx:02}.knp').open('wt') as fout:
            #     fout.write(''.join(sid2knp[sid] for sid in sids[start:end]))  # start から end まで書き出し
            idx += 1
            end += 1

    _write_partial_document = partial(
        write_partial_document,
        did2sids=did2sids,
        sid2knp=sid2knp,
        output_dir=output_dir
    )
    with Pool() as pool:
        list(pool.imap(_write_partial_document, document_divide_unit_list))


def process_kc(input_path: Path,
               output_path: Path,
               max_subword_length: int,
               tokenizer: TokenizeHandlerMeta,
               split: bool = False
               ) -> int:
    with tempfile.TemporaryDirectory() as tmp_dir:
        if split:
            tmp_dir = Path(tmp_dir)
            # 京大コーパスは1文書が長いのでできるだけ多くの context を含むように複数文書に分割する
            print('splitting kc...')
            split_kc(input_path, tmp_dir, max_subword_length, tokenizer)
            input_path = tmp_dir

        print(list(input_path.iterdir()))
        output_path.mkdir(exist_ok=True)
        reader = KyotoReader(input_path, extract_nes=False, did_from_sid=False)
        for document in tqdm(reader.process_all_documents(backend="multiprocessing"), desc='kc', total=len(reader)):
            with output_path.joinpath(document.doc_id + '.pkl').open(mode='wb') as f:
                cPickle.dump(document, f)

    return len(reader)


def process_commonsense(input_path: Path, output_path: Path) -> int:
    examples = []
    print('processing commonsense...')
    with input_path.open() as f:
        for line in f:
            label, string = line.strip().split(',')
            former_string, latter_string = string.split('@')
            examples.append(CommonsenseExample(former_string, latter_string, bool(int(label))))
    with output_path.open(mode='wb') as f:
        cPickle.dump(examples, f)
    return len(examples)


def main():
    all_archs = list(set(arch for arch in ARC2TOKENIZER_INFO.keys()))

    parser = argparse.ArgumentParser()
    parser.add_argument('--kwdlc', type=str, default=None,
                        help='path to directory where KWDLC data exists')
    parser.add_argument('--kc', type=str, default=None,
                        help='path to directory where Kyoto Corpus data exists')
    parser.add_argument('--fuman', type=str, default=None,
                        help='path to directory where Fuman Corpus data exists')
    parser.add_argument('--commonsense', type=str, default=None,
                        help='path to directory where commonsense inference data exists')
    parser.add_argument('--out', type=(lambda p: Path(p)), required=True,
                        help='path to directory where dataset to be located')
    parser.add_argument('--max-seq-length', type=int, default=128,
                        help='The maximum total input sequence length after WordPiece tokenization. Sequences '
                             'longer than this will be truncated, and sequences shorter than this will be padded.')
    parser.add_argument('--exophors', '--exo', type=str, default='著者,読者,不特定:人,不特定:物',
                        help='exophor strings separated by ","')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='Pretrained path. Pass full path to pretrained files or choice from PRETRAINED_PATH')
    parser.add_argument('--arch', choices=all_archs, type=str, default='bert', help='transformer archtecture name')
    args = parser.parse_args()

    # make directories to save dataset
    args.out.mkdir(exist_ok=True)
    exophors = args.exophors.split(',')
    tokenizer_class = ARC2TOKENIZER_INFO.get(args.arch)
    tokenizer = tokenizer_class.from_pretrained(args.pretrained, do_lower_case=False, tokenize_chinese_chars=False)

    config_path: Path = args.out / 'config.json'
    if config_path.exists():
        with config_path.open() as f:
            config = json.load(f)
    else:
        config = {}
    config.update(
        {
            'max_seq_length': args.max_seq_length,
            'exophors': exophors,
            'vocab_size': tokenizer.vocab_size,
            'model_name': args.arch,
            'pretrained_path': args.pretrained,
        }
    )
    if 'num_examples' not in config:
        config['num_examples'] = {}

    if args.kwdlc is not None:
        in_dir = Path(args.kwdlc).resolve()
        out_dir: Path = args.out / 'kwdlc'
        out_dir.mkdir(exist_ok=True)
        num_examples_train = process(in_dir / 'train', out_dir / 'train', 'kwdlc')
        num_examples_valid = process(in_dir / 'valid', out_dir / 'valid', 'kwdlc')
        num_examples_test = process(in_dir / 'test', out_dir / 'test', 'kwdlc')
        num_examples_dict = {'train': num_examples_train, 'valid': num_examples_valid, 'test': num_examples_test}
        config['num_examples']['kwdlc'] = num_examples_dict

    if args.kc is not None:
        in_dir = Path(args.kc).resolve()
        out_dir: Path = args.out / 'kc_split'
        out_dir.mkdir(exist_ok=True)
        # Memo: special tokens ([SEP], </s>, en_XX, etc..) are handled by tokenizer in function
        max_subword_length = args.max_seq_length - len(exophors) - 2  # [NULL], [NA]
        print("max_subword_length ->", max_subword_length)
        num_examples_train = process_kc(in_dir / 'train', out_dir / 'train', max_subword_length, tokenizer, split=True)
        num_examples_valid = process_kc(in_dir / 'valid', out_dir / 'valid', max_subword_length, tokenizer, split=True)
        num_examples_test = process_kc(in_dir / 'test', out_dir / 'test', max_subword_length, tokenizer, split=True)
        num_examples_dict = {'train': num_examples_train, 'valid': num_examples_valid, 'test': num_examples_test}
        config['num_examples']['kc'] = num_examples_dict

        out_dir: Path = args.out / 'kc'
        out_dir.mkdir(exist_ok=True)
        _ = process_kc(in_dir / 'valid', out_dir / 'valid', args.max_seq_length, tokenizer, split=False)
        _ = process_kc(in_dir / 'test', out_dir / 'test', args.max_seq_length, tokenizer, split=False)

    if args.fuman is not None:
        in_dir = Path(args.fuman).resolve()
        out_dir: Path = args.out / 'fuman'
        out_dir.mkdir(exist_ok=True)
        num_examples_train = process(in_dir / 'train', out_dir / 'train', 'fuman')
        num_examples_valid = process(in_dir / 'valid', out_dir / 'valid', 'fuman')
        num_examples_test = process(in_dir / 'test', out_dir / 'test', 'fuman')
        num_examples_dict = {'train': num_examples_train, 'valid': num_examples_valid, 'test': num_examples_test}
        config['num_examples']['fuman'] = num_examples_dict

    if args.commonsense is not None:
        in_dir = Path(args.commonsense).resolve()
        out_dir: Path = args.out / 'commonsense'
        out_dir.mkdir(exist_ok=True)
        num_examples_train = process_commonsense(in_dir / 'train.csv', out_dir / 'train.pkl')
        num_examples_valid = process_commonsense(in_dir / 'valid.csv', out_dir / 'valid.pkl')
        num_examples_test = process_commonsense(in_dir / 'test.csv', out_dir / 'test.pkl')
        num_examples_dict = {'train': num_examples_train, 'valid': num_examples_valid, 'test': num_examples_test}
        config['num_examples']['commonsense'] = num_examples_dict

    with config_path.open(mode='wt') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
