import argparse
from typing import List, Dict
from pathlib import Path

from pyknp import Tag

from kwdlc_reader.reader import KWDLCReader
from kwdlc_reader import ALL_CASES, ALL_COREFS, ALL_EXOPHORS

"""
TODO
- 複数のarg正解候補
- ng_arg_ids
"""


def convert(kwdlc_dir: Path, output_dir: Path):
    # reader = KWDLCReader(kwdlc_dir,
    #                      target_cases=ALL_CASES,
    #                      target_corefs=ALL_COREFS,
    #                      target_exophors=ALL_EXOPHORS)
    reader = KWDLCReader(kwdlc_dir,
                         target_cases=['ガ', 'ヲ', 'ニ', 'ガ２'],
                         target_corefs=["=", "=構", "=≒"],
                         target_exophors=['読者', '著者', '不特定:人'])
    for document in reader.process_all_documents():
        with output_dir.joinpath(f'{document.doc_id}.conll').open(mode='w') as writer:
            writer.write(f'# A-ID:{document.doc_id}\n')
            dmid = 0
            for sentence in document:
                items = ['_'] * 8
                items[4] = sentence.sid
                dmid2pred: Dict[int, Tag] = {pas.dmid: pas.predicate for pas in document.pas_list()}
                for tag in sentence.tag_list():
                    items[2] = str(tag.parent_id) + tag.dpndtype
                    for mrph in tag.mrph_list():
                        items[0] = str(dmid + 1)
                        items[1] = mrph.midasi
                        if ('<用言:' in tag.fstring) and ('<省略解析なし>' not in tag.fstring):
                            if '<内容語>' in mrph.fstring:
                                arguments: List[str] = []
                                cases = reader.target_cases
                                if dmid in dmid2pred:
                                    case2args = document.get_arguments(dmid2pred[dmid], relax=True)
                                    for case in cases:
                                        if case not in case2args:
                                            arguments.append('NULL')
                                            continue
                                        arg = case2args[case][0]  # use first argument now
                                        if arg.dep_type == 'exo':
                                            arguments.append(arg.midasi)
                                        elif arg.dep_type == 'overt':
                                            arguments.append(f'{arg.dmid + 1}%C')
                                        else:
                                            arguments.append(str(arg.dmid + 1))
                                else:
                                    arguments = ['NULL'] * len(cases)
                                items[5] = ','.join(f'{case}:{arg}' for case, arg in zip(cases, arguments))

                        writer.write('\t'.join(items) + '\n')
                        items = ['_'] * 8
                        dmid += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('TRAIN', help='path to directory where train data exists')
    parser.add_argument('VALID', help='path to directory where validation data exists')
    parser.add_argument('TEST', help='path to directory where test data exists')
    parser.add_argument('OUT', help='path to directory where dataset to be located')

    args = parser.parse_args()
    train_out_dir = Path(args.OUT) / 'train'
    valid_out_dir = Path(args.OUT) / 'valid'
    test_out_dir = Path(args.OUT) / 'test'

    # make directories to save dataset
    train_out_dir.mkdir(parents=True, exist_ok=True)
    valid_out_dir.mkdir(parents=True, exist_ok=True)
    test_out_dir.mkdir(parents=True, exist_ok=True)

    convert(Path(args.TRAIN), train_out_dir)
    convert(Path(args.VALID), valid_out_dir)
    convert(Path(args.TEST), test_out_dir)


if __name__ == '__main__':
    main()
