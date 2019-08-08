import argparse
from pathlib import Path
from typing import List

from pyknp import BList


NUM_COLUMNS = 11


def convert(input_file: Path, output_file: Path):
    target_cases = ['ガ', 'ヲ', 'ニ', 'ガ２']
    sentences = []
    with input_file.open() as reader:
        buff = ''
        for line in reader:
            buff += line
            if line == 'EOS\n':
                sentence = BList(buff)
                sentences.append(sentence)
                buff = ''

    with output_file.open(mode='w') as writer:
        dmid = 0
        non_head_dmids = []
        num_mrphs = sum(len(sentence.mrph_list()) for sentence in sentences)
        for sentence in sentences:
            items = ['_'] * NUM_COLUMNS
            items[4] = sentence.sid
            for bnst in sentence.bnst_list():
                items[3] = str(bnst.parent_id) + bnst.dpndtype
                # dmid2pred: Dict[int, Tag] = {pas.dmid: pas.predicate for pas in document.pas_list()}
                for tag in bnst.tag_list():
                    items[2] = str(tag.parent_id) + tag.dpndtype
                    items[10] = tag.fstring
                    for idx, mrph in enumerate(tag.mrph_list()):
                        items[0] = str(dmid + 1)
                        items[1] = mrph.midasi
                        items[7] = mrph.spec().strip()
                        if '<内容語>' not in mrph.fstring and idx > 0:
                            non_head_dmids.append(dmid)
                        if '<用言:' in tag.fstring \
                                and '<省略解析なし>' not in tag.fstring \
                                and '<内容語>' in mrph.fstring:
                            arguments: List[str] = []
                            for case in target_cases:
                                for child in tag.children:
                                    if f'<{case}>' in child.fstring:
                                        arguments.append(f'{child.dmid + 1}%C')
                                        break
                                else:
                                    arguments.append('NULL')
                            items[5] = ','.join(f'{case}:{arg}' for case, arg in zip(target_cases, arguments))
                            items[6] = 'NA'
                            ng_arg_ids = non_head_dmids + list(range(dmid, num_mrphs))
                            items[8] = '/'.join(str(id_) for id_ in ng_arg_ids)

                        writer.write('\t'.join(items) + '\n')
                        items = ['_'] * NUM_COLUMNS
                        dmid += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='path to input knp file')
    parser.add_argument('--output', help='path to output conll file')
    args = parser.parse_args()

    # make directories to save dataset
    output_file = Path(args.output)
    output_file.mkdir(parents=True, exist_ok=True)

    convert(Path(args.input), output_file)


if __name__ == '__main__':
    main()
