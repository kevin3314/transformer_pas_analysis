import io
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Optional, TextIO

from kyoto_reader import Document, Argument
from pyknp import BList

from writer.prediction_writer import PredictionKNPWriter
from analyzer import Analyzer


def draw_tree(document: Document,
              sid: str,
              cases: List[str],
              coreference: bool = False,
              fh: Optional[TextIO] = None,
              ) -> None:
    """sid で指定された文の述語項構造・共参照関係をツリー形式で fh に書き出す

    Args:
        document (Document): sid が含まれる文書
        sid (str): 出力対象の文ID
        cases (List[str]): 表示対象の格
        coreference (bool): 共参照関係も表示するかどうか
        fh (Optional[TextIO]): 出力ストリーム
    """
    sentence: BList = document.sid2sentence[sid]
    with io.StringIO() as string:
        sentence.draw_tag_tree(fh=string)
        tree_strings = string.getvalue().rstrip('\n').split('\n')
    assert len(tree_strings) == len(sentence.tag_list())
    all_midasis = [m.midasi for m in document.mentions.values()]
    for predicate in filter(lambda p: p.sid == sid, document.get_predicates()):
        idx = predicate.tid
        tree_strings[idx] += '  '
        arguments = document.get_arguments(predicate)
        for case in cases:
            args = arguments[case]
            if case == 'ガ':
                args += arguments['判ガ']
            if case == 'ノ':
                args += arguments['ノ？']
            targets = set()
            for arg in args:
                target = arg.midasi
                if all_midasis.count(arg.midasi) > 1 and isinstance(arg, Argument):
                    target += str(arg.dtid)
                targets.add(target)
            tree_strings[idx] += f'{",".join(targets)}:{case} '
    if coreference:
        for src_mention in filter(lambda m: m.sid == sid, document.mentions.values()):
            tgt_mentions = [tgt for tgt in document.get_siblings(src_mention) if tgt.dtid < src_mention.dtid]
            targets = set()
            for tgt_mention in tgt_mentions:
                target = tgt_mention.midasi
                if all_midasis.count(tgt_mention.midasi) > 1:
                    target += str(tgt_mention.dtid)
                targets.add(target)
            for eid in src_mention.eids:
                entity = document.entities[eid]
                if entity.is_special:
                    targets.add(entity.exophor)
            if not targets:
                continue
            idx = src_mention.tid
            tree_strings[idx] += '  ＝:'
            tree_strings[idx] += ','.join(targets)

    print('\n'.join(tree_strings), file=fh)


def main(args):
    logger = logging.getLogger(__name__)
    analyzer = Analyzer(args.model, device=args.device, logger=logger, bertknp=args.use_bertknp)

    if args.input is not None:
        source = args.input
    elif args.knp_dir is not None:
        source = Path(args.knp_dir)
    else:
        source = ''.join(sys.stdin.readlines())

    arguments_set, dataset = analyzer.analyze(source)

    prediction_writer = PredictionKNPWriter(dataset, logger)
    destination = Path(args.export_dir) if args.export_dir is not None else sys.stdout
    if args.tab is True:
        prediction_writer.write(arguments_set, destination, skip_untagged=False)
    else:
        documents_pred: List[Document] = prediction_writer.write(arguments_set, destination, skip_untagged=False)
        for document_pred in documents_pred:
            print()
            for sid in document_pred.sid2sentence.keys():
                draw_tree(document_pred, sid, dataset.target_cases, dataset.coreference, sys.stdout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', '-r', '--resume', required=True, type=str,
                        help='path to trained checkpoint')
    parser.add_argument('-d', '--device', default='', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('--input', default=None, type=str,
                        help='sentences to analysis (if not specified, use stdin)')
    parser.add_argument('--knp-dir', default=None, type=str,
                        help='path to the directory where parsed documents are saved'
                             'in case parsed files exist here, KNP is skipped')
    parser.add_argument('--export-dir', default=None, type=str,
                        help='directory where analysis result is exported')
    parser.add_argument('-tab', action='store_true', default=False,
                        help='whether to output details')
    parser.add_argument('--use-bertknp', action='store_true', default=False,
                        help='use BERTKNP in base phrase segmentation and parsing')
    # parser.add_argument('-c', '--config', default=None, type=str,
    #                     help='config file path (default: None)')
    main(parser.parse_args())
