import io
import logging
import argparse
import textwrap
from pathlib import Path
from typing import List, Dict, Optional
from functools import reduce
import operator

from pyknp import BList, Tag

from kwdlc_reader import KWDLCReader, Document, Argument


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Scorer:
    def __init__(self, prediction_output_dir: Path, reader_gold: KWDLCReader):
        reader_pred = KWDLCReader(
            prediction_output_dir,
            target_cases=reader_gold.target_cases,
            target_corefs=reader_gold.target_corefs,
            target_exophors=reader_gold.target_exophors,
            extract_nes=False
        )
        assert sorted(reader_pred.did2path.keys()) == sorted(reader_gold.did2path.keys())
        self.cases = reader_gold.target_cases
        self.doc_ids: List[str] = list(reader_gold.did2path.keys())
        self.did2document_pred: Dict[str, Document] = {doc.doc_id: doc for doc in reader_pred.process_all_documents()}
        self.did2document_gold: Dict[str, Document] = {doc.doc_id: doc for doc in reader_gold.process_all_documents()}
        self.measures: Dict[str, Measure] = {case: Measure() for case in self.cases}

        for doc_id in self.doc_ids:
            document_pred = self.did2document_pred[doc_id]
            document_gold = self.did2document_gold[doc_id]
            predicates_pred = document_pred.get_predicates()
            predicates_gold = document_gold.get_predicates()
            dtid2pred_pred = {document_pred.tag2dtid[tag]: tag for tag in predicates_pred}
            dtid2pred_gold = {document_gold.tag2dtid[tag]: tag for tag in predicates_gold}

            # calculate precision
            for dtid, predicate_pred in dtid2pred_pred.items():
                if dtid in dtid2pred_gold:
                    predicate_gold = dtid2pred_gold[dtid]
                    arguments_pred = document_pred.get_arguments(predicate_pred)
                    arguments_gold = document_gold.get_arguments(predicate_gold)
                    for case in self.cases:
                        argument_pred: List[Argument] = arguments_pred[case]
                        argument_gold: List[Argument] = arguments_gold[case]
                        if not argument_pred:
                            continue
                        assert len(argument_pred) == 1
                        arg = argument_pred[0]
                        # if any(arg in argument_gold for arg in argument_pred):
                        if arg in argument_gold:
                            self.measures[case].tp += 1
                        self.measures[case].denom_pred += 1

            # calculate recall
            for predicate_gold in dtid2pred_gold.values():
                arguments_gold = document_gold.get_arguments(predicate_gold)
                for case in self.cases:
                    argument_gold: List[Argument] = arguments_gold[case]
                    if argument_gold:
                        # print(case + ': ' + argument_gold[0].midasi)
                        self.measures[case].denom_gold += 1

    def __getitem__(self, case: Optional[str]) -> Optional['Measure']:
        if case is not None:
            if case not in self.cases:
                logger.warning(f'unknown case: {case}')
                return None
            return self.measures[case]
        else:
            return reduce(operator.add, self.measures.values(), Measure())

    def result_dict(self) -> dict:
        result = {}
        for case, measure in self.measures.items():
            result[case] = measure
        measure = reduce(operator.add, self.measures.values(), Measure())
        result['all_case'] = measure
        return result

    def print_result(self):
        for case, measure in self.result_dict().items():
            if case in self.cases:
                print(f'--{case}格--')
            else:
                print(f'--{case}--')
            print(f'precision: {measure.precision:.3} ({measure.denom_pred})')
            print(f'recall   : {measure.recall:.3} ({measure.denom_gold})')
            print(f'F        : {measure.f1:.3}')

    def write_html(self, output_file: Path):
        with output_file.open('w') as writer:
            writer.write('<html lang="ja">\n')
            writer.write(self._html_header())
            writer.write('<body>\n')
            writer.write('<h2>Bert Result</h2>\n')
            writer.write('<pre>\n')
            for doc_id in self.doc_ids:
                document_pred = self.did2document_pred[doc_id]
                document_gold = self.did2document_gold[doc_id]
                writer.write('<h3 class="obi1">')
                for sid, sentence in document_gold.sid2sentence.items():
                    writer.write(sid + ' ')
                    writer.write(''.join(bnst.midasi for bnst in sentence.bnst_list()))
                    writer.write('<br>')
                writer.write('</h3>\n')
                writer.write('<table>')
                writer.write('<tr>\n<th>gold</th>\n')
                writer.write('<th>prediction</th>\n</tr>')

                writer.write('<tr>')
                writer.write('<td><pre>\n')
                for sid in document_gold.sid2sentence.keys():
                    self._draw_tree(sid, document_gold, fh=writer)
                    writer.write('\n')
                writer.write('</pre>')

                writer.write('<td><pre>\n')
                for sid in document_pred.sid2sentence.keys():
                    self._draw_tree(sid, document_pred, fh=writer)
                    writer.write('\n')
                writer.write('</pre>\n</tr>\n')

                writer.write('</table>\n')
            writer.write('</pre>\n')
            writer.write('</body>\n')
            writer.write('</html>\n')

    @staticmethod
    def _html_header():
        return textwrap.dedent('''
        <head>
        <title>Bert Result</title>
        <style type="text/css">
        <!--
        td {font-size: 11pt;}
        td {border: 1px solid #606060;}
        td {vertical-align: top;}
        pre {font-family: "ＭＳ ゴシック","Osaka-Mono","Osaka-等幅","さざなみゴシック","Sazanami Gothic",DotumChe,GulimChe,BatangChe,MingLiU, NSimSun, Terminal; white-space:pre;}
        -->

        </style>

        <meta HTTP-EQUIV="content-type" CONTENT="text/html" charset="utf-8">
        <link rel="stylesheet" href="result.css" type="text/css"/>
        </head>
        ''')

    def _draw_tree(self, sid: str, document: Document, fh=None) -> None:
        sentence: BList = document.sid2sentence[sid]
        predicates: List[Tag] = document.get_predicates()
        with io.StringIO() as string:
            sentence.draw_tag_tree(fh=string)
            tree_strings = string.getvalue().strip().split('\n')
        tag_list = sentence.tag_list()
        assert len(tree_strings) == len(tag_list)
        for i, (line, tag) in enumerate(zip(tree_strings, tag_list)):
            if tag in predicates:
                arguments = document.get_arguments(tag)
                tree_strings[i] += '  '
                for case in self.cases:
                    argument = arguments[case]
                    arg = argument[0].midasi if argument else 'NULL'
                    tree_strings[i] += f'{case}:{arg} '

        print('\n'.join(tree_strings), file=fh)


class Measure:
    def __init__(self, denom_pred: int = 0, denom_gold: int = 0, tp: int = 0):
        self.denom_pred = denom_pred
        self.denom_gold = denom_gold
        self.tp = tp

    def __add__(self, other: 'Measure'):
        return Measure(self.denom_pred + other.denom_pred, self.denom_gold + other.denom_gold, self.tp + other.tp)

    @property
    def precision(self) -> float:
        if self.denom_pred == 0:
            logger.warning('zero division at precision')
            return 0
        return self.tp / self.denom_pred

    @property
    def recall(self) -> float:
        if self.denom_gold == 0:
            logger.warning('zero division at recall')
            return 0
        return self.tp / self.denom_gold

    @property
    def f1(self) -> float:
        if self.denom_pred + self.denom_gold == 0:
            logger.warning('zero division at f1')
            return 0
        return 2 * self.tp / (self.denom_pred + self.denom_gold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction-dir', default=None, type=str,
                        help='path to directory where system output KWDLC files exist (default: None)')
    parser.add_argument('--gold-dir', default=None, type=str,
                        help='path to directory where gold KWDLC files exist (default: None)')
    parser.add_argument('--result-html', default=None, type=str,
                        help='path to html file which prediction result is exported (default: None)')
    parser.add_argument('--case-string', type=str, default='ガ,ヲ,ニ,ガ２',
                        help='Case strings. Separate by ","')
    parser.add_argument('--exophors', type=str, default='著者,読者,不特定:人',
                        help='Special tokens. Separate by ",".')
    args = parser.parse_args()

    reader_gold = KWDLCReader(
        Path(args.gold_dir),
        target_cases=args.case_string.split(','),
        target_corefs=['=', '=構', '=≒'],
        target_exophors=args.exophors.split(','),
        extract_nes=False
    )

    scorer = Scorer(Path(args.prediction_dir), reader_gold)
    if args.result_html:
        scorer.write_html(Path(args.result_html))
    scorer.print_result()


if __name__ == '__main__':
    main()
