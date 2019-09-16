import io
import logging
import argparse
import textwrap
from pathlib import Path
from typing import List, Dict, Union
from collections import OrderedDict

from pyknp import BList, Tag

from kwdlc_reader import KWDLCDirectoryReader, Document, Argument
from utils.util import OrderedDefaultDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Scorer:
    def __init__(self, documents_pred: List[Document], reader_gold: KWDLCDirectoryReader):

        assert sorted(doc.doc_id for doc in documents_pred) == sorted(reader_gold.did2path.keys())
        self.cases = reader_gold.target_cases
        self.doc_ids: List[str] = list(reader_gold.did2path.keys())
        self.did2document_pred: Dict[str, Document] = {doc.doc_id: doc for doc in documents_pred}
        self.did2document_gold: Dict[str, Document] = {doc.doc_id: doc for doc in reader_gold.process_all_documents()}
        self.sid2predicates_pred: Dict[str, List[Tag]] = OrderedDefaultDict(list)
        self.sid2predicates_gold: Dict[str, List[Tag]] = OrderedDefaultDict(list)
        for pas in [pas for doc in self.did2document_pred.values() for pas in doc.pas_list()]:
            self.sid2predicates_pred[pas.sid].append(pas.predicate)
        for pas in [pas for doc in self.did2document_gold.values() for pas in doc.pas_list()]:
            tag = pas.predicate
            if '<用言:' in tag.fstring \
                    and '<省略解析なし>' not in tag.fstring \
                    and any('<内容語>' in mrph.fstring for mrph in tag.mrph_list()):
                self.sid2predicates_gold[pas.sid].append(tag)

        self.measures: Dict[str, Dict[str, Measure]] = \
            OrderedDict((case, OrderedDefaultDict(lambda: Measure())) for case in self.cases)
        self.comp_result = {}
        self.deptype2analysis = {'dep': 'case_analysis',
                                 'intra': 'zero_intra_sentential',
                                 'inter': 'zero_inter_sentential',
                                 'exo': 'zero_exophora'}

        for doc_id in self.doc_ids:
            document_pred = self.did2document_pred[doc_id]
            document_gold = self.did2document_gold[doc_id]
            dtid2pred_pred: Dict[int, Tag] = {document_pred.tag2dtid[tag]: tag
                                              for sid in document_pred.sid2sentence.keys()
                                              for tag in self.sid2predicates_pred[sid]}
            dtid2pred_gold: Dict[int, Tag] = {document_gold.tag2dtid[tag]: tag
                                              for sid in document_gold.sid2sentence.keys()
                                              for tag in self.sid2predicates_gold[sid]}

            # calculate precision
            for dtid, predicate_pred in dtid2pred_pred.items():
                if dtid in dtid2pred_gold:
                    predicate_gold = dtid2pred_gold[dtid]
                    arguments_pred = document_pred.get_arguments(predicate_pred, relax=False)
                    arguments_gold = document_gold.get_arguments(predicate_gold, relax=True)
                    for case in self.cases:
                        argument_pred: List[Argument] = arguments_pred[case]
                        argument_gold: List[Argument] = arguments_gold[case]
                        if not argument_pred:
                            continue
                        assert len(argument_pred) == 1
                        arg = argument_pred[0]
                        if arg.dep_type == 'overt':  # ignore overt case
                            self.comp_result[(doc_id, dtid, case)] = 'overt'
                            continue
                        analysis = self.deptype2analysis[arg.dep_type]
                        if arg in argument_gold:
                            self.measures[case][analysis].correct += 1
                            self.comp_result[(doc_id, dtid, case)] = analysis
                        else:
                            self.comp_result[(doc_id, dtid, case)] = 'wrong'
                        self.measures[case][analysis].denom_pred += 1
                        # if arg in argument_gold:
                        #     if arg.dep_type == 'dep':
                        #         self.comp_result[(doc_id, dtid, case)] = 'case_analysis'
                        #     elif arg.dep_type == 'intra':
                        #         self.comp_result[(doc_id, dtid, case)] = 'zero_intra'
                        #     elif arg.dep_type == 'inter':
                        #         self.comp_result[(doc_id, dtid, case)] = 'zero_inter'
                        #     elif arg.dep_type == 'exo':
                        #         self.comp_result[(doc_id, dtid, case)] = 'exophora'
                        #     else:
                        #         logger.warning(f'unknown dep_type: {arg.dep_type}')
                        #         continue
                        #     self.measures[case].correct += 1
                        # else:
                        #     self.comp_result[(doc_id, dtid, case)] = 'wrong'
                        # self.measures[case].denom_pred += 1

            # calculate recall
            for predicate_gold in dtid2pred_gold.values():
                arguments_gold = document_gold.get_arguments(predicate_gold, relax=True)
                predicate_gold_dtid: int = document_gold.tag2dtid[predicate_gold]
                for case in self.cases:
                    argument_gold: List[Argument] = list(filter(
                        lambda argument: argument.dtid is None or argument.dtid < predicate_gold_dtid,
                        arguments_gold[case]))
                    if argument_gold:
                        arg = argument_gold[0]  # dataset.pyとの一貫性を保つため[0]を用いる
                        if arg.dep_type == 'overt':  # ignore overt case
                            continue
                        analysis = self.deptype2analysis[arg.dep_type]
                        self.measures[case][analysis].denom_gold += 1

    def result_dict(self) -> Dict[str, Dict[str, 'Measure']]:
        result = OrderedDict()
        all_case_result = OrderedDefaultDict(lambda: Measure())
        for case, measures in self.measures.items():
            case_result = {anal: Measure() for anal in self.deptype2analysis.values()}
            for analysis, measure in measures.items():
                case_result[analysis] = measure
                all_case_result[analysis] += measure
            case_result['zero_all'] = case_result['zero_intra_sentential'] + \
                                      case_result['zero_inter_sentential'] + \
                                      case_result['zero_exophora']
            case_result['all'] = case_result['case_analysis'] + case_result['zero_all']
            all_case_result['zero_all'] += case_result['zero_all']
            all_case_result['all'] += case_result['all']
            result[case] = case_result
        result['all_case'] = all_case_result
        # measure = reduce(operator.add, self.measures.values(), Measure())
        return result

    def export_txt(self, destination: Union[str, Path, io.TextIOBase]):
        lines = []
        for case, measures in self.result_dict().items():
            if case in self.cases:
                lines.append(f'{case}格')
            else:
                lines.append(f'{case}')
            for analysis, measure in measures.items():
                lines.append(f'  {analysis}')
                lines.append(f'    precision: {measure.precision:.3} ({measure.denom_pred})')
                lines.append(f'    recall   : {measure.recall:.3} ({measure.denom_gold})')
                lines.append(f'    F        : {measure.f1:.3}')
        text = '\n'.join(lines) + '\n'

        if isinstance(destination, str) or isinstance(destination, Path):
            with Path(destination).open('wt') as writer:
                writer.write(text)
        elif isinstance(destination, io.TextIOBase):
            destination.write(text)

    def export_csv(self, destination: Union[str, Path, io.TextIOBase], sep: str = ', '):
        text = ''
        result_dict = self.result_dict()
        text += 'case' + sep
        text += sep.join(result_dict['all_case'].keys()) + '\n'
        for case, measures in result_dict.items():
            text += f'{case}' + sep
            text += sep.join(f'{measure.f1:.3}' for measure in measures.values())
            text += '\n'

        if isinstance(destination, str) or isinstance(destination, Path):
            with Path(destination).open('wt') as writer:
                writer.write(text)
        elif isinstance(destination, io.TextIOBase):
            destination.write(text)

    def write_html(self, output_file: Union[str, Path]):
        if isinstance(output_file, str):
            output_file = Path(output_file)
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
                    self._draw_tree(sid, self.sid2predicates_gold[sid], document_gold, fh=writer)
                    writer.write('\n')
                writer.write('</pre>')

                writer.write('<td><pre>\n')
                for sid in document_pred.sid2sentence.keys():
                    self._draw_tree(sid, self.sid2predicates_pred[sid], document_pred, fh=writer)
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
        pre {font-family: "ＭＳ ゴシック","Osaka-Mono","Osaka-等幅","さざなみゴシック","Sazanami Gothic",DotumChe,GulimChe,
        BatangChe,MingLiU, NSimSun, Terminal; white-space:pre;}
        -->

        </style>

        <meta HTTP-EQUIV="content-type" CONTENT="text/html" charset="utf-8">
        <link rel="stylesheet" href="result.css" type="text/css"/>
        </head>
        ''')

    def _draw_tree(self,
                   sid: str,
                   predicates: List[Tag],
                   document: Document,
                   fh=None,
                   html: bool = True
                   ) -> None:
        sentence: BList = document.sid2sentence[sid]
        with io.StringIO() as string:
            sentence.draw_tag_tree(fh=string)
            tree_strings = string.getvalue().rstrip('\n').split('\n')
        tag_list = sentence.tag_list()
        assert len(tree_strings) == len(tag_list)
        for i, (line, tag) in enumerate(zip(tree_strings, tag_list)):
            if tag in predicates:
                arguments = document.get_arguments(tag)
                tree_strings[i] += '  '
                for case in self.cases:
                    argument = arguments[case]
                    if argument:
                        arg = argument[0].midasi
                        if self.comp_result.get((document.doc_id, document.tag2dtid[tag], case), None) == 'overt':
                            color = 'green'
                        elif self.comp_result.get((document.doc_id, document.tag2dtid[tag], case), None) \
                                in self.deptype2analysis.values():
                            color = 'blue'
                        else:
                            color = 'red'
                    else:
                        arg = 'NULL'
                        color = 'gray'
                    if html:
                        tree_strings[i] += f'<font color="{color}">{arg}:{case}</font> '
                    else:
                        tree_strings[i] += f'{arg}:{case} '

        print('\n'.join(tree_strings), file=fh)


class Measure:
    def __init__(self,
                 denom_pred: int = 0,
                 denom_gold: int = 0,
                 correct: int = 0):
        self.denom_pred = denom_pred
        self.denom_gold = denom_gold
        self.correct = correct

    def __add__(self, other: 'Measure'):
        return Measure(self.denom_pred + other.denom_pred,
                       self.denom_gold + other.denom_gold,
                       self.correct + other.correct)

    @property
    def precision(self) -> float:
        if self.denom_pred == 0:
            logger.warning('zero division at precision')
            return 0.0
        return self.correct / self.denom_pred

    @property
    def recall(self) -> float:
        if self.denom_gold == 0:
            logger.warning('zero division at recall')
            return 0.0
        return self.correct / self.denom_gold

    @property
    def f1(self) -> float:
        if self.denom_pred + self.denom_gold == 0:
            logger.warning('zero division at f1')
            return 0.0
        return 2 * self.correct / (self.denom_pred + self.denom_gold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction-dir', default=None, type=str,
                        help='path to directory where system output KWDLC files exist (default: None)')
    parser.add_argument('--gold-dir', default=None, type=str,
                        help='path to directory where gold KWDLC files exist (default: None)')
    parser.add_argument('--result-html', default=None, type=str,
                        help='path to html file which prediction result is exported (default: None)')
    parser.add_argument('--result-csv', default=None, type=str,
                        help='path to csv file which prediction result is exported (default: None)')
    parser.add_argument('--case-string', type=str, default='ガ,ヲ,ニ,ガ２',
                        help='Case strings. Separate by ","')
    parser.add_argument('--exophors', type=str, default='著者,読者,不特定:人',
                        help='Special tokens. Separate by ",".')
    args = parser.parse_args()

    reader_gold = KWDLCDirectoryReader(
        Path(args.gold_dir),
        target_cases=args.case_string.split(','),
        target_corefs=['=', '=構', '=≒'],
        target_exophors=args.exophors.split(','),
        extract_nes=False
    )
    reader_pred = KWDLCDirectoryReader(
        Path(args.prediction_dir),
        target_cases=reader_gold.target_cases,
        target_corefs=reader_gold.target_corefs,
        target_exophors=reader_gold.target_exophors,
        extract_nes=False
    )
    documents_pred = list(reader_pred.process_all_documents())

    scorer = Scorer(documents_pred, reader_gold)
    if args.result_html:
        scorer.write_html(Path(args.result_html))
    if args.result_csv:
        scorer.export_result_csv(args.result_csv)
    scorer.print_result()


if __name__ == '__main__':
    main()
