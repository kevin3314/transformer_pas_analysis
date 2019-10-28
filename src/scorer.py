import io
import sys
import logging
import argparse
import textwrap
from pathlib import Path
from typing import List, Dict, Union
from collections import OrderedDict

from pyknp import BList

from kwdlc_reader import KWDLCReader, Document, Argument, SpecialArgument, BaseArgument, Predicate
from utils.util import OrderedDefaultDict

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Scorer:
    def __init__(self,
                 documents_pred: List[Document],
                 documents_gold: List[Document],
                 target_exophors: List[str],
                 kc: bool = False):
        # long document may have been ignored
        assert set(doc.doc_id for doc in documents_pred) <= set(doc.doc_id for doc in documents_gold)
        self.cases: List[str] = documents_gold[0].target_cases
        self.doc_ids: List[str] = [doc.doc_id for doc in documents_pred]
        self.did2document_pred: Dict[str, Document] = {doc.doc_id: doc for doc in documents_pred}
        self.did2document_gold: Dict[str, Document] = {doc.doc_id: doc for doc in documents_gold}
        self.comp_result: Dict[tuple, str] = {}
        self.deptype2analysis = OrderedDict([('overt', 'overt'),
                                             ('dep', 'case_analysis'),
                                             ('intra', 'zero_intra_sentential'),
                                             ('inter', 'zero_inter_sentential'),
                                             ('exo', 'zero_exophora')])
        self.measures: Dict[str, Dict[str, Measure]] = OrderedDict(
            (case, OrderedDict((anal, Measure()) for anal in self.deptype2analysis.values()))
            for case in self.cases)
        self.relax_exophors: Dict[str, str] = {}
        for exophor in target_exophors:
            self.relax_exophors[exophor] = exophor
            if exophor in ('不特定:人', '不特定:物', '不特定:状況'):
                for n in '１２３４５６７８９':
                    self.relax_exophors[exophor + n] = exophor
        # make sid2predicates_pred and sid2predicates_gold
        self.sid2predicates_pred: Dict[str, List[Predicate]] = OrderedDefaultDict(list)
        self.sid2predicates_gold: Dict[str, List[Predicate]] = OrderedDefaultDict(list)
        for doc_id in self.doc_ids:
            document_pred = self.did2document_pred[doc_id]
            document_gold = self.did2document_gold[doc_id]
            process_all = (kc is False) or (doc_id.split('-')[-1] == '00')
            last_sid = document_pred.sentences[-1].sid if len(document_pred) > 0 else None
            for predicate_pred in document_pred.get_predicates():
                if process_all or (predicate_pred.sid == last_sid):
                    self.sid2predicates_pred[predicate_pred.sid].append(predicate_pred)

            for predicate_gold in document_gold.get_predicates():
                process: bool = process_all or (predicate_gold.sid == last_sid)
                tag = predicate_gold.tag
                if '用言' in tag.features \
                        and process is True:
                    self.sid2predicates_gold[predicate_gold.sid].append(predicate_gold)

        for doc_id in self.doc_ids:
            document_pred = self.did2document_pred[doc_id]
            document_gold = self.did2document_gold[doc_id]

            process_all = (kc is False) or (doc_id.split('-')[-1] == '00')
            last_sid = document_pred.sentences[-1].sid if len(document_pred) > 0 else None
            dtid2pred_pred: Dict[int, Predicate] = {}
            dtid2pred_gold: Dict[int, Predicate] = {}
            for sid in document_pred.sid2sentence.keys():  # gold と pred で sid は共通
                if not (process_all or (sid == last_sid)):  # いらないかも
                    continue
                for predicate in self.sid2predicates_pred[sid]:
                    dtid2pred_pred[predicate.dtid] = predicate
                for predicate in self.sid2predicates_gold[sid]:
                    dtid2pred_gold[predicate.dtid] = predicate

            # calculate precision
            for dtid, predicate_pred in dtid2pred_pred.items():
                arguments_pred = document_pred.get_arguments(predicate_pred, relax=False)
                if dtid in dtid2pred_gold:
                    predicate_gold = dtid2pred_gold[dtid]
                    arguments_gold = document_gold.get_arguments(predicate_gold, relax=True)
                else:
                    predicate_gold = arguments_gold = None
                for case in self.cases:
                    if predicate_gold is not None:
                        args_gold = self._filter_args(arguments_gold[case], predicate_gold, self.relax_exophors)
                    else:
                        args_gold = []
                    args_pred: List[BaseArgument] = arguments_pred[case]
                    if not args_pred:
                        continue
                    assert len(args_pred) == 1  # in bert_pas_analysis, predict one argument for one predicate
                    arg = args_pred[0]
                    assert not (isinstance(arg, SpecialArgument) and arg.exophor not in target_exophors)
                    key = (doc_id, dtid, case)
                    analysis = self.deptype2analysis[arg.dep_type]
                    if arg in args_gold:
                        self.comp_result[key] = analysis
                        self.measures[case][analysis].correct += 1
                    else:
                        self.comp_result[key] = 'wrong'
                    self.measures[case][analysis].denom_pred += 1

            # calculate recall
            # 正解が複数ある場合、そのうち一つが当てられていればそれを正解に採用．
            # いずれも当てられていなければ、relax されていない項から一つを選び正解に採用．
            for dtid, predicate_gold in dtid2pred_gold.items():
                arguments_gold = document_gold.get_arguments(predicate_gold, relax=False)
                arguments_gold_relaxed = document_gold.get_arguments(predicate_gold, relax=True)
                if dtid in dtid2pred_pred:
                    predicate_pred = dtid2pred_pred[dtid]
                    arguments_pred = document_pred.get_arguments(predicate_pred, relax=False)
                else:
                    arguments_pred = None
                for case in self.cases:
                    args_pred: List[BaseArgument] = arguments_pred[case] if arguments_pred is not None else []
                    assert len(args_pred) in (0, 1)
                    args_gold = self._filter_args(arguments_gold[case], predicate_gold, self.relax_exophors)
                    args_gold_relaxed = self._filter_args(arguments_gold_relaxed[case], predicate_gold,
                                                          self.relax_exophors)
                    if not args_gold:
                        continue
                    correct = False
                    arg = args_gold[0]
                    for arg_ in args_gold_relaxed:
                        if arg_ in args_pred:
                            arg = arg_  # 予測されている項を優先して正解の項に採用
                            correct = True
                    key = (doc_id, dtid, case)
                    analysis = self.deptype2analysis[arg.dep_type]
                    if correct is True:
                        assert self.comp_result[key] == analysis
                    elif args_pred:
                        assert self.comp_result[key] == 'wrong'
                    else:
                        self.comp_result[key] = 'wrong'
                    self.measures[case][analysis].denom_gold += 1

    @staticmethod
    def _filter_args(args: List[BaseArgument],
                     predicate: Predicate,
                     relax_exophors: Dict[str, str]
                     ) -> List[BaseArgument]:
        filtered_args = []
        for arg in args:
            if isinstance(arg, SpecialArgument):
                if arg.exophor not in relax_exophors:  # filter out non-target exophors
                    continue
                arg.exophor = relax_exophors[arg.exophor]  # 「不特定:人１」なども「不特定:人」として扱う
            else:
                if Scorer._is_inter_sentential_cataphor(arg, predicate):  # filter out cataphoras
                    continue
            filtered_args.append(arg)
        return filtered_args

    @staticmethod
    def _is_inter_sentential_cataphor(arg: BaseArgument, predicate: Predicate):
        return isinstance(arg, Argument) and predicate.dtid < arg.dtid and arg.sid != predicate.sid

    def result_dict(self) -> Dict[str, Dict[str, 'Measure']]:
        result = OrderedDict()
        all_case_result = OrderedDefaultDict(lambda: Measure())
        for case, measures in self.measures.items():
            case_result = OrderedDefaultDict(lambda: Measure())
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
                lines.append(f'    precision: {measure.precision:.3f} ({measure.correct}/{measure.denom_pred})')
                lines.append(f'    recall   : {measure.recall:.3f} ({measure.correct}/{measure.denom_gold})')
                lines.append(f'    F        : {measure.f1:.3f}')
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
                # gold
                writer.write('<td><pre>\n')
                for sid in document_gold.sid2sentence.keys():
                    self._draw_tree(sid, self.sid2predicates_gold[sid], document_gold, fh=writer)
                    writer.write('\n')
                writer.write('</pre>')
                # prediction
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
                   predicates: List[Predicate],
                   document: Document,
                   fh=None,
                   html: bool = True
                   ) -> None:
        sentence: BList = document.sid2sentence[sid]
        with io.StringIO() as string:
            sentence.draw_tag_tree(fh=string)
            tree_strings = string.getvalue().rstrip('\n').split('\n')
        assert len(tree_strings) == len(sentence.tag_list())
        current_document_predicates = document.get_predicates()
        for predicate in predicates:
            if predicate not in current_document_predicates:
                continue
            idx = predicate.tid
            arguments = document.get_arguments(predicate)
            tree_strings[idx] += '  '
            for case in self.cases:
                args = arguments[case]
                if args:
                    arg = args[0].midasi
                    result = self.comp_result.get((document.doc_id, predicate.dtid, case), None)
                    if result == 'overt':
                        color = 'green'
                    elif result in self.deptype2analysis.values():
                        color = 'blue'
                    elif result == 'wrong':
                        if isinstance(args[0], SpecialArgument) and arg not in self.relax_exophors:
                            color = 'gray'
                        else:
                            color = 'red'
                    elif result is None:
                        color = 'gray'
                    else:
                        logger.warning(f'unknown result: {result}')
                        color = 'gray'
                else:
                    arg = 'NULL'
                    color = 'gray'
                if html:
                    tree_strings[idx] += f'<font color="{color}">{arg}:{case}</font> '
                else:
                    tree_strings[idx] += f'{arg}:{case} '

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
    parser.add_argument('--read-prediction-from-pas-tag', action='store_true', default=False,
                        help='use <述語項構造:> tag instead of <rel > tag in prediction files')
    args = parser.parse_args()

    reader_gold = KWDLCReader(
        Path(args.gold_dir),
        target_cases=args.case_string.split(','),
        target_corefs=['=', '=構', '=≒'],
        extract_nes=False
    )
    reader_pred = KWDLCReader(
        Path(args.prediction_dir),
        target_cases=reader_gold.target_cases,
        target_corefs=reader_gold.target_corefs,
        extract_nes=False,
        use_pas_tag=args.read_prediction_from_pas_tag,
    )
    documents_pred = list(reader_pred.process_all_documents())
    documents_gold = list(reader_gold.process_all_documents())

    scorer = Scorer(documents_pred, documents_gold, target_exophors=args.exophors.split(','))
    if args.result_html:
        scorer.write_html(Path(args.result_html))
    if args.result_csv:
        scorer.export_csv(args.result_csv)
    scorer.export_txt(sys.stdout)


if __name__ == '__main__':
    main()
