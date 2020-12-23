import argparse
import io
import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Set, Union, Optional, TextIO

from jinja2 import Template, Environment, FileSystemLoader
from kyoto_reader import KyotoReader, Document, Argument, SpecialArgument, BaseArgument, Predicate, Mention, BasePhrase
from pyknp import BList

from utils.constants import CASE2YOMI
from utils.util import OrderedDefaultDict, is_pas_target, is_bridging_target, is_coreference_target

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Scorer:
    DEPTYPE2ANALYSIS = OrderedDict([('overt', 'overt'),
                                    ('dep', 'case'),
                                    ('intra', 'zero_intra'),
                                    ('inter', 'zero_inter'),
                                    ('exo', 'zero_exophora')])

    def __init__(self,
                 documents_pred: List[Document],
                 documents_gold: List[Document],
                 target_cases: List[str],
                 target_exophors: List[str],
                 coreference: bool = False,
                 bridging: bool = False,
                 pas_target: str = 'pred'):
        # long document may have been ignored
        assert set(doc.doc_id for doc in documents_pred) <= set(doc.doc_id for doc in documents_gold)
        self.cases: List[str] = target_cases
        self.bridging: bool = bridging
        self.doc_ids: List[str] = [doc.doc_id for doc in documents_pred]
        self.did2document_pred: Dict[str, Document] = {doc.doc_id: doc for doc in documents_pred}
        self.did2document_gold: Dict[str, Document] = {doc.doc_id: doc for doc in documents_gold}
        self.coreference = coreference
        self.comp_result: Dict[tuple, str] = {}
        self.measures: Dict[str, Dict[str, Measure]] = OrderedDict(
            (case, OrderedDict((anal, Measure()) for anal in Scorer.DEPTYPE2ANALYSIS.values()))
            for case in self.cases)
        self.measure_coref: Measure = Measure()
        self.measures_bridging: Dict[str, Measure] = OrderedDict(
            (anal, Measure()) for anal in Scorer.DEPTYPE2ANALYSIS.values())
        self.relax_exophors: Dict[str, str] = {}
        for exophor in target_exophors:
            self.relax_exophors[exophor] = exophor
            if exophor in ('不特定:人', '不特定:物', '不特定:状況'):
                for n in ('１', '２', '３', '４', '５', '６', '７', '８', '９', '１０', '１１'):
                    self.relax_exophors[exophor + n] = exophor

        self.did2predicates_pred: Dict[str, List[BasePhrase]] = OrderedDefaultDict(list)
        self.did2predicates_gold: Dict[str, List[BasePhrase]] = OrderedDefaultDict(list)
        self.did2bridgings_pred: Dict[str, List[BasePhrase]] = OrderedDefaultDict(list)
        self.did2bridgings_gold: Dict[str, List[BasePhrase]] = OrderedDefaultDict(list)
        self.did2mentions_pred: Dict[str, List[BasePhrase]] = OrderedDefaultDict(list)
        self.did2mentions_gold: Dict[str, List[BasePhrase]] = OrderedDefaultDict(list)
        for doc_id in self.doc_ids:
            for bp in self.did2document_pred[doc_id].bp_list():
                if is_pas_target(bp, verbal=(pas_target in ('pred', 'all')), nominal=(pas_target in ('noun', 'all'))):
                    self.did2predicates_pred[doc_id].append(bp)
                if self.bridging and is_bridging_target(bp):
                    self.did2bridgings_pred[doc_id].append(bp)
                if self.coreference and is_coreference_target(bp):
                    self.did2mentions_pred[doc_id].append(bp)
            for bp in self.did2document_gold[doc_id].bp_list():
                if is_pas_target(bp, verbal=(pas_target in ('pred', 'all')), nominal=(pas_target in ('noun', 'all'))):
                    self.did2predicates_gold[doc_id].append(bp)
                if self.bridging and is_bridging_target(bp):
                    self.did2bridgings_gold[doc_id].append(bp)
                if self.coreference and is_coreference_target(bp):
                    self.did2mentions_gold[doc_id].append(bp)

        for doc_id in self.doc_ids:
            document_pred = self.did2document_pred[doc_id]
            document_gold = self.did2document_gold[doc_id]
            self._evaluate_pas(doc_id, document_pred, document_gold)
            if self.bridging:
                self._evaluate_bridging(doc_id, document_pred, document_gold)
            if self.coreference:
                self._evaluate_coref(doc_id, document_pred, document_gold)

    def _evaluate_pas(self, doc_id: str, document_pred: Document, document_gold: Document):
        """calculate PAS analysis scores"""
        dtid2predicate_pred: Dict[int, Predicate] = {pred.dtid: pred for pred in self.did2predicates_pred[doc_id]}
        dtid2predicate_gold: Dict[int, Predicate] = {pred.dtid: pred for pred in self.did2predicates_gold[doc_id]}

        for dtid in range(len(document_pred.bp_list())):
            if dtid in dtid2predicate_pred:
                predicate_pred = dtid2predicate_pred[dtid]
                arguments_pred = document_pred.get_arguments(predicate_pred, relax=False)
            else:
                arguments_pred = None

            if dtid in dtid2predicate_gold:
                predicate_gold = dtid2predicate_gold[dtid]
                arguments_gold = document_gold.get_arguments(predicate_gold, relax=False)
                arguments_gold_relaxed = document_gold.get_arguments(predicate_gold, relax=True)
            else:
                predicate_gold = arguments_gold = arguments_gold_relaxed = None

            for case in self.cases:
                args_pred: List[BaseArgument] = arguments_pred[case] if arguments_pred is not None else []
                assert len(args_pred) in (0, 1)  # in bert_pas_analysis, predict one argument for one predicate
                if predicate_gold is not None:
                    args_gold = self._filter_args(arguments_gold[case], predicate_gold)
                    args_gold_relaxed = self._filter_args(
                        arguments_gold_relaxed[case] + (arguments_gold_relaxed['判ガ'] if case == 'ガ' else []),
                        predicate_gold)
                else:
                    args_gold = args_gold_relaxed = []

                key = (doc_id, dtid, case)

                # calculate precision
                if args_pred:
                    arg = args_pred[0]
                    analysis = Scorer.DEPTYPE2ANALYSIS[arg.dep_type]
                    if arg in args_gold_relaxed:
                        self.comp_result[key] = analysis
                        self.measures[case][analysis].correct += 1
                    else:
                        self.comp_result[key] = 'wrong'  # precision が下がる
                    self.measures[case][analysis].denom_pred += 1

                # calculate recall
                # 正解が複数ある場合、そのうち一つが当てられていればそれを正解に採用．
                # いずれも当てられていなければ、relax されていない項から一つを選び正解に採用．
                if args_gold or (self.comp_result.get(key, None) in Scorer.DEPTYPE2ANALYSIS.values()):
                    arg_gold = None
                    for arg in args_gold_relaxed:
                        if arg in args_pred:
                            arg_gold = arg  # 予測されている項を優先して正解の項に採用
                            break
                    if arg_gold is not None:
                        analysis = Scorer.DEPTYPE2ANALYSIS[arg_gold.dep_type]
                        assert self.comp_result[key] == analysis
                    else:
                        analysis = Scorer.DEPTYPE2ANALYSIS[args_gold[0].dep_type]
                        if args_pred:
                            assert self.comp_result[key] == 'wrong'
                        else:
                            self.comp_result[key] = 'wrong'  # recall が下がる
                    self.measures[case][analysis].denom_gold += 1

    def _filter_args(self,
                     args: List[BaseArgument],
                     predicate: Predicate,
                     ) -> List[BaseArgument]:
        filtered_args = []
        for arg in args:
            if isinstance(arg, SpecialArgument):
                if arg.exophor not in self.relax_exophors:  # filter out non-target exophors
                    continue
                arg.exophor = self.relax_exophors[arg.exophor]  # 「不特定:人１」なども「不特定:人」として扱う
            else:
                assert isinstance(arg, Argument)
                # filter out self-anaphora and cataphoras
                if predicate.dtid == arg.dtid or (predicate.dtid < arg.dtid and arg.sid != predicate.sid):
                    continue
            filtered_args.append(arg)
        return filtered_args

    def _evaluate_bridging(self, doc_id: str, document_pred: Document, document_gold: Document):
        dtid2anaphor_pred: Dict[int, Predicate] = {pred.dtid: pred for pred in self.did2bridgings_pred[doc_id]}
        dtid2anaphor_gold: Dict[int, Predicate] = {pred.dtid: pred for pred in self.did2bridgings_gold[doc_id]}

        for dtid in range(len(document_pred.bp_list())):
            if dtid in dtid2anaphor_pred:
                anaphor_pred = dtid2anaphor_pred[dtid]
                antecedents_pred: List[BaseArgument] = \
                    self._filter_args(document_pred.get_arguments(anaphor_pred, relax=False)['ノ'], anaphor_pred)
            else:
                antecedents_pred = []
            assert len(antecedents_pred) in (0, 1)  # in bert_pas_analysis, predict one argument for one predicate

            if dtid in dtid2anaphor_gold:
                anaphor_gold: Predicate = dtid2anaphor_gold[dtid]
                antecedents_gold: List[BaseArgument] = \
                    self._filter_args(document_gold.get_arguments(anaphor_gold, relax=False)['ノ'], anaphor_gold)
                antecedents_gold_relaxed: List[BaseArgument] = \
                    self._filter_args(document_gold.get_arguments(anaphor_gold, relax=True)['ノ'] +
                                      document_gold.get_arguments(anaphor_gold, relax=True)['ノ？'], anaphor_gold)
            else:
                antecedents_gold = antecedents_gold_relaxed = []

            key = (doc_id, dtid, 'ノ')

            # calculate precision
            if antecedents_pred:
                antecedent_pred = antecedents_pred[0]
                analysis = Scorer.DEPTYPE2ANALYSIS[antecedent_pred.dep_type]
                if antecedent_pred in antecedents_gold_relaxed:
                    self.comp_result[key] = analysis
                    self.measures_bridging[analysis].correct += 1
                else:
                    self.comp_result[key] = 'wrong'
                self.measures_bridging[analysis].denom_pred += 1

            # calculate recall
            if antecedents_gold or (self.comp_result.get(key, None) in Scorer.DEPTYPE2ANALYSIS.values()):
                antecedent_gold = None
                for ant in antecedents_gold_relaxed:
                    if ant in antecedents_pred:
                        antecedent_gold = ant  # 予測されている先行詞を優先して正解の先行詞に採用
                        break
                if antecedent_gold is not None:
                    analysis = Scorer.DEPTYPE2ANALYSIS[antecedent_gold.dep_type]
                    assert self.comp_result[key] == analysis
                else:
                    analysis = Scorer.DEPTYPE2ANALYSIS[antecedents_gold[0].dep_type]
                    if antecedents_pred:
                        assert self.comp_result[key] == 'wrong'
                    else:
                        self.comp_result[key] = 'wrong'
                self.measures_bridging[analysis].denom_gold += 1

    def _evaluate_coref(self, doc_id: str, document_pred: Document, document_gold: Document):
        dtid2mention_pred: Dict[int, Mention] = {bp.dtid: document_pred.mentions[bp.dtid]
                                                 for bp in self.did2mentions_pred[doc_id]
                                                 if bp.dtid in document_pred.mentions}
        dtid2mention_gold: Dict[int, Mention] = {bp.dtid: document_gold.mentions[bp.dtid]
                                                 for bp in self.did2mentions_gold[doc_id]
                                                 if bp.dtid in document_gold.mentions}

        for dtid in range(len(document_pred.bp_list())):
            if dtid in dtid2mention_pred:
                src_mention_pred = dtid2mention_pred[dtid]
                tgt_mentions_pred = \
                    self._filter_mentions(document_pred.get_siblings(src_mention_pred), src_mention_pred)
                exophors_pred = [e.exophor for e in map(document_pred.entities.get, src_mention_pred.eids)
                                 if e.is_special]
            else:
                tgt_mentions_pred = exophors_pred = []

            if dtid in dtid2mention_gold:
                src_mention_gold = dtid2mention_gold[dtid]
                tgt_mentions_gold = \
                    self._filter_mentions(document_gold.get_siblings(src_mention_gold, relax=False), src_mention_gold)
                tgt_mentions_gold_relaxed = \
                    self._filter_mentions(document_gold.get_siblings(src_mention_gold, relax=True), src_mention_gold)
                exophors_gold = [e.exophor for e in map(document_gold.entities.get, src_mention_gold.eids)
                                 if e.is_special and e.exophor in self.relax_exophors.values()]
                exophors_gold_relaxed = [e.exophor for e in map(document_gold.entities.get, src_mention_gold.all_eids)
                                         if e.is_special and e.exophor in self.relax_exophors.values()]
            else:
                tgt_mentions_gold = tgt_mentions_gold_relaxed = exophors_gold = exophors_gold_relaxed = []

            key = (doc_id, dtid, '=')

            # calculate precision
            if tgt_mentions_pred or exophors_pred:
                if (set(tgt_mentions_pred) & set(tgt_mentions_gold_relaxed)) \
                        or (set(exophors_pred) & set(exophors_gold_relaxed)):
                    self.comp_result[key] = 'correct'
                    self.measure_coref.correct += 1
                else:
                    self.comp_result[key] = 'wrong'
                self.measure_coref.denom_pred += 1

            # calculate recall
            if tgt_mentions_gold or exophors_gold or (self.comp_result.get(key, None) == 'correct'):
                if (set(tgt_mentions_pred) & set(tgt_mentions_gold_relaxed)) \
                        or (set(exophors_pred) & set(exophors_gold_relaxed)):
                    assert self.comp_result[key] == 'correct'
                else:
                    self.comp_result[key] = 'wrong'
                self.measure_coref.denom_gold += 1

    @staticmethod
    def _filter_mentions(tgt_mentions: Set[Mention], src_mention: Mention) -> List[Mention]:
        return [tgt_mention for tgt_mention in tgt_mentions if tgt_mention.dtid < src_mention.dtid]

    def result_dict(self) -> Dict[str, Dict[str, 'Measure']]:
        result = OrderedDict()
        all_case_result = OrderedDefaultDict(lambda: Measure())
        for case, measures in self.measures.items():
            case_result = OrderedDefaultDict(lambda: Measure())
            case_result.update(measures)
            case_result['zero'] = case_result['zero_intra'] + case_result['zero_inter'] + case_result['zero_exophora']
            case_result['case_zero'] = case_result['zero'] + case_result['case']
            case_result['all'] = case_result['case_zero'] + case_result['overt']
            for analysis, measure in case_result.items():
                all_case_result[analysis] += measure
            result[case] = case_result

        if self.coreference:
            all_case_result['coreference'] = self.measure_coref

        if self.bridging:
            case_result = OrderedDefaultDict(lambda: Measure())
            case_result.update(self.measures_bridging)
            case_result['zero'] = case_result['zero_intra'] + case_result['zero_inter'] + case_result['zero_exophora']
            case_result['case_zero'] = case_result['zero'] + case_result['case']
            case_result['all'] = case_result['case_zero'] + case_result['overt']
            all_case_result['bridging'] = case_result['all']

        result['all_case'] = all_case_result
        return result

    def export_txt(self, destination: Union[str, Path, TextIO]):
        lines = []
        for case, measures in self.result_dict().items():
            if case in self.cases:
                lines.append(f'{case}格')
            else:
                lines.append(f'{case}')
            for analysis, measure in measures.items():
                lines.append(f'  {analysis}')
                lines.append(f'    precision: {measure.precision:.4f} ({measure.correct}/{measure.denom_pred})')
                lines.append(f'    recall   : {measure.recall:.4f} ({measure.correct}/{measure.denom_gold})')
                lines.append(f'    F        : {measure.f1:.4f}')
        text = '\n'.join(lines) + '\n'

        if isinstance(destination, str) or isinstance(destination, Path):
            with Path(destination).open('wt') as writer:
                writer.write(text)
        elif isinstance(destination, io.TextIOBase):
            destination.write(text)

    def export_csv(self, destination: Union[str, Path, TextIO], sep: str = ','):
        text = ''
        result_dict = self.result_dict()
        text += 'case' + sep
        text += sep.join(result_dict['all_case'].keys()) + '\n'
        for case, measures in result_dict.items():
            text += CASE2YOMI.get(case, case) + sep
            text += sep.join(f'{measure.f1:.5}' for measure in measures.values())
            text += '\n'

        if isinstance(destination, str) or isinstance(destination, Path):
            with Path(destination).open('wt') as writer:
                writer.write(text)
        elif isinstance(destination, io.TextIOBase):
            destination.write(text)

    def write_html(self, output_file: Union[str, Path]):
        data: List[tuple] = []
        for doc_id in self.doc_ids:
            document_gold = self.did2document_gold[doc_id]
            document_pred = self.did2document_pred[doc_id]

            gold_tree = ''
            for sid in document_gold.sid2sentence.keys():
                with io.StringIO() as string:
                    self._draw_tree(sid,
                                    self.did2predicates_gold[doc_id],
                                    self.did2mentions_gold[doc_id],
                                    self.did2bridgings_gold[doc_id],
                                    document_gold,
                                    fh=string)
                    gold_tree += string.getvalue()

            pred_tree = ''
            for sid in document_pred.sid2sentence.keys():
                with io.StringIO() as string:
                    self._draw_tree(sid,
                                    self.did2predicates_pred[doc_id],
                                    self.did2mentions_pred[doc_id],
                                    self.did2bridgings_pred[doc_id],
                                    document_pred,
                                    fh=string)
                    pred_tree += string.getvalue()
            data.append((document_gold.sentences, gold_tree, pred_tree))

        env = Environment(loader=FileSystemLoader('.'))
        template: Template = env.get_template('src/template.html')

        with Path(output_file).open('wt') as f:
            f.write(template.render({'data': data}))

    def _draw_tree(self,
                   sid: str,
                   predicates: List[BasePhrase],
                   mentions: List[BasePhrase],
                   anaphors: List[BasePhrase],
                   document: Document,
                   fh: Optional[TextIO] = None,
                   html: bool = True
                   ) -> None:
        """sid で指定された文の述語項構造・共参照関係をツリー形式で fh に書き出す

        Args:
            sid (str): 出力対象の文ID
            predicates (List[BasePhrase]): document に含まれる全ての述語
            mentions (List[BasePhrase]): document に含まれる全ての mention
            anaphors (List[BasePhrase]): document に含まれる全ての橋渡し照応詞
            document (Document): sid が含まれる文書
            fh (Optional[TextIO]): 出力ストリーム
            html (bool): html 形式で出力するかどうか
        """
        result2color = {anal: 'blue' for anal in Scorer.DEPTYPE2ANALYSIS.values()}
        result2color.update({'overt': 'green', 'wrong': 'red', None: 'gray'})
        result2color_coref = {'correct': 'blue', 'wrong': 'red', None: 'gray'}
        blist: BList = document.sid2sentence[sid].blist
        with io.StringIO() as string:
            blist.draw_tag_tree(fh=string, show_pos=False)
            tree_strings = string.getvalue().rstrip('\n').split('\n')
        assert len(tree_strings) == len(blist.tag_list())
        all_targets = [m.core for m in document.mentions.values()]
        tid2predicate: Dict[int, BasePhrase] = {predicate.tid: predicate for predicate in predicates
                                                if predicate.sid == sid}
        tid2mention: Dict[int, BasePhrase] = {mention.tid: mention for mention in mentions if mention.sid == sid}
        tid2bridging: Dict[int, BasePhrase] = {anaphor.tid: anaphor for anaphor in anaphors if anaphor.sid == sid}
        for tid in range(len(tree_strings)):
            tree_strings[tid] += '  '
            if tid in tid2predicate:
                predicate = tid2predicate[tid]
                arguments = document.get_arguments(predicate)
                for case in self.cases:
                    args = arguments[case]
                    if case == 'ガ':
                        args += arguments['判ガ']
                    targets = set()
                    for arg in args:
                        target = str(arg)
                        if all_targets.count(str(arg)) > 1 and isinstance(arg, Argument):
                            target += str(arg.dtid)
                        targets.add(target)
                    result = self.comp_result.get((document.doc_id, predicate.dtid, case), None)
                    if html:
                        tree_strings[tid] += f'<font color="{result2color[result]}">{",".join(targets)}:{case}</font> '
                    else:
                        tree_strings[tid] += f'{",".join(targets)}:{case} '

            if self.bridging and tid in tid2bridging:
                anaphor = tid2bridging[tid]
                arguments = document.get_arguments(anaphor)
                args = arguments['ノ'] + arguments['ノ？']
                targets = set()
                for arg in args:
                    target = str(arg)
                    if all_targets.count(str(arg)) > 1 and isinstance(arg, Argument):
                        target += str(arg.dtid)
                    targets.add(target)
                result = self.comp_result.get((document.doc_id, anaphor.dtid, 'ノ'), None)
                if html:
                    tree_strings[tid] += f'<font color="{result2color[result]}">{",".join(targets)}:ノ</font> '
                else:
                    tree_strings[tid] += f'{",".join(targets)}:ノ '

            if self.coreference and tid in tid2mention:
                targets = set()
                src_dtid = tid2mention[tid].dtid
                if src_dtid in document.mentions:
                    src_mention = document.mentions[src_dtid]
                    tgt_mentions_relaxed = self._filter_mentions(
                        document.get_siblings(src_mention, relax=True), src_mention)
                    for tgt_mention in tgt_mentions_relaxed:
                        target: str = tgt_mention.core
                        if all_targets.count(target) > 1:
                            target += str(tgt_mention.dtid)
                        targets.add(target)
                    for eid in src_mention.eids:
                        entity = document.entities[eid]
                        if entity.exophor in self.relax_exophors.values():
                            targets.add(entity.exophor)
                result = self.comp_result.get((document.doc_id, src_dtid, '='), None)
                if html:
                    tree_strings[tid] += f'<font color="{result2color_coref[result]}">＝:{",".join(targets)}</font>'
                else:
                    tree_strings[tid] += '＝:' + ','.join(targets)

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
    parser.add_argument('--coreference', '--coref', '--cr', action='store_true', default=False,
                        help='perform coreference resolution')
    parser.add_argument('--bridging', '--brg', '--bar', action='store_true', default=False,
                        help='perform bridging anaphora resolution')
    parser.add_argument('--case-string', type=str, default='ガ,ヲ,ニ,ガ２',
                        help='case strings separated by ","')
    parser.add_argument('--exophors', '--exo', type=str, default='著者,読者,不特定:人,不特定:物',
                        help='exophor strings separated by ","')
    parser.add_argument('--read-prediction-from-pas-tag', action='store_true', default=False,
                        help='use <述語項構造:> tag instead of <rel > tag in prediction files')
    parser.add_argument('--pas-target', choices=['', 'pred', 'noun', 'all'], default='pred',
                        help='PAS analysis evaluation target (pred: verbal predicates, noun: nominal predicates)')
    parser.add_argument('--result-html', default=None, type=str,
                        help='path to html file which prediction result is exported (default: None)')
    parser.add_argument('--result-csv', default=None, type=str,
                        help='path to csv file which prediction result is exported (default: None)')
    args = parser.parse_args()

    reader_gold = KyotoReader(Path(args.gold_dir), extract_nes=False, use_pas_tag=False)
    reader_pred = KyotoReader(
        Path(args.prediction_dir),
        extract_nes=False,
        use_pas_tag=args.read_prediction_from_pas_tag,
    )
    documents_pred = list(reader_pred.process_all_documents())
    documents_gold = list(reader_gold.process_all_documents())

    assert set(args.case_string.split(',')) <= set(CASE2YOMI.keys())
    msg = '"ノ" found in case string. If you want to perform bridging anaphora resolution, specify "--bridging" ' \
          'option instead'
    assert 'ノ' not in args.case_string.split(','), msg
    scorer = Scorer(documents_pred, documents_gold,
                    target_cases=args.case_string.split(','),
                    target_exophors=args.exophors.split(','),
                    coreference=args.coreference,
                    bridging=args.bridging,
                    pas_target=args.pas_target)
    if args.result_html:
        scorer.write_html(Path(args.result_html))
    if args.result_csv:
        scorer.export_csv(args.result_csv)
    scorer.export_txt(sys.stdout)


if __name__ == '__main__':
    main()
