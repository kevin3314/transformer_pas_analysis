import os
import hashlib
import logging
from typing import List, Dict
from pathlib import Path
import _pickle as cPickle
from collections import OrderedDict

from pyknp import BList, Tag, Morpheme
from kyoto_reader import Document, BaseArgument, Argument, SpecialArgument, UNCERTAIN

logger = logging.getLogger(__file__)


def read_example(document: Document,
                 cases: List[str],
                 exophors: List[str],
                 coreference: bool,
                 bridging: bool,
                 kc: bool,
                 pas_targets: List[str],
                 dataset_config: dict,
                 ) -> 'PasExample':
    load_cache: bool = ('BPA_DISABLE_CACHE' not in os.environ and 'BPA_OVERWRITE_CACHE' not in os.environ)
    save_cache: bool = ('BPA_DISABLE_CACHE' not in os.environ)
    bpa_cache_dir: Path = Path(os.environ.get('BPA_CACHE_DIR', f'/data/{os.environ["USER"]}/bpa_cache'))
    example_hash = _hash(document, cases, exophors, coreference, bridging, kc, pas_targets, dataset_config)
    cache_path = bpa_cache_dir / example_hash / f'{document.doc_id}.pkl'
    if cache_path.exists() and load_cache:
        with cache_path.open('rb') as f:
            example = cPickle.load(f)
    else:
        example = PasExample()
        example.load(document,
                     cases=cases,
                     exophors=exophors,
                     coreference=coreference,
                     bridging=bridging,
                     kc=kc,
                     pas_targets=pas_targets)
        if save_cache:
            cache_path.parent.mkdir(exist_ok=True, parents=True)
            with cache_path.open('wb') as f:
                cPickle.dump(example, f)
    return example


def _hash(document, *args) -> str:
    attrs = ('cases', 'corefs', 'relax_cases', 'extract_nes', 'use_pas_tag')
    assert set(attrs) <= set(vars(document).keys())
    vars_document = {k: v for k, v in vars(document).items() if k in attrs}
    string = repr(sorted(vars_document)) + ''.join(repr(a) for a in args)
    return hashlib.md5(string.encode()).hexdigest()


class PasExample:
    """A single training/test example for pas analysis."""

    def __init__(self) -> None:
        self.words: List[str] = []
        self.arguments_set: List[Dict[str, List[str]]] = []
        self.arg_candidates_set: List[List[int]] = []
        self.ment_candidates_set: List[List[int]] = []
        self.dtids: List[int] = []  # dmid -> dtid
        self.ddeps: List[int] = []  # dmid -> dmid which has dep
        self.doc_id: str = ''

    def load(self,
             document: Document,
             cases: List[str],
             exophors: List[str],
             coreference: bool,
             bridging: bool,
             kc: bool,
             pas_targets: List[str],
             ) -> None:
        self.doc_id = document.doc_id
        process_all = (kc is False) or (document.doc_id.split('-')[-1] == '00')
        last_sent = document.sentences[-1] if len(document) > 0 else None
        relations = cases + (['ノ'] if bridging else []) + (['='] if coreference else [])
        relax_exophors = {}
        for exophor in exophors:
            relax_exophors[exophor] = exophor
            if exophor in ('不特定:人', '不特定:物', '不特定:状況'):
                for n in '１２３４５６７８９':
                    relax_exophors[exophor + n] = exophor
        dmid2arguments: Dict[int, Dict[str, List[BaseArgument]]] = {pred.dmid: document.get_arguments(pred)
                                                                    for pred in document.get_predicates()}
        dmid = 0
        head_dmids = []
        for sentence in document:
            process: bool = process_all or (sentence is last_sent)
            head_dmids += self._get_head_dmids(sentence, document.mrph2dmid)
            for tag in sentence.tag_list():
                mrph_list: List[Morpheme] = tag.mrph_list()
                if not mrph_list:
                    continue
                target_mrph = mrph_list[0]
                for mrph in mrph_list:
                    if '<内容語>' in mrph.fstring:
                        target_mrph = mrph
                        break
                for mrph in tag.mrph_list():
                    self.words.append(mrph.midasi)
                    self.dtids.append(document.tag2dtid[tag])
                    self.ddeps.append(document.tag2dtid[tag.parent] if tag.parent is not None else -1)
                    arguments = OrderedDict((rel, []) for rel in relations)
                    arg_candidates = ment_candidates = []
                    if mrph is target_mrph and process is True:
                        if ('pred' in pas_targets and '用言' in tag.features) or \
                                ('noun' in pas_targets and '非用言格解析' in tag.features):
                            arg_candidates = [x for x in head_dmids if x != dmid]
                            for case in cases:
                                dmid2args = {dmid: arguments[case] for dmid, arguments in dmid2arguments.items()}
                                arguments[case] = self._get_args(dmid, dmid2args, relax_exophors, arg_candidates)

                        if 'ノ' in relations:
                            arg_candidates = [x for x in head_dmids if x != dmid]
                            if '体言' in tag.features and '非用言格解析' not in tag.features:
                                dmid2args = {dmid: arguments['ノ'] for dmid, arguments in dmid2arguments.items()}
                                arguments['ノ'] = self._get_args(dmid, dmid2args, relax_exophors, arg_candidates)

                        if '=' in relations:
                            if '体言' in tag.features:
                                ment_candidates = [x for x in head_dmids if x < dmid]
                                arguments['='] = self._get_mentions(tag, document, relax_exophors, ment_candidates)

                    self.arguments_set.append(arguments)
                    self.arg_candidates_set.append(arg_candidates)
                    self.ment_candidates_set.append(ment_candidates)
                    dmid += 1

    def _get_args(self,
                  dmid: int,
                  dmid2args: Dict[int, List[BaseArgument]],
                  relax_exophors: Dict[str, str],
                  candidates: List[int],
                  ) -> List[str]:
        """述語の dmid と その項 dmid2args から、項の文字列を得る
        返り値が空リストの場合、この項について loss は計算されない
        overt: {dmid}%C
        case: {dmid}%N
        zero: {dmid}%O
        exophor: {exophor}
        no arg: NULL
        """
        # filter out non-target exophors
        args = []
        for arg in dmid2args.get(dmid, []):
            if isinstance(arg, SpecialArgument):
                if arg.exophor in relax_exophors:
                    arg.exophor = relax_exophors[arg.exophor]
                    args.append(arg)
                elif arg.exophor == UNCERTAIN:
                    return []  # don't train uncertain argument
            else:
                args.append(arg)
        if not args:
            return ['NULL']
        arg_strings: List[str] = []
        for arg in args:
            if isinstance(arg, Argument):
                if arg.dmid not in candidates:
                    logger.debug(f'argument: {arg.midasi} in {self.doc_id} is not in candidates and ignored')
                    continue
                string = str(arg.dmid)  # arg.dmid が内容語形態素を指しているという前提(kyoto-reader と密結合で良くない)
                if arg.dep_type == 'overt':
                    string += '%C'
                elif arg.dep_type == 'dep':
                    string += '%N'
                else:
                    assert arg.dep_type in ('intra', 'inter')
                    string += '%O'
            # exophor
            else:
                string = arg.midasi
            arg_strings.append(string)
        return arg_strings

    def _get_mentions(self,
                      tag: Tag,
                      document: Document,
                      relax_exophors: Dict[str, str],
                      candidates: List[int],
                      ) -> List[str]:
        ment_strings: List[str] = []
        dtid = document.tag2dtid[tag]
        if dtid in document.mentions:
            src_mention = document.mentions[dtid]
            tgt_mentions = document.get_siblings(src_mention)
            exophors = [document.entities[eid].exophor for eid in src_mention.eids
                        if document.entities[eid].is_special]
            for mention in tgt_mentions:
                if mention.dmid not in candidates:
                    logger.debug(f'mention: {mention.midasi} in {self.doc_id} is not in candidates and ignored')
                    continue
                ment_strings.append(str(mention.dmid))
            for exophor in exophors:
                if exophor in relax_exophors:
                    ment_strings.append(relax_exophors[exophor])  # 不特定:人１ -> 不特定:人
            if ment_strings:
                return ment_strings
            elif tgt_mentions:
                return []  # don't train cataphor
            else:
                return ['NA']
        else:
            return ['NA']

    @staticmethod
    def _get_head_dmids(sentence: BList, mrph2dmid: Dict[Morpheme, int]) -> List[int]:
        """sentence 中の基本句それぞれについて、内容語である形態素の dmid を返す

        内容語がなかった場合、先頭の形態素の dmid を返す

        Args:
            sentence (BList): 対象の文
            mrph2dmid (dict): 形態素IDと文書レベルの形態素IDを紐付ける辞書

        Returns:
            list: 各基本句に含まれる内容語形態素の文書レベル形態素ID
        """
        head_dmids = []
        for tag in sentence.tag_list():
            head_dmid = None
            for idx, mrph in enumerate(tag.mrph_list()):
                if idx == 0:
                    head_dmid = mrph2dmid[mrph]
                if '<内容語>' in mrph.fstring:
                    head_dmid = mrph2dmid[mrph]
                    break
            if head_dmid is not None:
                head_dmids.append(head_dmid)
        return head_dmids

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        string = ''
        for i, (word, args) in enumerate(zip(self.words, self.arguments_set)):
            pad = ' ' * (5 - len(word)) * 2
            string += f'{i:02} {word}{pad}({" ".join(f"{case}:{arg}" for case, arg in args.items())})\n'
        return string
