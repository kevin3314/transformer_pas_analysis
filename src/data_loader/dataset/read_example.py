from typing import List, Dict, Optional
from collections import OrderedDict

from pyknp import BList, Morpheme

from kwdlc_reader import Document, BaseArgument, Argument, SpecialArgument, Entity


class PasExample:
    """A single training/test example for pas analysis."""

    def __init__(self,
                 words: List[str],
                 arguments_set: List[Dict[str, Optional[str]]],
                 arg_candidates_set: List[List[int]],
                 dtids: List[int],
                 ddeps: List[int],
                 doc_id: str,
                 ) -> None:
        self.words = words
        self.arguments_set = arguments_set
        self.arg_candidates_set = arg_candidates_set
        self.dtids = dtids  # dmid -> dtid
        self.ddeps = ddeps  # dmid -> dmid which has dep
        self.doc_id = doc_id

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        string = ''
        for i, (word, args) in enumerate(zip(self.words, self.arguments_set)):
            pad = ' ' * (5 - len(word)) * 2
            string += f'{i:02} {word}{pad}({" ".join(f"{case}:{arg}" for case, arg in args.items())})\n'
        return string


def read_example(document: Document,
                 target_exophors: List[str],
                 coreference: bool,
                 kc: bool,
                 ) -> PasExample:
    process_all = (kc is False) or (document.doc_id.split('-')[-1] == '00')
    last_sent = document.sentences[-1] if len(document) > 0 else None
    cases = document.target_cases
    relax_exophors = {}
    for exophor in target_exophors:
        relax_exophors[exophor] = exophor
        if exophor in ('不特定:人', '不特定:物', '不特定:状況'):
            for n in '１２３４５６７８９':
                relax_exophors[exophor + n] = exophor
    words, dtids, ddeps, arguments_set, arg_candidates_set = [], [], [], [], []
    dmid = 0
    head_dmids = []
    for sentence in document:
        process: bool = process_all or (sentence is last_sent)
        head_dmids += get_head_dmids(sentence, document.mrph2dmid)
        dmid2arguments: Dict[int, Dict[str, List[BaseArgument]]] = {pred.dmid: document.get_arguments(pred)
                                                                    for pred in document.get_predicates()}
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
                words.append(mrph.midasi)
                dtids.append(document.tag2dtid[tag])
                ddeps.append(document.tag2dtid[tag.parent] if tag.parent is not None else -1)
                if '用言' in tag.features \
                        and mrph is target_mrph \
                        and process is True:
                    arguments: Dict[str, Optional[str]] = OrderedDict()
                    for case in cases:
                        if dmid in dmid2arguments:
                            # filter out non-target exophors
                            args = []
                            for arg in dmid2arguments[dmid][case]:
                                if isinstance(arg, SpecialArgument):
                                    if arg.exophor in relax_exophors:
                                        arg.exophor = relax_exophors[arg.exophor]
                                        args.append(arg)
                                else:
                                    args.append(arg)
                            if not args:
                                arguments[case] = 'NULL'
                                continue
                            arg: BaseArgument = args[0]  # use first argument now
                            if isinstance(arg, Argument):
                                arguments[case] = str(arg.dmid)
                                if arg.dep_type == 'overt':
                                    arguments[case] += '%C'
                            # exophor
                            else:
                                arguments[case] = arg.midasi
                        else:
                            arguments[case] = 'NULL'
                    arg_candidates = [x for x in head_dmids if x != dmid]
                else:
                    arguments = OrderedDict((case, None) for case in cases)
                    arg_candidates = []

                if coreference:
                    if '体言' in tag.features \
                            and mrph is target_mrph \
                            and process is True:
                        entities: List[Entity] = document.get_entities(tag)
                        if entities:
                            exophors = [e.exophor for e in entities if e.is_special]
                            dtid = document.tag2dtid[tag]
                            mentions = set(m for e in entities for m in e.mentions if m.dtid != dtid)
                            preceding_mentions = [m for m in mentions if m.dtid < dtid].sort(key=lambda m: m.dtid)
                            if preceding_mentions:
                                arguments['='] = str(preceding_mentions[-1].dmid)  # choose nearest preceding mention
                            elif exophors and exophors[0] in relax_exophors:
                                arguments['='] = relax_exophors[exophors[0]]
                            elif mentions:
                                arguments['='] = str(list(mentions)[0].dmid)
                            else:
                                arguments['='] = 'NA'
                        else:
                            arguments['='] = 'NA'
                    else:
                        arguments['='] = None

                arguments_set.append(arguments)
                arg_candidates_set.append(arg_candidates)
                dmid += 1

    return PasExample(words, arguments_set, arg_candidates_set, dtids, ddeps, document.doc_id)


def get_head_dmids(sentence: BList, mrph2dmid: dict) -> List[int]:
    head_dmids = []
    for tag in sentence.tag_list():
        head_dmid = None
        for idx, mrph in enumerate(tag.mrph_list()):
            if idx == 0:
                head_dmid = mrph2dmid[mrph]
            if '<内容語>' in mrph.fstring:
                head_dmid = mrph2dmid[mrph]
                break
        head_dmids.append(head_dmid)
    return head_dmids
