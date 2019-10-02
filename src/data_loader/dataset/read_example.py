from typing import List, Dict, Optional
from collections import OrderedDict

from pyknp import BList, Tag

from kwdlc_reader import Document


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
                 coreference: bool,
                 kc: bool
                 ) -> PasExample:
    process_all = (kc is False) or (document.doc_id.split('-')[-1] == '00')
    last_sent = document.sentences[-1] if len(document) > 0 else None
    cases = document.target_cases
    words, dtids, ddeps, arguments_set, arg_candidates_set = [], [], [], [], []
    dmid = 0
    head_dmids = []
    for sentence in document:
        process: bool = process_all or (sentence is last_sent)
        head_dmids += get_head_dmids(sentence, document.mrph2dmid)
        dmid2pred: Dict[int, Tag] = {pas.dmid: pas.predicate for pas in document.pas_list()}
        for tag in sentence.tag_list():
            pas_head_found = False
            for mrph in tag.mrph_list():
                words.append(mrph.midasi)
                dtids.append(document.tag2dtid[tag])
                ddeps.append(document.tag2dtid[tag.parent] if tag.parent is not None else -1)
                if '<用言:' in tag.fstring \
                        and '<省略解析なし>' not in tag.fstring \
                        and '<内容語>' in mrph.fstring \
                        and pas_head_found is False \
                        and process is True:
                    arguments: Dict[str, str] = OrderedDict()
                    for case in cases:
                        if dmid in dmid2pred:
                            case2args = document.get_arguments(dmid2pred[dmid], relax=True)
                            if case not in case2args:
                                arguments[case] = 'NULL'
                                continue
                            arg = case2args[case][0]  # use first argument now
                            # exophor
                            if arg.dep_type == 'exo':
                                arguments[case] = arg.midasi
                            # overt
                            elif arg.dep_type == 'overt':
                                arguments[case] = str(arg.dmid) + '%C'
                            # normal
                            else:
                                arguments[case] = str(arg.dmid)
                        else:
                            arguments[case] = 'NULL'
                    arg_candidates = [x for x in head_dmids if x != dmid]
                    pas_head_found = True
                else:
                    arguments = OrderedDict((case, None) for case in cases)
                    arg_candidates = []

                # TODO: coreference
                if coreference:
                    if '<体言>' in tag.fstring and '<内容語>' in mrph.fstring:
                        entity = document.get_entity(tag)
                        if entity is None:
                            arguments['='] = 'NA'
                        else:
                            document.get_all_mentions()

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
