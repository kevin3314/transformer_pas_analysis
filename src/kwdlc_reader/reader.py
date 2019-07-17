import os
import glob
import logging
from typing import List, Dict, Optional
from collections import OrderedDict

# from kwdlc_reader.blist import BList
# from kwdlc_reader.tag import Tag
from pyknp import BList, Tag, Rel
from kwdlc_reader.pas import Pas, Argument
from kwdlc_reader.constants import ALL_CASES, CORE_CASES, ALL_EXOPHORS, ALL_COREFS, CORE_COREFS


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class KyotoCorpus:
    """https://bitbucket.org/ku_nlp/seqzero/src/master/seqzero/corpus_reader.py"""

    def __init__(self, dirname: str, glob_pat: str = "*.knp"):
        self.file_paths = self.get_file_paths(dirname, glob_pat)

    def load_files(self):
        for file_path in self.file_paths:
            yield KWDLCReader(file_path)

    @staticmethod
    def get_file_paths(dirname: str, glob_pat: str):
        return sorted(glob.glob(os.path.join(dirname, glob_pat)))


class Document:

    def __init__(self, sentences: List[BList], doc_id: str):
        self.sentences = sentences
        self.doc_id = doc_id
        self.sid2sent = {sentence.sid: sentence for sentence in sentences}

        self.entities = {}  # eid -> list of mention keys
        self.mentions = {}  # mention key -> {is_special, key, dmid_range}

    def bnst_list(self):
        return [bnst for sentence in self.sentences for bnst in sentence.bnst_list()]

    def tag_list(self):
        return [tag for sentence in self.sentences for tag in sentence.tag_list()]

    def mrph_list(self):
        return [mrph for sentence in self.sentences for mrph in sentence.mrph_list()]

    def add_corefs(self, tobj1: dict, tobj2: dict):
        eid1 = self.add_mention(tobj1)
        eid2 = self.add_mention(tobj2)
        if eid1 is None:
            if eid2 is None:
                eid = len(self.entities)
                self.entities[eid] = [tobj1["key"], tobj2["key"]]
                tobj1["eid"] = tobj2["eid"] = eid
            else:
                self.entities[eid2].append(tobj1["key"])
                tobj1["eid"] = eid2
        else:
            if eid2 is None:
                self.entities[eid1].append(tobj2["key"])
                tobj2["eid"] = eid1
            else:
                if not eid1 == eid2:
                    sys.stderr.write(f"different clusters: {tobj1['key']}\t{tobj2['key']}\n")

    def add_mention(self, tobj: dict):
        if tobj["key"] in self.mentions:
            return self.mentions[tobj["key"]]["eid"]
        else:
            self.mentions[tobj["key"]] = tobj
            return None

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, sid: str):
        return self.sid2sent[sid]

    def __iter__(self):
        return iter(self.sentences)


class KWDLCReader:
    def __init__(self,
                 file_path: str,
                 target_cases: Optional[List[str]] = None,
                 target_corefs: Optional[List[str]] = None,
                 target_exophors: Optional[List[str]] = None,
                 ) -> None:
        self.target_cases: List[str] = self._get_target(target_cases, ALL_CASES, CORE_CASES, 'case')
        self.target_corefs: List[str] = self._get_target(target_corefs, ALL_COREFS, CORE_COREFS, 'coref')
        self.target_exophors: List[str] = self._get_target(target_exophors, ALL_EXOPHORS, ALL_EXOPHORS, 'exophor')

        self.sid2sentence: Dict[str, BList] = OrderedDict()
        with open(file_path) as f:
            buff = ""
            for line in f:
                buff += line
                if line == "EOS\n":
                    sentence = BList(buff)
                    self.sid2sentence[sentence.sid] = sentence
                    buff = ""

        self.bnst2dbid = {}
        self.tag2dtid = {}
        self.mrph2dmid = {}
        self._assign_document_wide_id()

        self._pas: Dict[Tag, Pas] = self._extract_pas()

        self.mention2entity = {}

    @staticmethod
    def _get_target(input_: Optional[list], all_: list, default: list, type_: str) -> list:
        if input_ is None:
            return default
        target = []
        for item in input_:
            if item in all_:
                target.append(item)
            else:
                logger.warning(f'Unknown {type_}: {item}')
        return target

    # @staticmethod
    # def _read_knp_stream(f):
    #     buff = ""
    #     for line in f:
    #         buff += line
    #         if line == "EOS\n":
    #             yield BList(buff)
    #             buff = ""

    def _assign_document_wide_id(self):
        dbid, dtid, dmid = 0, 0, 0
        for sentence in self.sentences:
            for bnst in sentence.bnst_list():
                for tag in bnst.tag_list():
                    for mrph in tag.mrph_list():
                        self.mrph2dmid[mrph] = dmid
                        dmid += 1
                    self.tag2dtid[tag] = dtid
                    dtid += 1
                self.bnst2dbid[bnst] = dbid
                dbid += 1

        self.entities = []

    def _extract_pas(self) -> Dict[Tag, Pas]:
        tag2pas = OrderedDict()
        for tag in self.tag_list():
            if tag.features.rels is None:
                logger.debug(f'Tag: "{tag.midasi}" has no relation tags.')
                continue
            pas = Pas(self.tag2dtid[tag], tag.tag_id)
            print(tag.midasi)
            for rel in tag.features.rels:
                assert rel.ignore is False
                if rel.atype in self.target_cases:
                    if rel.sid is not None:
                        tag_list = self.sid2sentence[rel.sid].tag_list()
                        dtid = self.tag2dtid[tag_list[rel.tid]]
                    # exophora
                    else:
                        if rel.target not in ALL_EXOPHORS:
                            logger.warning(f'Unknown exophor: {rel.target}')
                            continue
                        elif rel.target not in self.target_exophors:
                            logger.info(f'Argument: {rel.target} ({rel.atype}) of {tag.midasi} is ignored.')
                            continue
                        dtid = None
                    pas.add_argument(rel.atype, rel.target, dtid, rel.tid, rel.mode)

            if pas.arguments:
                tag2pas[tag] = pas
        return tag2pas

    def _extract_coreference(self):
        pass

    @property
    def sentences(self):
        return list(self.sid2sentence.values())

    def bnst_list(self):
        return [bnst for sentence in self.sentences for bnst in sentence.bnst_list()]

    def tag_list(self):
        return [tag for sentence in self.sentences for tag in sentence.tag_list()]

    def mrph_list(self):
        return [mrph for sentence in self.sentences for mrph in sentence.mrph_list()]

    def get_all_entities(self) -> List[Entity]:
        return list(self.mention2entity.values())

    def get_entity(self, mention: Tag) -> Optional[Entity]:
        return self.mention2entity.get(mention, None)

    # def pas_list(self) -> List[Pas]:
    #     pas_list = []
    #     for sentence in self.sentences:
    #         for tag in sentence.tag_list():
    #             if tag.pas is not None:
    #                 pas_list.append(tag.pas)
    #     return pas_list


    # NEに関して
    # tag.features = {'NE': 'LOCATION:ダーマ神殿'}
    # などとなっている
    # なお、NEは対象NEの最後の基本句に付与されており、形態素単位
    # dmid_range を持っておく必要あり

    def pas_list(self) -> List[Pas]:
        return list(self._pas.values())

    def get_predicates(self) -> List[Tag]:
        return list(self._pas.keys())

    def get_arguments(self,
                      tag: Tag,
                      relax: bool = False,
                      ) -> List[Argument]:
        if tag not in self._pas:
            return []
        return self._pas[tag].arguments


if __name__ == '__main__':
    # path = 'data/train/w201106-0000060050.knp'
    path = 'data/train/w201106-0000074273.knp'
    kwdlc = KWDLCReader(path,
                        target_cases=['ガ', 'ヲ', 'ニ', 'ノ'],
                        target_exophors=['読者', '著者', '不特定:人'])
    kwdlc.get_entities()  # -> List[Entity]
    predicates: List[Tag] = kwdlc.get_predicates()

    # tags = kwdlc.tag_list()  # -> List[Tag]
    # predicates = list(filter(lambda x: x.pas is not None, tags))
    kwdlc.get_arguments(predicates[0], relax=True)  # -> List[Argument]
