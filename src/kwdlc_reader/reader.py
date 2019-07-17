import os
import glob
import logging
from typing import List, Dict, Optional
from collections import OrderedDict

# from kwdlc_reader.blist import BList
# from kwdlc_reader.tag import Tag
from pyknp import BList, Bunsetsu, Tag, Morpheme, Rel
from kwdlc_reader.pas import Pas, Argument
from kwdlc_reader.entity import Entity
from kwdlc_reader.constants import ALL_CASES, CORE_CASES, ALL_EXOPHORS, ALL_COREFS, CORE_COREFS, DEP_TYPES


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

"""
# TODO
- modeの扱い(ANDの他には?)
- dep_type
- named entity
- テストコード

# MEMO
- アノテーション基準には著者読者とのcorefは=:著者と書いてあるが、<rel>タグの中身はatype='=≒', target='著者'となっている
- corefタグは用言に対しても振られる
- Entityクラスに用言か体言かをもたせる？
- 前文/後文への照応もある(<rel type="=" target="後文"/>)
- 述語から項への係り受けもdep?
"""


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
        self.dtid2tag = {dtid: tag for tag, dtid in self.tag2dtid.items()}

        self._pas: Dict[Tag, Pas] = OrderedDict()
        self._mention2entity: Dict[Tag, Entity] = {}
        self._extract_relations()

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

    def _extract_relations(self):
        tag2sid = {tag: sentence.sid for sentence in self.sentences for tag in sentence.tag_list()}
        for tag in self.tag_list():
            if tag.features.rels is None:
                logger.debug(f'Tag: "{tag.midasi}" has no relation tags.')
                continue
            pas = Pas(tag, self.tag2dtid[tag], tag2sid[tag])
            for rel in tag.features.rels:
                assert rel.ignore is False
                # extract PAS
                if rel.atype in self.target_cases:
                    if rel.sid is not None:
                        assert rel.tid is not None
                        arg_tag = self.sid2sentence[rel.sid].tag_list()[rel.tid]
                        pas.add_argument(rel, arg_tag, self.tag2dtid[arg_tag])
                    # exophora
                    else:
                        if rel.target not in ALL_EXOPHORS:
                            logger.warning(f'Unknown exophor: {rel.target}')
                            continue
                        elif rel.target not in self.target_exophors:
                            logger.info(f'Argument: {rel.target} ({rel.atype}) of {tag.midasi} is ignored.')
                            continue
                        pas.add_argument(rel, None, None)

                # extract coreference
                elif rel.atype in self.target_corefs:
                    self._add_corefs(tag, rel)

            if pas.arguments:
                self._pas[tag] = pas

    def _add_corefs(self, source_mention: Tag, rel):
        if rel.sid is not None:
            tag_list = self.sid2sentence[rel.sid].tag_list()
            target_dtid = self.tag2dtid[tag_list[rel.tid]]
            if target_dtid >= self.tag2dtid[source_mention]:
                logger.warning('Coreference with self or latter entity was found.')
                return
            target_mention = self.dtid2tag[target_dtid]
            if target_mention in self._mention2entity:
                entity = self._mention2entity[target_mention]
            else:
                entity = Entity.create()
                self._add_mention(target_mention, entity)
        # exophora
        else:
            if rel.target not in ALL_EXOPHORS:
                logger.warning(f'Unknown exophor: {rel.target}')
                return
            elif rel.target not in self.target_exophors:
                logger.info(f'Coreference with {rel.target} ({rel.atype}) of {source_mention.midasi} is ignored.')
                return
            entity = Entity.create(exophor=rel.target)
        self._add_mention(source_mention, entity)

    def _add_mention(self, mention, entity):
        entity.add_mention(mention)
        self._mention2entity[mention] = entity

    @property
    def sentences(self) -> List[BList]:
        return list(self.sid2sentence.values())

    def bnst_list(self) -> List[Bunsetsu]:
        return [bnst for sentence in self.sentences for bnst in sentence.bnst_list()]

    def tag_list(self) -> List[Tag]:
        return [tag for sentence in self.sentences for tag in sentence.tag_list()]

    def mrph_list(self) -> List[Morpheme]:
        return [mrph for sentence in self.sentences for mrph in sentence.mrph_list()]

    def get_all_entities(self) -> List[Entity]:
        return list(self._mention2entity.values())

    def get_entity(self, mention: Tag) -> Optional[Entity]:
        return self._mention2entity.get(mention, None)


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
    path = 'data/train/w201106-0000060050.knp'
    # path = 'data/train/w201106-0000074273.knp'
    kwdlc = KWDLCReader(path,
                        target_cases=['ガ', 'ヲ', 'ニ', 'ノ'],
                        target_corefs=["=", "=構", "=≒"],
                        target_exophors=['読者', '著者', '不特定:人'])
    kwdlc.get_all_entities()  # -> List[Entity]
    predicates: List[Tag] = kwdlc.get_predicates()

    # tags = kwdlc.tag_list()  # -> List[Tag]
    # predicates = list(filter(lambda x: x.pas is not None, tags))
    kwdlc.get_arguments(predicates[0], relax=True)  # -> List[Argument]
