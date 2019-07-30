import os
import glob
import logging
from typing import List, Dict, Optional
from collections import OrderedDict

from pyknp import BList, Bunsetsu, Tag, Morpheme, Rel
from kwdlc_reader.pas import Pas, Argument
from kwdlc_reader.coreference import Mention, Entity
from kwdlc_reader.ne import NamedEntity
from kwdlc_reader.constants import ALL_CASES, CORE_CASES, ALL_EXOPHORS, ALL_COREFS, CORE_COREFS, NE_CATEGORIES


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

"""
# TODO
- relax match

# MEMO
- アノテーション基準には著者読者とのcorefは=:著者と書いてあるが、<rel>タグの中身はatype='=≒', target='著者'となっている
- corefタグは用言に対しても振られる
- Entityクラスに用言か体言かをもたせる？
- 用言かつ体言の基本句もある
- 前文/後文への照応もある(<rel type="=" target="後文"/>)
- 述語から項への係り受けもdep?
- BasePhraseクラス作っちゃう？
"""


class KyotoCorpus:
    """https://bitbucket.org/ku_nlp/seqzero/src/master/seqzero/corpus_reader.py"""

    def __init__(self, dirname: str, glob_pat: str = "*.knp", **kwargs):
        self.file_paths = self.get_file_paths(dirname, glob_pat)
        self.kwargs = kwargs

    def load_files(self):
        for file_path in self.file_paths:
            yield KWDLCReader(file_path, **self.kwargs)

    @staticmethod
    def get_file_paths(dirname: str, glob_pat: str):
        return sorted(glob.glob(os.path.join(dirname, glob_pat)))


class KWDLCReader:
    def __init__(self,
                 file_path: str,
                 target_cases: Optional[List[str]] = None,
                 target_corefs: Optional[List[str]] = None,
                 target_exophors: Optional[List[str]] = None,
                 extract_nes: bool = True,
                 ) -> None:
        self.target_cases: List[str] = self._get_target(target_cases, ALL_CASES, CORE_CASES, 'case')
        self.target_corefs: List[str] = self._get_target(target_corefs, ALL_COREFS, CORE_COREFS, 'coref')
        self.target_exophors: List[str] = self._get_target(target_exophors, ALL_EXOPHORS, ALL_EXOPHORS, 'exophor')
        self.extract_nes: bool = extract_nes

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
        self._mentions: Dict[int, Mention] = OrderedDict()
        self._entities: List[Entity] = []
        self._extract_relations()

        if extract_nes:
            self.named_entities: List[NamedEntity] = []
            self._extract_nes()

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

    def _assign_document_wide_id(self) -> None:
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

    def _extract_relations(self) -> None:
        tag2sid = {tag: sentence.sid for sentence in self.sentences for tag in sentence.tag_list()}
        for tag in self.tag_list():
            dtid = self.tag2dtid[tag]
            if tag.features.rels is None:
                logger.debug(f'Tag: "{tag.midasi}" has no relation tags.')
                continue
            pas = Pas(tag, dtid, tag2sid[tag])
            for rel in tag.features.rels:
                assert rel.ignore is False
                # extract PAS
                if rel.atype in self.target_cases:
                    if rel.sid is not None:
                        assert rel.tid is not None
                        arg_tag = self.sid2sentence[rel.sid].tag_list()[rel.tid]
                        pas.add_argument(rel.atype, arg_tag, rel.sid, self.tag2dtid[arg_tag], rel.target, rel.mode)
                    # exophora
                    else:
                        if rel.target not in ALL_EXOPHORS:
                            logger.warning(f'Unknown exophor: {rel.target}')
                            continue
                        elif rel.target not in self.target_exophors:
                            logger.info(f'Argument: {rel.target} ({rel.atype}) of {tag.midasi} is ignored.')
                            continue
                        pas.add_argument(rel.atype, None, None, None, rel.target, rel.mode)

                # extract coreference
                elif rel.atype in self.target_corefs:
                    self._add_corefs(tag2sid[tag], tag, rel)

                else:
                    logger.info(f'Relation type: {rel.atype} is ignored.')

            if pas.arguments:
                self._pas[tag] = pas

    def _add_corefs(self,
                    source_sid: str,
                    source_tag: Tag,
                    rel: Rel
                    ) -> None:
        source_dtid = self.tag2dtid[source_tag]
        if rel.sid is not None:
            target_tag = self.sid2sentence[rel.sid].tag_list()[rel.tid]
            target_dtid = self.tag2dtid[target_tag]
            if target_dtid >= source_dtid:
                logger.warning('Coreference with self or latter entity is found.')
                return
        else:
            target_tag = target_dtid = None
            if rel.target not in ALL_EXOPHORS:
                logger.warning(f'Unknown exophor: {rel.target}')
                return
            elif rel.target not in self.target_exophors:
                logger.info(f'Coreference with {rel.target} ({rel.atype}) of {source_tag.midasi} is ignored.')
                return

        if source_dtid in self._mentions:
            entity = self._entities[self._mentions[source_dtid].eid]
            if rel.sid is not None:
                if target_dtid in self._mentions:
                    target_entity = self._entities[self._mentions[target_dtid].eid]
                    logger.info(f'Merge entity {entity.eid} and {target_entity.eid}.')
                    entity.merge(target_entity)
                else:
                    target_mention = Mention(rel.sid, target_tag, target_dtid)
                    self._mentions[target_dtid] = target_mention
                    entity.add_mention(target_mention)
            # exophor
            else:
                if entity.exophor is None:
                    logger.info(f'Mark entity {entity.eid} as {rel.target}.')
                    entity.exophor = rel.target
                elif entity.exophor != rel.target:
                    if rel.mode != '':
                        entity.additional_exophor[rel.mode].append(rel.target)
                    else:
                        logger.warning(f'Overwrite entity {entity.eid} {entity.exophor} to {rel.target}.')
                        entity.exophor = rel.target
        else:
            source_mention = Mention(source_sid, source_tag, source_dtid)
            self._mentions[source_dtid] = source_mention
            if rel.sid is not None:
                if target_dtid in self._mentions:
                    entity = self._entities[self._mentions[target_dtid].eid]
                else:
                    target_mention = Mention(rel.sid, target_tag, target_dtid)
                    self._mentions[target_dtid] = target_mention
                    entity = self._create_entity()
                    entity.add_mention(target_mention)
            # exophor
            else:
                entity = self._create_entity(exophor=rel.target)
            entity.add_mention(source_mention)

    def _create_entity(self, exophor: Optional[str] = None) -> Entity:
        entity = Entity(len(self._entities), exophor=exophor)
        self._entities.append(entity)
        return entity

    def _extract_nes(self) -> None:
        for sentence in self.sentences:
            tag_list = sentence.tag_list()
            # tag.features = {'NE': 'LOCATION:ダーマ神殿'}
            for tag in tag_list:
                if 'NE' not in tag.features:
                    continue
                category, midasi = tag.features['NE'].split(':', maxsplit=1)
                if category not in NE_CATEGORIES:
                    logger.warning(f'unknown NE category: {category}')
                    continue
                mrph_list = [m for t in tag_list[:tag.tag_id + 1] for m in t.mrph_list()]
                mrph_span = self._find_mrph_span(midasi, mrph_list, tag)
                if mrph_span is None:
                    logger.warning(f'mrph span of "{midasi}" was not found in {sentence.sid}')
                    continue
                ne = NamedEntity(category, midasi, sentence, mrph_span, self.mrph2dmid)
                self.named_entities.append(ne)

    @staticmethod
    def _find_mrph_span(midasi: str,
                        mrph_list: List[Morpheme],
                        tag: Tag
                        ) -> Optional[range]:
        for i in range(len(tag.mrph_list())):
            end_mid = len(mrph_list) - i
            mrph_span = ''
            for mrph in reversed(mrph_list[:end_mid]):
                mrph_span = mrph.midasi + mrph_span
                if mrph_span == midasi:
                    return range(mrph.mrph_id, end_mid)
        return None

    @property
    def sentences(self) -> List[BList]:
        return list(self.sid2sentence.values())

    def bnst_list(self) -> List[Bunsetsu]:
        return [bnst for sentence in self.sentences for bnst in sentence.bnst_list()]

    def tag_list(self) -> List[Tag]:
        return [tag for sentence in self.sentences for tag in sentence.tag_list()]

    def mrph_list(self) -> List[Morpheme]:
        return [mrph for sentence in self.sentences for mrph in sentence.mrph_list()]

    def get_all_mentions(self) -> List[Mention]:
        return list(self._mentions.values())

    def get_all_entities(self) -> List[Entity]:
        return [entity for entity in self._entities if entity.mentions]

    def get_entity(self, tag: Tag) -> Optional[Entity]:
        entities = [e for e in self._entities for m in e.mentions if m.dtid == self.tag2dtid[tag]]
        if entities:
            assert len(entities) == 1
            return entities[0]
        else:
            return None

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
                      predicate: Tag,
                      relax: bool = False,
                      ) -> Dict[str, List[Argument]]:
        if predicate not in self._pas:
            return {}
        pas = self._pas[predicate]

        if relax is True:
            for case, args in self._pas[predicate].arguments.items():
                for arg in args:
                    entity = self.get_entity(self.dtid2tag[arg.dtid])
                    if entity is None:
                        continue
                    for mention in entity.mentions:
                        if mention.dtid == arg.dtid:
                            continue
                        pas.add_argument(case, mention.tag, mention.sid, mention.dtid, mention.midasi, '')

        return pas.arguments


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
