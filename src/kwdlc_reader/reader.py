import logging
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from collections import OrderedDict

from pyknp import BList, Bunsetsu, Tag, Morpheme, Rel
import mojimoji

from kwdlc_reader.pas import Pas, Argument
from kwdlc_reader.coreference import Mention, Entity
from kwdlc_reader.ne import NamedEntity
from kwdlc_reader.constants import ALL_CASES, CORE_CASES, ALL_EXOPHORS, ALL_COREFS, CORE_COREFS, NE_CATEGORIES


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)

"""
# TODO

# MEMO
- アノテーション基準には著者読者とのcorefは=:著者と書いてあるが、<rel>タグの中身はatype='=≒', target='著者'となっている
- corefタグは用言に対しても振られる
- Entityクラスに用言か体言かをもたせる？
- 用言かつ体言の基本句もある
- 前文/後文への照応もある(<rel type="=" target="後文"/>)
- 述語から項への係り受けもdep?
- BasePhraseクラス作っちゃう？
- 不特定:１などは他の不特定:１と同一の対象
- 不特定:1などの"1"で全角と半角の表記ゆれ
"""


class KWDLCReader:
    """ KWDLC(または Kyoto Corpus)の文書集合を扱うクラス

    Args:
        corpus_dir (Path): コーパスの存在するディレクトリ
        glob_pat (str): コーパスとして扱うファイルのパターン
        target_cases (list): 抽出の対象とする格
        target_corefs (list): 抽出の対象とする共参照関係(=など)
        target_exophors (list): 抽出の対象とする外界照応詞
        extract_nes (bool): 固有表現をコーパスから抽出するかどうか

    Attributes:
        target_cases (list): 抽出の対象とする格
        target_corefs (list): 抽出の対象とする共参照関係(=など)
        target_exophors (list): 抽出の対象とする外界照応詞
        extract_nes (bool): 固有表現をコーパスから抽出するかどうか
        file_paths (list): コーパスのファイルリスト
        did2path (dict): 文書IDとその文書のファイルを紐付ける辞書
    """
    def __init__(self,
                 corpus_dir: Path,
                 glob_pat: str = '*.knp',
                 target_cases: Optional[List[str]] = None,
                 target_corefs: Optional[List[str]] = None,
                 target_exophors: Optional[List[str]] = None,
                 extract_nes: bool = True,
                 ) -> None:
        self.target_cases: List[str] = self._get_target(target_cases, ALL_CASES, CORE_CASES, 'case')
        self.target_corefs: List[str] = self._get_target(target_corefs, ALL_COREFS, CORE_COREFS, 'coref')
        self.target_exophors: List[str] = self._get_target(target_exophors, ALL_EXOPHORS, ALL_EXOPHORS, 'exophor')
        self.extract_nes: bool = extract_nes

        self.file_paths: List[Path] = sorted(corpus_dir.glob(glob_pat))
        self.did2path: Dict[str, Path] = {path.stem: path for path in self.file_paths}

    @staticmethod
    def _get_target(input_: Optional[list], all_: list, default: list, type_: str) -> list:
        if input_ is None:
            return default
        target = []
        for item in input_:
            if item in all_:
                target.append(item)
            else:
                logger.warning(f'Unknown target {type_}: {item}')
        return target

    def get_doc_ids(self) -> List[str]:
        return list(self.did2path.keys())

    def process_document(self, doc_id: str) -> Optional['Document']:
        if doc_id not in self.did2path:
            logger.error(f'Unknown document id: {doc_id}')
            return None
        path = self.did2path[doc_id]
        return Document(path, self.target_cases, self.target_corefs, self.target_exophors, self.extract_nes)

    def process_documents(self, doc_ids: List[str]) -> Iterator[Optional['Document']]:
        for doc_id in doc_ids:
            yield self.process_document(doc_id)

    def process_all_documents(self) -> Iterator[Optional['Document']]:
        for path in self.file_paths:
            yield Document(path, self.target_cases, self.target_corefs, self.target_exophors, self.extract_nes)


class Document:
    """ KWDLC(または Kyoto Corpus)の1文書を扱うクラス

        Args:
            file_path (str): 文書ファイルのパス
            target_cases (list): 抽出の対象とする格
            target_corefs (list): 抽出の対象とする共参照関係(=など)
            target_exophors (list): 抽出の対象とする外界照応詞
            extract_nes (bool): 固有表現をコーパスから抽出するかどうか

        Attributes:
            doc_id (str): 文書ID(ファイル名から拡張子を除いたもの)
            target_cases (list): 抽出の対象とする格
            target_corefs (list): 抽出の対象とする共参照関係(=など)
            target_exophors (list): 抽出の対象とする外界照応詞
            extract_nes (bool): 固有表現をコーパスから抽出するかどうか
            sid2sentence (dict): 文IDと文を紐付ける辞書
            bnst2dbid (dict): 文節IDと文書レベルの文節IDを紐付ける辞書
            tag2dtid (dict): 基本句IDと文書レベルの基本句IDを紐付ける辞書
            mrph2dmid (dict): 形態素IDと文書レベルの形態素IDを紐付ける辞書
            dtid2tag (dict): 文書レベルの基本句IDと基本句を紐付ける辞書
            named_entities (list): 抽出した固有表現
        """
    def __init__(self,
                 file_path: Path,
                 target_cases: List[str],
                 target_corefs: List[str],
                 target_exophors: List[str],
                 extract_nes: bool,
                 ) -> None:
        self.doc_id = file_path.stem
        self.target_cases: List[str] = target_cases
        self.target_corefs: List[str] = target_corefs
        self.target_exophors: List[str] = target_exophors
        self.extract_nes: bool = extract_nes

        self.sid2sentence: Dict[str, BList] = OrderedDict()
        with file_path.open() as f:
            buff = ''
            for line in f:
                buff += line
                if line == 'EOS\n':
                    sentence = BList(buff)
                    self.sid2sentence[sentence.sid] = sentence
                    buff = ''

        self.bnst2dbid = {}
        self.tag2dtid = {}
        self.mrph2dmid = {}
        self._assign_document_wide_id()
        self.dtid2tag = {dtid: tag for tag, dtid in self.tag2dtid.items()}

        self._pas: Dict[int, Pas] = OrderedDict()
        self._mentions: Dict[int, Mention] = OrderedDict()
        self._entities: List[Entity] = []
        self._extract_relations()

        if extract_nes:
            self.named_entities: List[NamedEntity] = []
            self._extract_nes()

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
            pas = Pas(tag, dtid, tag2sid[tag], self.mrph2dmid)
            for rel in tag.features.rels:
                assert rel.ignore is False
                if rel.sid is not None and rel.sid not in self.sid2sentence:
                    logger.warning(f'sentence: {rel.sid} not found in {self.doc_id}')
                    continue
                rel.target = mojimoji.han_to_zen(rel.target, ascii=False)  # 不特定:人1 -> 不特定:人１
                # extract PAS
                if rel.atype in self.target_cases:
                    if rel.sid is not None:
                        assert rel.tid is not None
                        arg_tag = self._get_tag(rel.sid, rel.tid)
                        if arg_tag is None:
                            return
                        pas.add_argument(rel.atype, arg_tag, rel.sid, self.tag2dtid[arg_tag], rel.target, rel.mode)
                    # exophora
                    else:
                        if rel.target not in ALL_EXOPHORS:
                            logger.warning(f'Unknown exophor: {rel.target}\t{pas.sid}')
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
                self._pas[dtid] = pas

    def _add_corefs(self,
                    source_sid: str,
                    source_tag: Tag,
                    rel: Rel
                    ) -> None:
        source_dtid = self.tag2dtid[source_tag]
        if rel.sid is not None:
            target_tag = self._get_tag(rel.sid, rel.tid)
            if target_tag is None:
                return
            target_dtid = self.tag2dtid[target_tag]
            if target_dtid >= source_dtid:
                logger.warning(f'Coreference with self or latter mention\t{source_tag.midasi}\t{source_sid}.')
                return
        else:
            target_tag = target_dtid = None
            if rel.target not in ALL_EXOPHORS:
                logger.warning(f'Unknown exophor: {rel.target}\t{source_sid}')
                return
            elif rel.target not in self.target_exophors:
                logger.info(f'Coreference with {rel.target} ({rel.atype}) of {source_tag.midasi} is ignored.')
                return

        if source_dtid in self._mentions:
            entity = self._entities[self._mentions[source_dtid].eid]
            if rel.sid is not None:
                if target_dtid in self._mentions:
                    target_entity = self._entities[self._mentions[target_dtid].eid]
                    logger.info(f'Merge entity{entity.eid} and {target_entity.eid}.')
                    entity.merge(target_entity)
                else:
                    target_mention = Mention(rel.sid, target_tag, target_dtid)
                    self._mentions[target_dtid] = target_mention
                    entity.add_mention(target_mention)
            # exophor
            else:
                if len(entity.exophors) == 0:
                    logger.info(f'Mark entity{entity.eid} as {rel.target}.')
                    entity.exophors.append(rel.target)
                elif rel.target not in entity.exophors:
                    if rel.mode != '':
                        entity.exophors.append(rel.target)
                        entity.mode = rel.mode
                    else:
                        logger.warning(f'Overwrite entity{entity.eid} {entity.exophors} to {rel.target}\t{source_sid}.')
                        entity.exophors = [rel.target]
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

    def _get_tag(self, sid: str, tid: int) -> Optional[Tag]:
        tag_list = self.sid2sentence[sid].tag_list()
        if not (0 <= tid < len(tag_list)):
            logger.warning(f'tag out of range\t{tid}\t({sid})')
            return None
        return tag_list[tid]

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

    def pas_list(self) -> List[Pas]:
        return list(self._pas.values())

    def get_predicates(self) -> List[Tag]:
        return [pas.predicate for pas in self._pas.values()]

    def get_arguments(self,
                      predicate: Tag,
                      relax: bool = False,
                      ) -> Dict[str, List[Argument]]:
        predicate_dtid = self.tag2dtid[predicate]
        if predicate_dtid not in self._pas:
            return {}
        pas = self._pas[predicate_dtid].copy()

        if relax is True:
            for case, args in self._pas[predicate_dtid].arguments.items():
                for arg in args:
                    # exophor
                    if arg.dtid is None:
                        continue
                    entity = self.get_entity(self.dtid2tag[arg.dtid])
                    if entity is None:
                        continue
                    if entity.is_special:
                        for exophor in entity.exophors:
                            pas.add_argument(case, None, None, None, exophor, entity.mode)
                    for mention in entity.mentions:
                        if mention.dtid == arg.dtid:
                            continue
                        pas.add_argument(case, mention.tag, mention.sid, mention.dtid, mention.midasi, '')

        return pas.arguments

    def __len__(self):
        return len(self.sid2sentence)

    def __getitem__(self, sid: str):
        if sid in self.sid2sentence:
            return self.sid2sentence[sid]
        else:
            logger.error(f'sentence: {sid} is not in this document')
            return None

    def __iter__(self):
        return iter(self.sid2sentence.values())
