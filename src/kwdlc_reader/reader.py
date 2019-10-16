import io
import logging
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Union
from collections import OrderedDict

from pyknp import BList, Bunsetsu, Tag, Morpheme, Rel
import mojimoji

from kwdlc_reader.pas import Pas, Predicate, BaseArgument, Argument
from kwdlc_reader.coreference import Mention, Entity
from kwdlc_reader.ne import NamedEntity
from kwdlc_reader.constants import ALL_CASES, CORE_CASES, ALL_EXOPHORS, ALL_COREFS, CORE_COREFS, NE_CATEGORIES
from kwdlc_reader.base_phrase import BasePhrase


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

"""
# TODO

# MEMO
- corefタグは用言に対しても振られる
- 用言かつ体言の基本句もある
- 不特定:人１などは他の不特定:人１と同一の対象
- 不特定:人1などの"1"で全角と半角の表記ゆれ
- mode=?, target=なし の修飾的表現の対応(mode=ANDの場合も確認(w201106-0000104324-1))
"""


class KWDLCReader:
    """ KWDLC(または Kyoto Corpus)の文書集合を扱うクラス

    Args:
        source (Path or str): 入力ソース．Path オブジェクトを指定するとその場所のファイルを読む
        target_cases (list): 抽出の対象とする格
        target_corefs (list): 抽出の対象とする共参照関係(=など)
        # target_exophors (list): 抽出の対象とする外界照応詞
        extract_nes (bool): 固有表現をコーパスから抽出するかどうか
        glob_pat (str): コーパスとして扱うファイルのパターン
        use_pas_tag (bool): <rel >タグからではなく、<述語項構造:>タグから PAS を読むかどうか
    """
    def __init__(self,
                 source: Union[Path, str],
                 target_cases: Optional[List[str]],
                 target_corefs: Optional[List[str]],
                 # target_exophors: Optional[List[str]],
                 extract_nes: bool = True,
                 glob_pat: str = '*.knp',
                 use_pas_tag: bool = False,
                 ) -> None:
        if not (isinstance(source, Path) or isinstance(source, str)):
            raise TypeError(f'source must be an instance of Path or str: got {type(source)}')
        if isinstance(source, Path):
            if source.is_dir():
                logger.info(f'got directory path, use files in the directory as source files')
                file_paths: List[Path] = sorted(source.glob(glob_pat))
                self.did2source: Dict[str, Union[Path, str]] = OrderedDict((path.stem, path) for path in file_paths)
            else:
                logger.info(f'got file path, use this file as source file')
                self.did2source: Dict[str, Union[Path, str]] = {source.stem: source}
        else:
            logger.info(f'got string, use this string as source content')
            self.did2source: Dict[str, Union[Path, str]] = {'doc': source}

        self.target_cases: List[str] = self._get_target(target_cases, ALL_CASES, CORE_CASES, 'case')
        self.target_corefs: List[str] = self._get_target(target_corefs, ALL_COREFS, CORE_COREFS, 'coref')
        # self.target_exophors: List[str] = self._get_target(target_exophors, ALL_EXOPHORS, ALL_EXOPHORS, 'exophor')
        self.extract_nes: bool = extract_nes
        self.use_pas_tag: bool = use_pas_tag

    @staticmethod
    def _get_target(input_: Optional[list],
                    all_: list,
                    default: list,
                    type_: str
                    ) -> list:
        if input_ is None:
            return default
        target = []
        for item in input_:
            if item not in all_:
                logger.warning(f'Unknown target {type_}: {item}')
                continue
            # if type_ == 'exophor':
            #     for exo in all_:
            #         if exo.startswith(item) and exo not in target:
            #             target.append(exo)
            else:
                target.append(item)

        return target

    def get_doc_ids(self) -> List[str]:
        return list(self.did2source.keys())

    def process_document(self, doc_id: str) -> Optional['Document']:
        if doc_id not in self.did2source:
            logger.error(f'Unknown document id: {doc_id}')
            return None
        if isinstance(self.did2source[doc_id], Path):
            with self.did2source[doc_id].open() as f:
                input_string = f.read()
        else:
            input_string = self.did2source[doc_id]
        return Document(input_string,
                        doc_id,
                        self.target_cases,
                        self.target_corefs,
                        # self.target_exophors,
                        self.extract_nes,
                        self.use_pas_tag)

    def process_documents(self, doc_ids: List[str]) -> Iterator[Optional['Document']]:
        for doc_id in doc_ids:
            yield self.process_document(doc_id)

    def process_all_documents(self) -> Iterator['Document']:
        for doc_id in self.did2source.keys():
            yield self.process_document(doc_id)


class Document:
    """ KWDLC(または Kyoto Corpus)の1文書を扱うクラス

    Args:
        knp_string (str): 文書ファイルの内容(knp形式)
        doc_id (str): 文書ID
        target_cases (list): 抽出の対象とする格
        target_corefs (list): 抽出の対象とする共参照関係(=など)
        # target_exophors (list): 抽出の対象とする外界照応詞
        extract_nes (bool): 固有表現をコーパスから抽出するかどうか

    Attributes:
        doc_id (str): 文書ID(ファイル名から拡張子を除いたもの)
        target_cases (list): 抽出の対象とする格
        target_corefs (list): 抽出の対象とする共参照関係(=など)
        # target_exophors (list): 抽出の対象とする外界照応詞
        extract_nes (bool): 固有表現をコーパスから抽出するかどうか
        sid2sentence (dict): 文IDと文を紐付ける辞書
        bnst2dbid (dict): 文節IDと文書レベルの文節IDを紐付ける辞書
        tag2dtid (dict): 基本句IDと文書レベルの基本句IDを紐付ける辞書
        mrph2dmid (dict): 形態素IDと文書レベルの形態素IDを紐付ける辞書
        # dtid2tag (dict): 文書レベルの基本句IDと基本句を紐付ける辞書
        named_entities (list): 抽出した固有表現
    """
    def __init__(self,
                 knp_string: str,
                 doc_id: str,
                 target_cases: List[str],
                 target_corefs: List[str],
                 # target_exophors: List[str],
                 extract_nes: bool,
                 use_pas_tag: bool,
                 ) -> None:
        self.doc_id = doc_id
        self.target_cases: List[str] = target_cases
        self.target_corefs: List[str] = target_corefs
        # self.target_exophors: List[str] = target_exophors
        self.extract_nes: bool = extract_nes

        self.sid2sentence: Dict[str, BList] = OrderedDict()
        buff = []
        for line in knp_string.strip().split('\n'):
            buff.append(line)
            if line.strip() == 'EOS':
                sentence = BList('\n'.join(buff) + '\n')
                self.sid2sentence[sentence.sid] = sentence
                buff = []

        self.bnst2dbid = {}
        self.tag2dtid = {}
        self.mrph2dmid = {}
        self._assign_document_wide_id()
        # self.dtid2tag = {dtid: tag for tag, dtid in self.tag2dtid.items()}

        self._pas: Dict[int, Pas] = OrderedDict()
        self._mentions: Dict[int, Mention] = OrderedDict()
        self._entities: Dict[int, Entity] = {}
        if use_pas_tag:
            self._extract_pas()
        else:
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

    def _extract_pas(self) -> None:
        sid2idx = {sid: idx for idx, sid in enumerate(self.sid2sentence.keys())}
        # tag2sid = {tag: sentence.sid for sentence in self.sentences for tag in sentence.tag_list()}
        for tag in self.tag_list():
            if tag.pas is None:
                continue
            pas = Pas(BasePhrase(tag, self.tag2dtid[tag], tag.pas.sid, self.mrph2dmid))
            for case, arguments in tag.pas.arguments.items():
                for arg in arguments:
                    arg.midasi = mojimoji.han_to_zen(arg.midasi, ascii=False)  # 不特定:人1 -> 不特定:人１
                    # exophor
                    if arg.flag == 'E':
                        entity = self._create_entity(exophor=arg.midasi, eid=arg.eid)
                        pas.add_special_argument(case, arg.midasi, entity.eid, '')
                    else:
                        sid = self.sentences[sid2idx[arg.sid] - arg.sdist].sid
                        arg_bp = self._get_bp(sid, arg.tid)
                        mention = self._create_mention(arg_bp)
                        pas.add_argument(case, mention, arg.midasi, '')
            if pas.arguments:
                self._pas[pas.dtid] = pas

    def _extract_relations(self) -> None:
        tag2sid = {tag: sentence.sid for sentence in self.sentences for tag in sentence.tag_list()}
        for tag in self.tag_list():
            rels = self._extract_rel_tags(tag)
            if not rels:
                logger.debug(f'Tag: "{tag.midasi}" has no relation tags.')
                continue
            src_bp = BasePhrase(tag, self.tag2dtid[tag], tag2sid[tag], self.mrph2dmid)
            pas = Pas(src_bp)
            for rel in rels:
                if rel.sid is not None and rel.sid not in self.sid2sentence:
                    logger.warning(f'sentence: {rel.sid} not found in {self.doc_id}')
                    continue
                rel.target = mojimoji.han_to_zen(rel.target, ascii=False)  # 不特定:人1 -> 不特定:人１
                # rel.target = re.sub(r'^(不特定:(人|物|状況))[１-９]$', r'\1', rel.target)  # 不特定:人１ -> 不特定:人
                # extract PAS
                if rel.atype in self.target_cases:
                    if rel.sid is not None:
                        assert rel.tid is not None
                        arg_bp = self._get_bp(rel.sid, rel.tid)
                        if arg_bp is None:
                            continue
                        mention = self._create_mention(arg_bp)  # 項を発見したら同時に mention と entity を作成
                        pas.add_argument(rel.atype, mention, rel.target, rel.mode)
                    # exophora
                    else:
                        if rel.target == 'なし':
                            pas.set_previous_argument_optional(rel.atype, rel.mode)
                            continue
                        if rel.target not in ALL_EXOPHORS:
                            logger.warning(f'Unknown exophor: {rel.target}\t{pas.sid}')
                            continue
                        # elif rel.target not in self.target_exophors:
                        #     logger.info(f'Argument: {rel.target} ({rel.atype}) of {tag.midasi} is ignored.')
                        #     continue
                        entity = self._create_entity(rel.target)
                        pas.add_special_argument(rel.atype, rel.target, entity.eid, rel.mode)

                # extract coreference
                elif rel.atype in self.target_corefs:
                    self._add_corefs(src_bp, rel)

                else:
                    logger.info(f'Relation type: {rel.atype} is ignored.')

            if pas.arguments:
                self._pas[pas.dtid] = pas

    # to extract rels with mode: '?', rewrite initializer of pyknp Futures class
    @staticmethod
    def _extract_rel_tags(tag: Tag) -> List[Rel]:
        splitter = "><"
        rels = []
        spec = tag.fstring

        tag_start = 1
        tag_end = None
        while tag_end != -1:
            tag_end = spec.find(splitter, tag_start)
            if spec[tag_start:].startswith('rel '):
                rel = Rel(spec[tag_start:tag_end])
                if rel.atype is not None:
                    rels.append(rel)

            tag_start = tag_end + len(splitter)
        return rels

    def _add_corefs(self,
                    source_bp: BasePhrase,
                    rel: Rel
                    ) -> None:
        source_dtid = source_bp.dtid
        if rel.sid is not None:
            target_bp = self._get_bp(rel.sid, rel.tid)
            if target_bp is None:
                return
            # target_dtid = self.tag2dtid[target_tag]
            if target_bp.dtid >= source_dtid:
                logger.warning(f'Coreference with self or latter mention\t{source_bp.midasi}\t{source_bp.sid}.')
                return
        else:
            target_bp = None
            if rel.target not in ALL_EXOPHORS:
                logger.warning(f'Unknown exophor: {rel.target}\t{source_bp.sid}')
                return
            # elif rel.target not in self.target_exophors:
            #     logger.info(f'Coreference of {source_bp.midasi} with {rel.target} ignored.\t{source_bp.sid}')
            #     return

        if source_dtid in self._mentions:
            entity = self._entities[self._mentions[source_dtid].eid]
            if rel.sid is not None:
                if target_bp.dtid in self._mentions:
                    target_entity = self._entities[self._mentions[target_bp.dtid].eid]
                    logger.info(f'Merge entity{entity.eid} and {target_entity.eid}.')
                    entity.merge(target_entity)
                else:
                    target_mention = Mention(target_bp, self.mrph2dmid)
                    self._mentions[target_bp.dtid] = target_mention
                    entity.add_mention(target_mention)
                return
            # exophor
            else:
                if rel.target not in ('不特定:人', '不特定:物', '不特定:状況'):  # 共参照先が singleton entity だった時
                    target_entities = [e for e in self._entities.values() if rel.target in e.exophors]
                    if target_entities:
                        assert len(target_entities) == 1  # singleton entity が1つしかないことを保証
                        target_entity = target_entities[0]
                        logger.info(f'Merge entity{entity.eid} and {target_entity.eid}.')
                        target_entity.merge(entity)
                        return
                if len(entity.exophors) == 0:
                    logger.info(f'Mark entity{entity.eid} as {rel.target}.')
                    entity.exophors.append(rel.target)
                elif rel.target not in entity.exophors:
                    if rel.mode != '':
                        entity.exophors.append(rel.target)
                        entity.mode = rel.mode
                    else:
                        logger.warning(f'Overwrite entity {entity.exophors} to {rel.target}\t{source_bp.sid}.')
                        entity.exophors = [rel.target]
                return
        else:
            source_mention = Mention(source_bp, self.mrph2dmid)
            self._mentions[source_dtid] = source_mention
            if rel.sid is not None:
                target_mention = self._create_mention(target_bp)
                entity = self._entities[target_mention.eid]
            # exophor
            else:
                entity = self._create_entity(exophor=rel.target)
            entity.add_mention(source_mention)

    def _create_mention(self, bp: BasePhrase) -> Mention:
        """

        Args:
            bp (BasePhrase): 基本句
        Returns:
            Mention: メンション
        """
        if bp.dtid not in self._mentions:
            # new coreference cluster is made
            mention = Mention(bp, self.mrph2dmid)
            self._mentions[bp.dtid] = mention
            entity = self._create_entity()
            entity.add_mention(mention)
        else:
            mention = self._mentions[bp.dtid]
        return mention

    def _create_entity(self, exophor: Optional[str] = None, eid: Optional[int] = None) -> Entity:
        """

        Args:
            exophor (Optional[str]): 外界照応詞(optional)
            eid (Optional[int]): エンティティID(省略推奨)
        Returns:
             Entity: エンティティ
        """
        if eid is None:
            eid = len(self._entities)
        if exophor:
            if exophor not in ('不特定:人', '不特定:物', '不特定:状況'):  # exophor が singleton entity だった時
                entities = [e for e in self._entities.values() if exophor in e.exophors]
                # すでに singleton entity が存在した場合、新しい entity は作らずにその entity を返す
                if entities:
                    assert len(entities) == 1  # singleton entity が1つしかないことを保証
                    return entities[0]
        entity = Entity(eid, exophor=exophor)
        self._entities[eid] = entity
        return entity

    def _get_bp(self, sid: str, tid: int) -> Optional[BasePhrase]:
        """文IDと基本句IDから基本句を得る

        Args:
            sid (str): 文ID
            tid (int): 基本句ID

        Returns:
            Optional[BasePhrase]: 対応する基本句
        """
        tag_list = self.sid2sentence[sid].tag_list()
        if not (0 <= tid < len(tag_list)):
            logger.warning(f'tag out of range\t{tid}\t({sid})')
            return None
        tag = tag_list[tid]
        return BasePhrase(tag, self.tag2dtid[tag], sid, self.mrph2dmid)

    def _extract_nes(self) -> None:
        """KNP の tag を参照して文書中から固有表現を抽出する"""
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
        return [entity for entity in self._entities.values() if entity.mentions]

    def get_entity(self, tag: Tag) -> Optional[Entity]:
        entities = [e for e in self._entities.values() for m in e.mentions if m.dtid == self.tag2dtid[tag]]
        if entities:
            assert len(entities) == 1
            return entities[0]
        else:
            return None

    def pas_list(self) -> List[Pas]:
        return list(self._pas.values())

    def get_predicates(self) -> List[Predicate]:
        return [pas.predicate for pas in self._pas.values()]

    def get_arguments(self,
                      predicate: Predicate,
                      relax: bool = False,
                      include_optional: bool = False  # 「すぐに」などの修飾的な項も返すかどうか
                      ) -> Dict[str, List[BaseArgument]]:
        if predicate.dtid not in self._pas:
            return {}
        pas = self._pas[predicate.dtid].copy()
        if include_optional is False:
            for case in self.target_cases:
                pas.arguments[case] = list(filter(lambda a: a.optional is False, pas.arguments[case]))

        if relax is True:
            for case, args in self._pas[predicate.dtid].arguments.items():
                for arg in args:
                    entity = self._entities[arg.eid]
                    for exophor in entity.exophors:
                        if exophor == arg.midasi:
                            continue
                        pas.add_special_argument(case, exophor, entity.eid, entity.mode)
                    for mention in entity.mentions:
                        if isinstance(arg, Argument) and mention.dtid == arg.dtid:
                            continue
                        pas.add_argument(case, mention, mention.midasi, '')

        return pas.arguments

    def draw_tree(self, sid: str, fh) -> None:
        predicates: List[Predicate] = self.get_predicates()
        sentence: BList = self[sid]
        with io.StringIO() as string:
            sentence.draw_tag_tree(fh=string)
            tree_strings = string.getvalue().rstrip('\n').split('\n')
        tag_list = sentence.tag_list()
        assert len(tree_strings) == len(tag_list)
        for predicate in predicates:
            idx = predicate.tid
            arguments = self.get_arguments(predicate)
            tree_strings[idx] += '  '
            for case in self.target_cases:
                argument = arguments[case]
                arg = argument[0].midasi if argument else 'NULL'
                tree_strings[idx] += f'{arg}:{case} '
        print('\n'.join(tree_strings), file=fh)

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
