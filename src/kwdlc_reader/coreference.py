import logging
from typing import List, Dict, Optional
from collections import defaultdict

from pyknp import Tag


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Mention:
    """ 共参照における mention を扱うクラス

    Attributes:
        eid (int): 対象のentity id
        tag (Tag): mention の基本句
        sid (str): mention が存在する文の文ID
        tid (int): mention の基本句ID
        dtid (int): mention の文書レベル基本句ID
        midasi (str): mention の見出し
    """
    def __init__(self, sid: str, tag: Tag, dtid: int):
        self.eid = None
        self.tag = tag
        self.sid = sid
        self.tid = tag.tag_id
        self.dtid = dtid
        self.midasi = tag.midasi


class Entity:
    """ 共参照における entity を扱うクラス

    Args:
        eid (int): entity id
        exophor (str?): entity が外界照応の場合はその種類

    Attributes:
        eid (int): entity id
        exophors (list): 外界照応詞
        mentions (list): この entity への mention 集合
        taigen (bool): entityが体言かどうか
        yougen (bool): entityが用言かどうか
    """
    def __init__(self, eid: int, exophor: Optional[str]):
        self.eid: int = eid
        self.exophors: List[str] = [exophor] if exophor is not None else []
        self.mentions: List[Mention] = []
        # self.additional_exophor: Dict[str, List[str]] = defaultdict(list)
        self.taigen: bool = True
        self.yougen: bool = True
        self.mode = ''

    @property
    def is_special(self) -> bool:
        return len(self.exophors) > 0

    @property
    def exophor(self) -> Optional[str]:
        if len(self.exophors) == 0:
            return None
        if len(self.exophors) > 1:
            logger.warning('There exists other exophors set to this entity. Use first one.')
        return self.exophors[0]

    def add_mention(self, mention: Mention):
        mention.eid = self.eid
        self.mentions.append(mention)
        # 全てのmentionの品詞が一致した場合のみentityに品詞を設定
        if '<用言:' not in mention.tag.fstring:
            self.yougen = False
        if '<体言>' not in mention.tag.fstring:
            self.taigen = False

    def merge(self, other: 'Entity'):
        for mention in other.mentions:
            mention.eid = self.eid
        self.mentions += other.mentions
        other.mentions = []
        if other.exophor is not None:
            if self.exophor is not None and self.exophor != other.exophor:  # TODO: consider exophors
                logger.warning(f'Another exophor is set to entity{self.eid} ({self.exophors}) while merging.')
                self.exophors.extend(other.exophors)
            else:
                logger.info(f'Mark entity{self.eid} as {other.exophors}.')
                self.exophors = other.exophors
        else:
            if self.exophor is not None:
                logger.info(f'Mark entity {other.eid} as {self.exophors}.')
                other.exophors = self.exophors
