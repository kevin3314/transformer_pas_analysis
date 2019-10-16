import logging
from typing import List, Dict, Optional, Set

from pyknp import Morpheme

from kwdlc_reader import BasePhrase

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Mention(BasePhrase):
    """ 共参照における mention を扱うクラス

    Args:
        bp (BasePhrase): mention の基本句オブジェクト
        mrph2dmid (dict): 形態素とその文書レベルIDを紐付ける辞書

    Attributes:
        eid (int): 対象の entity id
    """
    def __init__(self, bp: BasePhrase, mrph2dmid: Dict[Morpheme, int]):
        super().__init__(bp.tag, bp.dtid, bp.sid, mrph2dmid)
        self.eid = None
        self.siblings: List[Mention] = []  # TODO


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
    def __init__(self, eid: int, exophor: Optional[str] = None):
        self.eid: int = eid
        self.exophors: List[str] = [exophor] if exophor is not None else []
        self.mentions: Set[Mention] = set()
        # self.additional_exophor: Dict[str, List[str]] = defaultdict(list)
        self.taigen: bool = True
        self.yougen: bool = True
        self.mode = ''

    @property
    def is_special(self) -> bool:
        return len(self.exophors) > 0

    def add_mention(self, mention: Mention) -> None:
        """この entity を参照する mention を追加する

        Args:
            mention (Mention): メンション
        """
        mention.eid = self.eid
        self.mentions.add(mention)
        # 全てのmentionの品詞が一致した場合のみentityに品詞を設定
        if '<用言:' not in mention.tag.fstring:
            self.yougen = False
        if '<体言>' not in mention.tag.fstring:
            self.taigen = False

    def merge(self, other: 'Entity') -> None:
        """entity 同士をマージする"""
        for mention in other.mentions:
            self.add_mention(mention)
        other.mentions = set()
        self.exophors = list(set(self.exophors) | set(other.exophors))
        logger.info(f'merge entity {other.eid} ({", ".join(other.exophors)}) '
                    f'to entity {self.eid} ({", ".join(self.exophors)})')
