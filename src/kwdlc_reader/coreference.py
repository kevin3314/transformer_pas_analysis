import logging
from typing import List, Dict, Optional

from pyknp import Morpheme

from kwdlc_reader.base_phrase import BasePhrase

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Mention(BasePhrase):
    """ 共参照における mention を扱うクラス

    Args:
        bp (BasePhrase): mention の基本句オブジェクト
        mrph2dmid (dict): 形態素とその文書レベルIDを紐付ける辞書

    Attributes:
        eids (dict): key: mode (AND, OR, ?), value: 対象の entity id
    """
    def __init__(self, bp: BasePhrase, mrph2dmid: Dict[Morpheme, int]):
        super().__init__(bp.tag, bp.dtid, bp.sid, mrph2dmid)
        self.eids: List[int] = []

    def get_siblings(self):  # TODO
        pass


class Entity:
    """ 共参照における entity を扱うクラス

    Args:
        eid (int): entity id
        exophor (str?): entity が外界照応の場合はその種類

    Attributes:
        eid (int): entity id
        exophor (str): 外界照応詞
        mentions (list): この entity への mention 集合
        taigen (bool): entityが体言かどうか
        yougen (bool): entityが用言かどうか
    """
    def __init__(self, eid: int, exophor: Optional[str] = None):
        self.eid: int = eid
        self.exophor: Optional[str] = exophor
        self.mentions: List[Mention] = []
        self.taigen: Optional[bool] = None
        self.yougen: Optional[bool] = None

    @property
    def is_special(self) -> bool:
        return self.exophor is not None

    def add_mention(self, mention: Mention) -> None:
        """この entity を参照する mention を追加する

        Args:
            mention (Mention): メンション
        """
        if mention in self.mentions:
            return
        mention.eids.append(self.eid)
        self.mentions.append(mention)
        # 全てのmentionの品詞が一致した場合のみentityに品詞を設定
        self.yougen = (self.yougen is not False) and ('用言' in mention.tag.features)
        self.taigen = (self.taigen is not False) and ('体言' in mention.tag.features)

    # merge 実行時は document._entities から other を忘れずに削除する
    # def merge(self, other: 'Entity') -> None:
    #     """entity 同士をマージする"""
    #     for mention in other.mentions:
    #         self.add_mention(mention)
    #     other.mentions = set()
        # self.exophors = list(set(self.exophors) | set(other.exophors))
        # logger.info(f'merge entity {other.eid} ({", ".join(other.exophors)}) '
        #             f'to entity {self.eid} ({", ".join(self.exophors)})')

    def __del__(self):
        for mention in self.mentions:
            mention.eids.remove(self.eid)
