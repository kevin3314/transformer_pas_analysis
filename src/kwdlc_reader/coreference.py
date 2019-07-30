import logging
from typing import List, Dict, Optional
from collections import defaultdict

from pyknp import Tag


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Mention:
    def __init__(self, sid: str, tag: Tag, dtid: int):
        self.eid = None
        self.tag = tag
        self.sid = sid
        self.tid = tag.tag_id
        self.dtid = dtid
        self.midasi = tag.midasi


class Entity:
    def __init__(self, eid: int, exophor: Optional[str]):
        self.eid: int = eid
        self.exophor = exophor
        self.mentions: List[Mention] = []
        self.additional_exophor: Dict[str, List[str]] = defaultdict(list)
        self.taigen: bool = True
        self.yougen: bool = True

    def add_mention(self, mention: Mention):
        mention.eid = self.eid
        self.mentions.append(mention)
        # 全てのmentionの品詞が一致した場合のみentityに品詞を設定
        if '<用言:' not in mention.tag.fstring:
            self.yougen = False
        if '<体言>' not in mention.tag.fstring:
            self.taigen = False

    def merge(self, entity):
        for mention in entity.mentions:
            mention.eid = self.eid
        self.mentions += entity.mentions
        entity.mentions = []
        if entity.exophor is not None:
            if self.exophor is not None:
                logger.warning(f'Overwrite entity {self.eid} {self.exophor} to {entity.exophor}.')
                self.exophor = entity.exophor
            else:
                logger.info(f'Mark entity {self.eid} as {entity.exophor}.')
                self.exophor = entity.exophor
        else:
            if self.exophor is not None:
                logger.info(f'Mark entity {entity.eid} as {self.exophor}.')
                entity.exophor = self.exophor
