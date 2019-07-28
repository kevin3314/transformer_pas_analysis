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
    eid = -1

    def __init__(self, eid: int, exophor: Optional[str]):
        self.eid: int = eid
        self.exophor = exophor
        self.mentions: List[Mention] = []
        self.additional_exophor: Dict[str, List[str]] = defaultdict(list)

    def add_mention(self, mention: Mention):
        mention.eid = self.eid
        self.mentions.append(mention)

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
