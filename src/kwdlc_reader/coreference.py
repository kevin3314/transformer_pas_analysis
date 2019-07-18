from typing import List

from pyknp import Tag


class Mention:
    def __init__(self, sid: str, tag: Tag, dtid: int, midasi: str):
        self.tag = tag
        self.sid = sid
        self.tid = tag.tag_id
        self.dtid = dtid
        self.midasi = midasi


class Entity:
    eid = -1

    def __init__(self, eid: int, exophor):
        self.eid: int = eid
        self.exophor = exophor
        self.mentions: List[Mention] = []

    def add_mention(self, mention: Mention):
        self.mentions.append(mention)

    @classmethod
    def initialize(cls):
        cls.eid = -1

    @classmethod
    def create(cls, exophor=None):
        cls.eid += 1
        return cls(cls.eid, exophor)
