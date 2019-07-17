from typing import List

from pyknp import Tag


class Entity:
    eid = 0

    def __init__(self, eid: int, exophor):
        self.eid: int = eid
        self.exophor = exophor
        self.mentions: List[Tag] = []

    def add_mention(self, mention: Tag):
        self.mentions.append(mention)

    @classmethod
    def initialize(cls):
        cls.eid = 0

    @classmethod
    def create(cls, exophor=None):
        cls.eid += 1
        return cls(cls.eid, exophor)
