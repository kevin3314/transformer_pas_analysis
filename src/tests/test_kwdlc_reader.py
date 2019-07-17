from typing import List

from pyknp import Tag

from kwdlc_reader import KWDLCReader


def test_pas(fixture_kwdlc_reader: KWDLCReader):
    predicates: List[Tag] = fixture_kwdlc_reader.get_predicates()
    assert len(predicates) == 13

