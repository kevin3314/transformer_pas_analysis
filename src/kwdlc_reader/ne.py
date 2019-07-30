from typing import Dict

from pyknp import BList, Morpheme


class NamedEntity:
    def __init__(self,
                 category: str,
                 midasi: str,
                 sentence: BList,
                 mid_range: range,
                 mrph2dmid: Dict[Morpheme, int]):
        self.category: str = category
        self.midasi: str = midasi
        self.sid: str = sentence.sid
        self.mid_range: range = mid_range
        dmid_start = mrph2dmid[sentence.mrph_list()[mid_range[0]]]
        dmid_end = mrph2dmid[sentence.mrph_list()[mid_range[-1]]]
        self.dmid_range: range = range(dmid_start, dmid_end + 1)
