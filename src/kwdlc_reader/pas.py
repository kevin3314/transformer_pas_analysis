import logging
from typing import List, Dict, Optional
from collections import defaultdict

from pyknp import Tag, Rel


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Argument:
    """ 項に関する情報を保持するオブジェクト

    Attributes:
        sid (str): 文ID
        tid (int): 基本句ID
        midasi (str): 表記
        dtid (int): 文書レベル基本句ID
        dep_type (str): 係り受けタイプ ("overt", "dep", "intra", "inter", "exo")
        mode (str): モード
    """

    def __init__(self, rel: Rel, dtid: int, dep_type: str):
        self.sid = rel.sid
        self.tid = rel.tid
        self.midasi = rel.target
        self.dtid = dtid
        self.dep_type = dep_type
        self.mode = rel.mode

    # def __str__(self):
    #     return f'target: {self.target}, dtid: {self.dtid}, tid: {self.tid}'


class Pas:
    """ 述語項構造を扱うクラス

    """

    def __init__(self, tag: Tag, dtid: int, sid: str):
        # self.cfid = None  # always None (for compatibility)
        self.predicate = tag
        self.arguments: Dict[str, List[Argument]] = defaultdict(list)
        self.dtid = dtid
        self.sid = sid

    def add_argument(self, rel: Rel, tag: Optional[Tag], dtid: Optional[int]):
        assert tag.tag_id == rel.tid
        assert tag is not None or dtid is None
        dep_type = self._get_dep_type(self.predicate, tag, self.sid, rel.sid, rel.atype)
        self.arguments[rel.atype].append(Argument(rel, dtid, dep_type))

    @staticmethod
    def _get_dep_type(pred: Tag, arg: Tag, sid_pred: str, sid_arg: str, atype: str) -> str:
        if arg is not None:
            if arg in pred.children:
                if atype in arg.features:
                    return "overt"
                else:
                    return "dep"
            elif arg is pred.parent:
                return "dep"
            elif sid_arg == sid_pred:
                return "intra"
            else:
                return "inter"
        else:
            return "exo"
