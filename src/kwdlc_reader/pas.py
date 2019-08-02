import logging
from typing import List, Dict, Optional
from collections import defaultdict

from pyknp import Tag


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

    # def __init__(self, rel: Rel, dtid: Optional[int], dep_type: str):
    #     self.sid: str = rel.sid
    #     self.tid: Optional[int] = rel.tid
    #     self.midasi: str = rel.target
    #     self.dtid: Optional[int] = dtid
    #     self.dep_type: str = dep_type
    #     self.mode: str = rel.mode

    def __init__(self,
                 sid: str,
                 tid: int,
                 midasi: str,
                 dtid: Optional[int],
                 dep_type: str,
                 mode: str
                 ) -> None:
        self.sid = sid
        self.tid = tid
        self.midasi = midasi
        self.dtid = dtid
        self.dep_type = dep_type
        self.mode = mode

    # for test
    def __iter__(self):
        yield self.sid
        yield self.tid
        yield self.midasi
        yield self.dtid
        yield self.dep_type
        yield self.mode

    def __str__(self):
        return f'{self.midasi} (sid: {self.sid}, tid: {self.tid}, dtid: {self.dtid})'


class Pas:
    """ 述語項構造を扱うクラス

    """

    def __init__(self, tag: Tag, dtid: int, sid: str):
        self.predicate = tag
        self.arguments: Dict[str, List[Argument]] = defaultdict(list)
        self.dtid = dtid
        self.sid = sid

    def add_argument(self,
                     case: str,
                     tag: Optional[Tag],
                     sid: Optional[str],
                     dtid: Optional[int],
                     midasi: str,
                     mode: str,
                     ) -> None:
        if tag is None:
            assert sid is None and dtid is None
            tid = None
        else:
            tid = tag.tag_id
        if case in self.arguments:
            assert mode != ''
        dep_type = self._get_dep_type(self.predicate, tag, self.sid, sid, case)
        argument = Argument(sid, tid, midasi, dtid, dep_type, mode)
        self.arguments[case].append(argument)

    @staticmethod
    def _get_dep_type(pred: Tag, arg: Tag, sid_pred: str, sid_arg: str, atype: str) -> str:
        if arg is not None:
            if arg in pred.children:
                if arg.features.get('係', None) == atype.rstrip('？') + '格':
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
