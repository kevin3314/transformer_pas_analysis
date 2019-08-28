import logging
import copy
from typing import List, Dict, Optional
from collections import defaultdict

from pyknp import Tag, Morpheme


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Argument:
    """ 項に関する情報を保持するオブジェクト

    Attributes:
        sid (str): 文ID
        tid (int): 基本句ID
        midasi (str): 表記
        dtid (int): 文書レベル基本句ID
        dmid (int): 文書レベルの形態素ID
        dep_type (str): 係り受けタイプ ("overt", "dep", "intra", "inter", "exo")
        mode (str): モード
    """

    def __init__(self,
                 sid: Optional[str],
                 tid: Optional[int],
                 midasi: str,
                 dtid: Optional[int],
                 dmid: Optional[int],
                 dep_type: str,
                 mode: str
                 ) -> None:
        self.sid = sid
        self.tid = tid
        self.midasi = midasi
        self.dtid = dtid
        self.dmid = dmid
        self.dep_type = dep_type
        self.mode = mode

    # for test
    def __iter__(self):
        yield self.sid
        yield self.tid
        yield self.midasi
        yield self.dtid
        # yield self.dmid
        yield self.dep_type
        yield self.mode

    def __str__(self):
        return f'{self.midasi} (sid: {self.sid}, tid: {self.tid}, dtid: {self.dtid})'

    def __eq__(self, other: 'Argument'):
        if self.dtid is None and other.dtid is None:
            return self.midasi == other.midasi
        else:
            return self.dtid == other.dtid


class Pas:
    """ 述語項構造を保持するオブジェクト

    Args:
        tag (Tag): 述語の基本句
        dtid (int): 述語の文書レベル基本句ID
        sid (str): 述語の文ID
        mrph2dmid (dict): 形態素とその文書レベルIDを紐付ける辞書

    Attributes:
        predicate (Tag): 述語
        arguments (dict): 格と項
        dtid (int): 文書レベル基本句ID
        sid (str): 文ID
        dmid (int): 述語の中の内容語形態素の文書レベル形態素ID
    """

    def __init__(self, tag: Tag, dtid: int, sid: str, mrph2dmid: Dict[Morpheme, int]):
        self.predicate = tag
        self.arguments: Dict[str, List[Argument]] = defaultdict(list)
        self.dtid = dtid
        self.sid = sid
        self.mrph2dmid = mrph2dmid
        self.dmid = self._get_content_word(tag)

    def _get_content_word(self, tag: Tag, sid: str) -> int:
        for mrph in tag.mrph_list():
            if '<内容語>' in mrph.fstring:
                return self.mrph2dmid[mrph]
        else:
            logger.warning(f'cannot find content word: {tag.midasi}\t{sid}')
            return self.mrph2dmid[tag.mrph_list()[0]]

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
            dmid = None
        else:
            tid = tag.tag_id
            dmid = self._get_content_word(tag, sid)
        dep_type = self._get_dep_type(self.predicate, tag, self.sid, sid, case)
        argument = Argument(sid, tid, midasi, dtid, dmid, dep_type, mode)
        self.arguments[case].append(argument)

    @staticmethod
    def _get_dep_type(pred: Tag, arg: Tag, sid_pred: str, sid_arg: str, atype: str) -> str:
        if arg is not None:
            if arg in pred.children:
                if arg.features.get('係', None) == atype.rstrip('？') + '格':
                    return 'overt'
                else:
                    return 'dep'
            elif arg is pred.parent:
                return 'dep'
            elif sid_arg == sid_pred:
                return 'intra'
            else:
                return 'inter'
        else:
            return 'exo'

    def copy(self) -> 'Pas':
        # only for arguments, perform deepcopy
        new_obj = copy.copy(self)
        new_obj.arguments = copy.deepcopy(self.arguments)
        return new_obj
