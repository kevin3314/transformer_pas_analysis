import copy
import logging
from typing import List, Dict
from collections import defaultdict
from abc import abstractmethod

from pyknp import Tag

from kwdlc_reader.base_phrase import BasePhrase
from kwdlc_reader.coreference import Mention

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

Predicate = BasePhrase


class BaseArgument:
    """全ての項の基底クラス"""
    def __init__(self, eid: int, dep_type: str, mode: str):
        self.eid: int = eid
        self.dep_type: str = dep_type
        self.mode: str = mode
        self.optional = False

    @property
    def is_special(self) -> bool:
        return self.dep_type == 'exo'

    @property
    @abstractmethod
    def midasi(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other) -> bool:
        raise NotImplementedError


class Argument(BasePhrase, BaseArgument):
    """ 文中に出現する(外界ではない)項を表すオブジェクト

    Args:
        mention (Mention): メンション
        midasi (str): 表記
        dep_type (str): 係り受けタイプ ("overt", "dep", "intra", "inter", "exo")
        mode (str): モード
    """

    def __init__(self,
                 mention: Mention,
                 midasi: str,
                 dep_type: str,
                 mode: str
                 ) -> None:
        super(Argument, self).__init__(mention, mention.dtid, mention.sid)  # initialize BasePhrase
        super(BasePhrase, self).__init__(mention.eid, dep_type, mode)  # initialize BaseArgument
        self.mention = mention
        self._midasi = midasi

    @property
    def midasi(self) -> str:
        """表記"""
        return self._midasi

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

    def __eq__(self, other: BaseArgument):
        return isinstance(other, Argument) and self.dtid == other.dtid


class SpecialArgument(BaseArgument):
    """外界を指す項を表すオブジェクト

    Args:
        exophor (str): 外界照応詞 (不特定:人など)
        eid (int): 外界照応詞のエンティティID
        mode (str): モード
    """
    def __init__(self, exophor: str, eid: int, mode: str):
        dep_type = 'exo'
        super().__init__(eid, dep_type, mode)
        self.exophor: str = exophor

    @property
    def midasi(self) -> str:
        return self.exophor

    def __eq__(self, other: BaseArgument):
        return isinstance(other, SpecialArgument) and self.exophor == other.exophor


# class Predicate(BasePhrase):
#     def __init__(self, tag, dtid, sid, mrph2dmid):
#         super().__init__(tag, dtid, sid, mrph2dmid)


class Pas:
    """ 述語項構造を保持するオブジェクト

    Args:
        pred_bp (BasePhrase): 述語となる基本句

    Attributes:
        predicate (Predicate): 述語
        arguments (dict): 格と項
    """

    def __init__(self, pred_bp: BasePhrase):
        # self.predicate = Predicate(pred_bp.tag, pred_bp.dtid, pred_bp.sid)
        self.predicate: Predicate = pred_bp
        self.arguments: Dict[str, List[BaseArgument]] = defaultdict(list)

    def add_argument(self, case: str, mention: Mention, target: str, mode: str):
        dep_type = self._get_dep_type(self.predicate.tag, mention.tag, self.predicate.sid, mention.sid, case)
        argument = Argument(mention, target, dep_type, mode)
        self.arguments[case].append(argument)

    @staticmethod
    def _get_dep_type(pred: Tag, arg: Tag, sid_pred: str, sid_arg: str, case: str) -> str:
        if arg in pred.children:
            if arg.features.get('係', None) == case.rstrip('？') + '格' or case in arg.features:
                return 'overt'
            else:
                return 'dep'
        elif arg is pred.parent:
            return 'dep'
        elif sid_arg == sid_pred:
            return 'intra'
        else:
            return 'inter'

    def add_special_argument(self, case: str, exophor: str, eid: int, mode: str):
        special_argument = SpecialArgument(exophor, eid, mode)
        self.arguments[case].append(special_argument)

    def set_previous_argument_optional(self, case: str, mode: str):
        if mode not in ('？', '?', 'AND'):
            logger.warning(f'target: なし found with mode: "{mode}"\t{self.sid}')
        if self.arguments[case]:
            arg = self.arguments[case][-1]
            arg.optional = True
            logger.info(f'marked {arg.midasi} as optional\t{self.sid}')
        else:
            logger.info(f'no preceding argument found. なし is ignored.\t{self.sid}')

    def copy(self) -> 'Pas':
        # only for arguments, perform deepcopy
        new_obj = copy.copy(self)
        new_obj.arguments = copy.deepcopy(self.arguments)
        return new_obj

    @property
    def dtid(self) -> int:
        return self.predicate.dtid

    @property
    def sid(self) -> str:
        return self.predicate.sid

    @property
    def dmid(self) -> int:
        """述語の中の内容語形態素の文書レベル形態素ID"""
        return self.predicate.dmid
