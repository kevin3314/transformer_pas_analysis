import logging
from collections import defaultdict

from pyknp import Rel


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Argument:
    """ 項に関する情報を保持するオブジェクト

    詳しくは下記ページの「格要素側」の記述方法を参照
    http://nlp.ist.i.kyoto-u.ac.jp/index.php?KNP%2F%E6%A0%BC%E8%A7%A3%E6%9E%90%E7%B5%90%E6%9E%9C%E6%9B%B8%E5%BC%8F

    Attributes:
        sid (str): 文ID
        tid (int): 基本句ID
        midasi (str): 表記
        flag (str): フラグ (C, N, O, D, E, U)
    """

    def __init__(self, target: str, dtid: int, tid: int, mode: str):
        self.target = target
        self.dtid = dtid
        self.tid = tid
        self.mode = mode

    def __str__(self):
        return f'target: {self.target}, dtid: {self.dtid}, tid: {self.tid}'


class Pas:
    """ 述語項構造を扱うクラス

    """

    def __init__(self, dtid: int, tid: int):
        # self.cfid = None  # always None (for compatibility)
        self.arguments = defaultdict(list)
        self.dtid = dtid
        self.tid = tid

    def add_argument(self, atype: str, target: str, dtid: int, tid: int, mode: str):
        self.arguments[atype].append(Argument(target, dtid, tid, mode))
        # print(atype, end='')
        # print(Argument(target, dtid, tid))
