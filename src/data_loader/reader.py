"""Functions to load KWDLC."""
import collections
import copy
import glob
import os
import re
import sys

from pyknp import BList, Morpheme
from pyknp import Tag
from typing import List, Dict, Tuple, Optional


class KyotoCorpus(object):
    """https://bitbucket.org/ku_nlp/seqzero/src/master/seqzero/corpus_reader.py"""

    def __init__(self, dirname: str, glob_pat: str = "*.knp"):
        self.file_paths = self.get_file_paths(dirname, glob_pat)

    def load_files(self):
        for file_path in self.file_paths:
            yield self.load_file(file_path)

    def load_file(self, file_path: str):
        with open(file_path) as f:
            sentences = [sentence for sentence in self.read_knp_stream(f)]
        if len(sentences) > 0:
            return Document(sentences, os.path.splitext(os.path.basename(file_path))[0])

    @staticmethod
    def read_knp_stream(f):
        buff = ""
        for line in f:
            buff += line
            if line == "EOS\n":
                yield BList(buff)
                buff = ""

    @staticmethod
    def get_file_paths(dirname: str, glob_pat: str):
        return sorted(glob.glob(os.path.join(dirname, glob_pat)))


class Document(object):

    def __init__(self, sentences: List[BList], doc_id: str):
        self.sentences = sentences
        self.doc_id = doc_id
        self.sid2sent = {sentence.sid: sentence for sentence in sentences}

        self.entities = {}  # eid -> list of mention keys
        self.mentions = {}  # mention key -> {is_special, key, dmid_range}

    def bnst_list(self):
        return [bnst for sentence in self.sentences for bnst in sentence.bnst_list()]

    def tag_list(self):
        return [tag for sentence in self.sentences for tag in sentence.tag_list()]

    def mrph_list(self):
        return [mrph for sentence in self.sentences for mrph in sentence.mrph_list()]

    def add_corefs(self, tobj1: dict, tobj2: dict):
        eid1 = self.add_mention(tobj1)
        eid2 = self.add_mention(tobj2)
        if eid1 is None:
            if eid2 is None:
                eid = len(self.entities)
                self.entities[eid] = [tobj1["key"], tobj2["key"]]
                tobj1["eid"] = tobj2["eid"] = eid
            else:
                self.entities[eid2].append(tobj1["key"])
                tobj1["eid"] = eid2
        else:
            if eid2 is None:
                self.entities[eid1].append(tobj2["key"])
                tobj2["eid"] = eid1
            else:
                if not eid1 == eid2:
                    sys.stderr.write(f"different clusters: {tobj1['key']}\t{tobj2['key']}\n")

    def add_mention(self, tobj: dict):
        if tobj["key"] in self.mentions:
            return self.mentions[tobj["key"]]["eid"]
        else:
            self.mentions[tobj["key"]] = tobj
            return None

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, sid: str):
        return self.sid2sent[sid]

    def __iter__(self):
        return iter(self.sentences)


class KyotoPASGoldExtractor:
    """https://bitbucket.org/ku_nlp/seqzero/src/master/seqzero/pas.py"""

    rel_pat = re.compile(
        r"<rel type=\"(?P<type>[^\s]+?)\"(?: mode=\"(?P<mode>[^>]+?)\")? "
        r"target=\"(?P<target>[^\s]+?)\"(?: sid=\"(?P<sid>[^\"]+?)\" id=\"(?P<tid>\d*?)\")?/>")

    def __init__(self, document: Document):
        self.document = document
        # self._assign_dtid()
        self._assign_dmid()
        self._check_corefs()

    # @staticmethod
    # def _assign_dtid():
    #     """Assign document-wide tag index starting from the beginning of the document."""
    #     idx = 0
    #     for sentence in document:
    #         # sentence.dtid_assigned = True
    #         for tag in sentence.tag_list():
    #             tag.dtid = idx
    #             idx += 1

    def _assign_dmid(self):
        """Assign document-wide mrph index starting from the beginning of the document."""
        idx = 0
        for sentence in self.document:
            sentence.dmid_assigned = True
            for mrph in sentence.mrph_list():
                mrph.dmid = idx
                idx += 1
            # idx += 1  # count <EOS>

    def _check_corefs(self):
        for sentence in self.document:
            sentence.coref_checked = True
            for tag in sentence.tag_list():
                match_list = self.rel_pat.findall(tag.fstring)
                if len(match_list) <= 0:
                    continue
                dmid_range = [tag.mrph_list()[0].dmid, tag.mrph_list()[-1].dmid]
                for match in match_list:
                    # sid and tid are empty strings in the case of special anaphors
                    _type, mode, anaphor, sid, tid = match
                    # ignore =≒ for now
                    if not _type.startswith("=") or _type == "=≒":
                        continue
                    if sid == "":
                        anaphor = re.sub("[0-9０-９]+$", "", anaphor)
                        if anaphor == "なし":
                            continue
                        if anaphor not in KyotoPASSpec.special_anaphors:
                            sys.stderr.write(f"unknown special anaphor\t{anaphor}\t({sentence.sid})\n")
                            continue
                        self.document.add_corefs(
                            {
                                "is_special": False,
                                "key": f"{sentence.sid}:{tag.tag_id}",
                                "dmid_range": dmid_range
                            },
                            {
                                "is_special": True,
                                "key": anaphor,
                                "dmid_range": [-1, -1]
                            })
                    else:
                        tid = int(tid)
                        if sid == sentence.sid:
                            if not (0 <= tid < len(sentence.tag_list())):
                                sys.stderr.write(f"tag out of range\t{tid}\t({sentence.sid})\n")
                                continue
                            tag2 = sentence.tag_list()[tid]
                            if tag == tag2:
                                sys.stderr.write(f"reference to self\t{tid}\t{sentence.sid}\n")
                                continue
                        else:
                            if sid not in self.document.sid2sent:
                                sys.stderr.write(f"unknown sid\t{sid}\t({sentence.sid})\n")
                                continue
                            sent2 = self.document.sid2sent[sid]
                            if not (0 <= tid < len(sent2.tag_list())):
                                sys.stderr.write(f"tag out of range\t{tid}\t({sent2.sid} <- {sentence.sid})\n")
                                continue
                            tag2 = sent2.tag_list()[tid]
                        dmid_range2 = [tag2.mrph_list()[0].dmid, tag2.mrph_list()[-1].dmid]
                        self.document.add_corefs(
                            {
                                "is_special": False,
                                "key": f"{sentence.sid}:{tag.tag_id}",
                                "dmid_range": dmid_range
                            },
                            {
                                "is_special": False,
                                "key": f"{sid}:{tid}",
                                "dmid_range": dmid_range2
                            })

    def extract_from_doc(self):
        pas_list: List[KyotoPAS] = []
        for sentence in self.document:
            pas_list += self.extract_from_sent(sentence)
        return pas_list

    def extract_from_sent(self, sentence: BList):
        pas_list: List[KyotoPAS] = []
        for tag in sentence.tag_list():
            obj = self._extract_from_tag(tag, sentence)
            if obj:
                pas_list.append(obj)
        return pas_list

    def _extract_from_tag(self, tag: Tag, sentence: BList):
        """Check if this is a predicate to be analyzed."""
        match_list = self.rel_pat.findall(tag.fstring)
        if len(match_list) <= 0:
            return None

        # set target cases depending on the type of basic phrase
        target_cases = KyotoPASSpec.target_cases_pred if "<用言:" in tag.fstring else KyotoPASSpec.target_cases_noun

        # set of tags that have dependency relations
        dep_ids: Dict[int, str] = self._mark_deps(tag)

        pas = KyotoPAS(sentence, tag)
        marked_tags = []
        for match in match_list:
            _type, mode, anaphor, sid, tid = match

            dmid = None
            dmid_range = None
            dep_type = ""
            tid = -1 if tid == "" else int(tid)
            if _type not in target_cases:
                continue
            if mode == "？":
                continue
            if sid == "":  # 外界照応
                assert tid == -1
                # 不特定:人１; TODO: fix full-/half-width inconsistency
                anaphor = re.sub("[0-9０-９]+$", "", anaphor)  # 不特定:人１ -> 不特定:人
                if anaphor == "なし":
                    continue
                else:
                    if anaphor not in KyotoPASSpec.special_anaphors:
                        sys.stderr.write(f"unknown special anaphor\t{anaphor}\t({sentence.sid})\n")
                        continue
                anaphor = "<" + anaphor + ">"
            elif sid == sentence.sid:  # 文内照応
                if not (0 <= tid < len(sentence.tag_list())):
                    sys.stderr.write(f"tag out of range\t{tid}\t({sentence.sid})\n")
                    continue
                if tid in dep_ids:
                    dep_type = dep_ids[tid]
                    marked_tags.append(tid)
                _tag = sentence.tag_list()[tid]
                dmid = self._get_dmid(_tag, sentence)
                dmid_range = [_tag.mrph_list()[0].dmid, _tag.mrph_list()[-1].dmid]
            else:  # 文間照応
                if sid not in self.document.sid2sent:
                    sys.stderr.write(f"unknown sid\t{sid}\t({sentence.sid})\n")
                    continue
                sentence2 = self.document.sid2sent[sid]
                if not (0 <= tid < len(sentence2.tag_list())):
                    sys.stderr.write("tag out of range\t{}\t({} <- {})\n".format(tid, sentence2.sid, sentence.sid))
                    continue
                _tag = sentence2.tag_list()[tid]
                dmid = self._get_dmid(_tag, sentence2)
                dmid_range = [_tag.mrph_list()[0].dmid, _tag.mrph_list()[-1].dmid]
            pas.add(_type, mode, anaphor, sid, tid, dmid, dmid_range, dep_type)
        if pas.rel_count <= 0:
            return None
        for ctag in tag.children:
            if ctag.tag_id not in marked_tags and "<係:無格>" in ctag.fstring:
                anaphor = ""
                dmid = None
                dmid_range = [ctag.mrph_list()[0].dmid, ctag.mrph_list()[-1].dmid]
                for mrph in ctag.mrph_list():
                    if KyotoPASSpec.is_content_word(mrph):
                        anaphor += mrph.midasi
                        dmid = mrph.dmid
                if dmid is None:
                    sys.stderr.write(f"functional dep?\t{ctag.get_surface()}\t({sentence.sid})\n")
                    dmid = ctag.mrph_list()[0].dmid
                pas.add("無", "", anaphor, sentence.sid, ctag.tag_id, dmid, dmid_range, "dep")
        return pas

    @staticmethod
    def _mark_deps(tag: Tag) -> Dict[int, str]:

        def check_overtness(tag_: Tag):
            for case in KyotoPASSpec.target_cases:
                if case in tag_.features:
                    return 'overt'
            return 'dep'

        dep_ids = {}
        if tag.parent is not None:
            dep_ids[tag.parent.tag_id] = 'dep'
            # coordination: 受診 し たい 病院 や 診療 所
            for ctag in tag.parent.children:
                if ctag.dpndtype in {"P", "I"}:
                    dep_ids[ctag.tag_id] = 'dep'
        # assumption: no 複合辞 (e.g. ～について)
        for ctag in tag.children:
            dep_ids[ctag.tag_id] = check_overtness(ctag)
            # coordination (TODO: partial coordination)
            # w201106-0000088108-3: 上川支庁は西を天塩山地・夕張山地、東を北見山地・石狩山地などに挟まれた盆地帯にある
            for cctag in ctag.children:
                if cctag.dpndtype in ("P", "I"):
                    dep_ids[cctag.tag_id] = check_overtness(cctag)
        return dep_ids

    @staticmethod
    def _get_dmid(tag: Tag, sentence: BList):
        # find content word mrph
        for mrph in tag.mrph_list():
            if "<内容語>" in mrph.fstring:
                return mrph.dmid
        sys.stderr.write(f"cannot find content word:\n{tag.spec()}\t({sentence.sid})\n")
        return tag.mrph_list()[0].dmid


class KyotoPAS:
    """https://bitbucket.org/ku_nlp/seqzero/src/master/seqzero/pas.py"""

    # tag types
    T_OTHER = 0
    T_PRED_FULL = 1  # normal pred
    T_PRED_PART = 2  # (not used for now)
    T_ARG_HEAD_FULL = 3  # head of an argument
    T_ARG_HEAD_PART = 4  # skip succeeding function words
    T_ARG_DEP = 5  # dependent of an argument

    def __init__(self, sentence: BList, tag: Tag):
        self.sentence = sentence
        self.tid = tag.tag_id

        # find dmid for the base phrase
        self.dmid = None
        for mrph in tag.mrph_list():
            if "<内容語>" in mrph.fstring:
                self.dmid = mrph.dmid
                break
        if self.dmid is None:
            sys.stderr.write(f"cannot find content word:\n{tag.spec()}\t({sentence.sid})\n")
            self.dmid = tag.mrph_list()[0].dmid

        self.rel_count = 0
        self.type_dic = collections.defaultdict(list)
        self.ttype_list = None
        self.ttype_list_args = {}

    def add(self, _type: str, mode: str, anaphor: str, sid: str, tid: int, dmid: int, dmid_range: List[int], dep_type: str):
        self.type_dic[_type].append((self.rel_count, mode, anaphor, sid, tid, dmid, dmid_range, dep_type))
        self.rel_count += 1

    def _mark_ttypes(self):
        """Classify each tag according to its role in the PAS."""

        def mark_dep_arg_tag(tag: Tag, rid: int):
            if "<係:文節内>" not in tag.fstring:
                return
            if self.ttype_list[tag.tag_id] in (self.T_PRED_FULL, self.T_PRED_PART):
                sys.stderr.write("pred id: {}\targ id: {}\n".format(pred_tag.tag_id, tag.tag_id))
                sys.stderr.write("mark overlap\t{}\t({})\n".format(tag.spec(), self.sentence.sid))
                sys.stderr.write(pred_tag.spec())
                return False
            self.ttype_list[tag.tag_id] = self.T_ARG_DEP
            self.ttype_list_args[rid][tag.tag_id] = True
            for ctag in reversed(tag.children):
                status = mark_dep_arg_tag(ctag, rid)
                if not status:
                    return False
            return True

        tag_list = self.sentence.tag_list()
        self.ttype_list = [self.T_OTHER] * len(tag_list)
        self.ttype_list[self.tid] = self.T_PRED_FULL

        pred_tag = tag_list[self.tid]
        marked_targets = []
        for _type, rel_list in self.type_dic.items():
            for rid, mode, anaphor, sid, tid, dmid, dmid_range, dep_type in rel_list:
                if dep_type == "":
                    continue
                self.ttype_list_args[rid] = [False] * len(tag_list)
                self.ttype_list_args[rid][tid] = True
                if tid > pred_tag.tag_id:
                    self.ttype_list[tid] = self.T_ARG_HEAD_PART
                else:
                    self.ttype_list[tid] = self.T_ARG_HEAD_FULL
                if "<形副名詞>" not in tag_list[tid].fstring:  # 形式名詞 の is part of the bunsetsu
                    marked_targets.append((rid, tag_list[tid]))
        # to avoid potential conflicts, mark dependent arg tags only after marking head arg tags
        for rid, tag in marked_targets:
            for ctag in reversed(tag.children):
                mark_dep_arg_tag(ctag, rid)

    def _is_all_intersent(self, tlist: List[tuple]):
        for rid, mode, anaphor, sid, tid, dep_type in tlist:
            if dep_type == "" or sid == "":
                return False
        return True

    def dump_tokens(self, document: Document):
        tokens, pos_list, type_list = [], [], []
        for sentence in document:
            for mrph in sentence.mrph_list():
                tokens.append(mrph.genkei)
                pos_list.append(mrph.hinsi + "-" + mrph.bunrui + "-" + mrph.katuyou2)
            type_list += self._dump_type_sent(sentence)
            tokens.append("<EOS>")
            pos_list.append("<EOS>")
            if sentence == self.sentence:
                break
        return tokens, pos_list, type_list

    def _dump_type_sent(self, sentence: BList):
        # O: outside
        # D: in a dependency relation
        # P: predicate itself
        #   C: content word
        #   Q: quasi-content word
        #   F: functional word
        # BO: boundary
        if sentence == self.sentence:
            if self.ttype_list is None:
                self._mark_ttypes()
        type_list = []
        for tid, tag in enumerate(sentence.tag_list()):
            if sentence == self.sentence:
                # descendants of an argument are labeled O
                if self.ttype_list[tid] == self.T_PRED_FULL:
                    a1 = "P"
                elif self.ttype_list[tid] in (self.T_ARG_HEAD_FULL, self.T_ARG_HEAD_PART):
                    a1 = "D"
                else:
                    a1 = "O"
            else:
                a1 = "O"
            for mrph in tag.mrph_list():
                if "<内容語>" in mrph.fstring:
                    a2 = "C"
                elif "<準内容語>" in mrph.fstring:
                    a2 = "Q"
                else:
                    a2 = "F"
                type_list.append(a1 + a2)
        type_list.append("BO")
        return type_list

    def dump_latent_ids(self, document: Document, intersent_args: str="dump", case_type: str="core"):
        offset = KyotoPASSpec.get_offset()
        if intersent_args in ("special" or "drop"):
            offset -= self.sentence.mrph_list()[0].dmid
        case_order = KyotoPASSpec.canonical_case_order if case_type == "all" \
            else KyotoPASSpec.canonical_core_case_order

        mrph_list = self.sentence.mrph_list()
        tokens = [mrph_list[self.dmid - mrph_list[0].dmid].genkei]  # add content word
        ids = [offset + self.dmid]
        eval_structs = {}
        for _type in case_order:
            # 未 and 連 in self.type_dic will be ignored
            if _type not in self.type_dic:
                continue
            if intersent_args == "drop" and self._is_all_intersent(self.type_dic[_type]):
                continue

            tokens.append("<" + _type + ">")
            ids.append(KyotoPASSpec.get_id(_type))
            eval_structs[_type] = []
            for rid, mode, anaphor, sid, tid, dmid, dmid_range, dep_type in self.type_dic[_type]:
                key = None
                astruct = None

                if not (dep_type == "" or sid == "") and intersent_args == "drop":
                    continue

                if mode == "AND":
                    tokens.append("<AND>")
                    ids.append(KyotoPASSpec.get_id("AND"))
                elif mode == "OR":
                    tokens.append("<OR>")
                    ids.append(KyotoPASSpec.get_id("OR"))

                if dep_type != "":
                    if dmid is None:
                        sys.stderr.write(f"dmid None (dep): {anaphor}\t({sid})\n")
                    tokens.append(mrph_list[dmid - mrph_list[0].dmid].genkei)
                    ids.append(offset + dmid)
                    astruct = {
                        "mode": mode,
                        "is_special": False,
                        "dmid": dmid,
                        "dmid_range": dmid_range,
                        "anaphor": anaphor,
                        "type": dep_type,
                    }
                    key = f"{sid}:{tid}"
                elif sid == self.sentence.sid:  # intra-sentence anaphor
                    if dmid is None:
                        sys.stderr.write(f"dmid None (intra): {anaphor}\t({sid})\n")
                    tokens.append(mrph_list[dmid - mrph_list[0].dmid].genkei)
                    ids.append(offset + dmid)
                    astruct = {
                        "mode": mode,
                        "is_special": False,
                        "dmid": dmid,
                        "dmid_range": dmid_range,
                        "anaphor": anaphor,
                        "type": "intra",
                    }
                    key = f"{sid}:{tid}"
                elif sid == "":  # special anaphor
                    assert (anaphor.startswith("<"))
                    anaphor_raw = anaphor[1:len(anaphor) - 1]
                    tokens.append(anaphor)
                    ids.append(KyotoPASSpec.get_id(anaphor_raw))
                    astruct = {
                        "mode": mode,
                        "is_special": True,
                        "anaphor": anaphor_raw,
                        "type": "exo",
                    }
                    key = anaphor_raw
                else:  # inter-sentence anaphor
                    if intersent_args == "dump":
                        # only use the single tag for now
                        if dmid is None:
                            sys.stderr.write("dmid None (inter): {anaphor}\t({sid})\n")
                        mrph_list2 = document[sid].mrph_list()
                        tokens.append(mrph_list2[dmid - mrph_list2[0].dmid].genkei)
                        ids.append(offset + dmid)
                        astruct = {
                            "mode": mode,
                            "is_special": False,
                            "dmid": dmid,
                            "dmid_range": dmid_range,
                            "anaphor": anaphor,
                            "type": "inter",
                        }
                        key = f"{sid}:{tid}"
                    elif intersent_args == "special":
                        tokens.append("<INTER>")
                        ids.append(KyotoPASSpec.get_id("INTER"))
                        astruct = {
                            "mode": mode,
                            "is_special": True,
                            "anaphor": "INTER",
                            "type": "inter",
                        }
                        key = None
                if key is not None and key in document.mentions:
                    self._augment_by_corefs(document, key, astruct)
                eval_structs[_type].append(astruct)
        tokens.append("<EOS>")
        ids.append(KyotoPASSpec.get_id("EOS"))
        return tokens, ids, eval_structs

    def _augment_by_corefs(self, document: Document, key: str, astruct: dict):
        astruct["corefs"] = []
        for key2 in document.entities[document.mentions[key]["eid"]]:
            if not key == key2:
                mention = copy.copy(document.mentions[key2])
                if mention["is_special"]:
                    mention["type"] = "exo"
                else:
                    sid, tid = mention["key"].rsplit(":", 1)
                    mention["type"] = "intra" if self.sentence.sid == sid else "inter"
                astruct["corefs"].append(mention)


class KyotoPASSpec(object):
    """https://bitbucket.org/ku_nlp/seqzero/src/master/seqzero/pas.py"""

    types = ["dep", "intra", "inter", "exo"]
    special_tokens = {
        "EOS": 0,
        "AND": 1,
        "OR": 2,
        "INTER": 3
    }
    special_anaphors = {
        "著者": 0,
        "読者": 1,
        "不特定:人": 2,  # 不特定:人１, 不特定:人２
        # "不特定:物": 3,
        # "不特定:状況": 4,
    }
    # the following special case marks are assigned by KNP
    # - "未": は, も, etc
    # - "連": rentai (for predicates)
    canonical_case_order = ["ガ２", "ガ", "ヲ", "ニ", "ト", "デ", "カラ", "ヨリ", "ヘ", "マデ", "マデニ", "時間", "外の関係", "無"]
    canonical_core_case_order = ["ガ２", "ガ", "ヲ", "ニ", "ノ"]
    canonical_core_case_order_en = ["ga2", "ga", "wo", "ni", "no"]
    target_cases = {
        "ガ２": 0,
        "ガ": 1,
        "ヲ": 2,
        "ニ": 3,
        "ノ": 4,
        # "ト",
        # "デ",
        # "カラ",
        # "ヨリ",
        # "ヘ",
        # "マデ",
        # "マデニ",
        # "時間",
        # "外の関係",
        # "無",  # <係:無格>
        # "ニタイシテ"
        # "ノ？"
        # "トイウ"
        # "修飾"
        # "="
        # "=構""
        # "=≒"
    }
    target_cases_pred = {
        "ガ２": 0,
        "ガ": 1,
        "ヲ": 2,
        "ニ": 3,
    }
    target_cases_noun = {
        "ノ": 4,
    }
    core_target_cases = {
        "ガ２": 0,
        "ガ": 1,
        "ヲ": 2,
        "ニ": 3,
    }
    _special_tokens_start = 0
    _special_anaphors_start = len(special_tokens)
    _target_cases_start = _special_anaphors_start + len(special_anaphors)
    _offset = _target_cases_start + len(target_cases)

    @classmethod
    def get_offset(cls):
        return cls._offset

    @classmethod
    def get_id(cls, token: str) -> int:
        if token in cls.special_tokens:
            return cls.special_tokens[token]
        if token in cls.special_anaphors:
            return cls._special_anaphors_start + cls.special_anaphors[token]
        if token in cls.target_cases:
            return cls._target_cases_start + cls.target_cases[token]
        raise NotImplementedError

    @classmethod
    def id2token(cls, _id: int):
        if not hasattr(cls, "_id2token"):
            cls._id2token = [None] * cls._offset
            for tlist in cls.special_tokens, cls.special_anaphors, cls.target_cases:
                for t in tlist:
                    cls._id2token[cls.get_id(t)] = t
        return cls._id2token[_id]

    @classmethod
    def is_content_word(cls, mrph: Morpheme):
        if re.compile("<(?:準?内容語|名詞相当語)>").search(mrph.fstring):
            return True
        else:
            return False
