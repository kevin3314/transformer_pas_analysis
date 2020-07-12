import io
import re
from collections import defaultdict
from logging import Logger
from typing import List, Optional, Dict, NamedTuple, Union, TextIO
from pathlib import Path

from pyknp import Tag
from kyoto_reader import Document, Pas, BaseArgument, Argument

from data_loader.dataset import InputFeatures, PASDataset


class PredictionKNPWriter:
    rel_pat = re.compile(r'<rel type="([^\s]+?)"(?: mode="([^>]+?)")? target="(.*?)"(?: sid="(.*?)" id="(.+?)")?/>')
    tag_pat = re.compile(r'^\+ -?\d+\w ?')
    case_analysis_pat = re.compile(r'<格解析結果:(.+?)>')

    def __init__(self,
                 dataset: PASDataset,
                 logger: Logger,
                 use_gold_overt: bool = True,
                 ) -> None:
        self.gold_arguments_sets: List[List[Dict[str, List[str]]]] = \
            [example.arguments_set for example in dataset.examples]
        self.all_features: List[InputFeatures] = dataset.features
        self.cases: List[str] = dataset.target_cases
        self.exophors: List[str] = dataset.target_exophors
        self.index_to_special: Dict[int, str] = {idx: token for token, idx in dataset.special_to_index.items()}
        self.coreference: bool = dataset.coreference
        self.bridging: bool = dataset.bridging
        self.dids = [example.doc_id for example in dataset.examples]
        self.did2document: Dict[str, Document] = {doc.doc_id: doc for doc in dataset.documents}
        self.dtid2cfid: Dict[int, str] = {}
        self.logger = logger
        self.use_gold_overt = use_gold_overt
        self.kc: bool = dataset.kc
        self.reader = dataset.reader

    def write(self,
              arguments_sets: List[List[List[int]]],
              destination: Union[Path, TextIO, None],
              skip_untagged: bool = True,
              ) -> List[Document]:
        """Write final predictions to the file."""

        did2examples = {did: tuple(ex) for did, *ex
                        in zip(self.dids, self.all_features, arguments_sets, self.gold_arguments_sets)}

        if isinstance(destination, Path):
            self.logger.info(f'Writing predictions to: {destination}')
            destination.mkdir(exist_ok=True)
        elif not (destination is None or isinstance(destination, io.TextIOBase)):
            self.logger.warning('invalid output destination')

        did2knps: Dict[str, List[str]] = defaultdict(list)
        for did, document in self.did2document.items():
            if did in did2examples:
                features, arguments_set, gold_arguments_set = did2examples[did]
                output_knp_lines = self._rewrite_rel(document.knp_string,
                                                     features,
                                                     arguments_set,
                                                     gold_arguments_set,
                                                     document)
            else:
                if skip_untagged:
                    continue
                output_knp_lines = []
                for line in document.knp_string.strip().split('\n'):
                    if line.startswith('+ '):
                        line = self.rel_pat.sub('', line)  # remove gold data
                    output_knp_lines.append(line)
            knps: List[str] = []
            buff = ''
            for knp_line in output_knp_lines:
                buff += knp_line + '\n'
                if knp_line.strip() == 'EOS':
                    knps.append(buff)
                    buff = ''
            if self.kc:
                orig_did, idx = did.split('-')
                if idx == '00':
                    did2knps[orig_did] += knps
                else:
                    did2knps[orig_did].append(knps[-1])
            else:
                did2knps[did] = knps
        documents_pred: List[Document] = []  # kc については元通り結合された文書のリスト
        for did, knps in did2knps.items():
            document_pred = Document(''.join(knps),
                                     did,
                                     self.reader.target_cases,
                                     self.reader.target_corefs,
                                     self.reader.relax_cases,
                                     extract_nes=False,
                                     use_pas_tag=False)
            documents_pred.append(document_pred)
            if destination is None:
                continue
            output_knp_lines = self._add_pas_analysis(document_pred.knp_string.split('\n'), document_pred)
            output_string = '\n'.join(output_knp_lines) + '\n'
            if isinstance(destination, Path):
                output_basename = did + '.knp'
                with destination.joinpath(output_basename).open('w') as writer:
                    writer.write(output_string)
            elif isinstance(destination, io.TextIOBase):
                destination.write(output_string)

        return documents_pred

    def _rewrite_rel(self,
                     knp_string: str,
                     features: InputFeatures,
                     arguments_set: List[List[int]],
                     gold_arguments_set: List[Dict[str, List[str]]],
                     document: Document,
                     ) -> List[str]:
        self.dtid2cfid = {}
        dtid2tag: Dict[int, Tag] = {dtid: tag for tag, dtid in document.tag2dtid.items()}
        dtid = 0
        sent_idx = 0
        output_knp_lines = []
        for line in knp_string.strip().split('\n'):
            if not line.startswith('+ '):
                output_knp_lines.append(line)
                if line == 'EOS':
                    sent_idx += 1
                continue

            # <格解析結果:>タグから overt case を見つける(inference用)
            match = self.case_analysis_pat.search(line)
            overt_dict = self._extract_overt_from_case_analysis_result(dtid, match, sent_idx, document)

            rel_removed: str = self.rel_pat.sub('', line)  # remove gold data
            assert '<rel ' not in rel_removed
            match = self.tag_pat.match(rel_removed)
            if match is not None:
                rel_idx = match.end()
                rel_string = self._rel_string(dtid2tag[dtid],
                                              arguments_set,
                                              gold_arguments_set,
                                              features,
                                              document,
                                              overt_dict)
                rel_inserted_line = rel_removed[:rel_idx] + rel_string + rel_removed[rel_idx:]
                output_knp_lines.append(rel_inserted_line)
            else:
                self.logger.warning(f'invalid format line: {line}')
                output_knp_lines.append(rel_removed)

            dtid += 1

        return output_knp_lines

    def _extract_overt_from_case_analysis_result(self,
                                                 dtid: int,
                                                 match: Optional,
                                                 sent_idx: int,
                                                 document: Document
                                                 ) -> Dict[str, int]:
        if match is None:
            return {}
        sentence = document.sentences[sent_idx]
        case_analysis_result = match.group(1)
        c0 = case_analysis_result.find(':')
        c1 = case_analysis_result.find(':', c0 + 1)
        cfid = case_analysis_result[:c0] + ':' + case_analysis_result[c0 + 1:c1]
        self.dtid2cfid[dtid] = cfid

        if case_analysis_result.count(':') < 2:  # For copula
            return {}

        overt_dict = {}
        for k in case_analysis_result[c1 + 1:].split(';'):
            items = k.split('/')
            caseflag = items[1]
            if caseflag == 'C':
                case = items[0]
                midasi = items[2]
                tid = int(items[3])
                target_tag: Tag = sentence.tag_list()[tid]
                for mrph in target_tag.mrph_list():
                    if mrph.midasi == midasi:
                        overt_dict[case] = document.mrph2dmid[mrph]
        return overt_dict

    def _rel_string(self,
                    tag: Tag,
                    arguments_set: List[List[int]],  # (max_seq_len, cases)
                    gold_arguments_set: List[Dict[str, List[str]]],  # (mrph_len, cases)
                    features: InputFeatures,
                    document: Document,
                    overt_dict: Dict[str, int],
                    ) -> str:
        rels: List[RelTag] = []
        dmid2tag = {document.mrph2dmid[mrph]: tag for tag in document.tag_list() for mrph in tag.mrph_list()}
        tag2sid = {tag: sentence.sid for sentence in document for tag in sentence.tag_list()}
        assert len(gold_arguments_set) == len(dmid2tag)
        relations: List[str] = self.cases + (['ノ'] * self.bridging) + (['='] * self.coreference)
        for mrph in tag.mrph_list():
            dmid = document.mrph2dmid[mrph]
            token_index = features.orig_to_tok_index[dmid]
            arguments: List[int] = arguments_set[token_index]
            # {'ガ': ['14%O', '著者'], 'ヲ': ['23%C'], 'ニ': ['NULL'], 'ガ２': ['NULL'], '=': []}
            gold_arguments: Dict[str, List[str]] = gold_arguments_set[dmid]
            assert len(relations) == len(arguments)
            assert relations == list(gold_arguments.keys())
            for (case, gold_args), argument in zip(gold_arguments.items(), arguments):
                # 助詞などの非解析対象形態素については gold_args が空になっている
                if not gold_args:
                    continue
                # overt(train/test)
                if self.use_gold_overt and any(arg.endswith('%C') for arg in gold_args):
                    # use gold data for overt case
                    prediction_dmid = int([arg for arg in gold_args if arg.endswith('%C')][0][:-2])
                # overt(inference)
                elif self.use_gold_overt and case in overt_dict:
                    prediction_dmid = int(overt_dict[case])
                else:
                    # special
                    if argument in self.index_to_special:
                        special_anaphor = self.index_to_special[argument]
                        if special_anaphor in self.exophors:  # exclude NULL and NA
                            rels.append(RelTag(case, special_anaphor, None, None))
                        continue
                    # [SEP] or [CLS]
                    elif features.tok_to_orig_index[argument] is None:
                        self.logger.warning("Choose [SEP] as an argument. Tentatively, change it to NULL.")
                        continue
                    # normal
                    else:
                        prediction_dmid = features.tok_to_orig_index[argument]
                prediction_tag: Tag = dmid2tag[prediction_dmid]
                target = ''.join(mrph.midasi for mrph in prediction_tag.mrph_list() if '<内容語>' in mrph.fstring)
                if not target:
                    target = prediction_tag.midasi
                rels.append(RelTag(case, target, tag2sid[prediction_tag], prediction_tag.tag_id))

        return ''.join(rel.to_string() for rel in rels)

    def _add_pas_analysis(self,
                          knp_lines: List[str],
                          document: Document,
                          ) -> List[str]:
        sid2index = {sid: i for i, sid in enumerate(document.sid2sentence.keys())}
        dtid2pas = {pas.dtid: pas for pas in document.pas_list()}
        dtid = 0
        output_knp_lines = []
        for line in knp_lines:
            if not line.startswith('+ '):
                output_knp_lines.append(line)
                continue
            if dtid in dtid2pas:
                pas_string = self._pas_string(dtid2pas[dtid], self.dtid2cfid.get(dtid, 'dummy:dummy'), sid2index)
                output_knp_lines.append(line + pas_string)
            else:
                output_knp_lines.append(line)

            dtid += 1

        return output_knp_lines

    def _pas_string(self,
                    pas: Pas,
                    cfid: str,
                    sid2index: Dict[str, int],
                    ) -> str:
        dtype2caseflag = {'overt': 'C', 'dep': 'N', 'intra': 'O', 'inter': 'O', 'exo': 'E'}
        case_elements = []
        for case in self.cases + (['ノ'] * self.bridging):
            items = ['-'] * 6
            items[0] = case
            args = pas.arguments[case]
            if args:
                arg: BaseArgument = args[0]
                items[1] = dtype2caseflag[arg.dep_type]  # フラグ (C/N/O/D/E/U)
                items[2] = arg.midasi  # 見出し
                if isinstance(arg, Argument):
                    items[3] = str(sid2index[pas.sid] - sid2index[arg.sid])  # N文前
                    items[4] = str(arg.tid)  # tag id
                    items[5] = str(list(arg.eids)[0])  # Entity ID
                else:
                    items[3] = str(-1)
                    items[4] = str(-1)
                    items[5] = str(list(arg.eids)[0])  # Entity ID
            else:
                items[1] = 'U'
            case_elements.append('/'.join(items))
        return f"<述語項構造:{cfid}:{';'.join(case_elements)}>"


class RelTag(NamedTuple):
    type_: str
    target: str
    sid: Optional[str]
    tid: Optional[int]

    def to_string(self):
        string = f'<rel type="{self.type_}" target="{self.target}"'
        if self.sid is not None:
            string += f' sid="{self.sid}" id="{self.tid}"'
        string += '/>'
        return string
